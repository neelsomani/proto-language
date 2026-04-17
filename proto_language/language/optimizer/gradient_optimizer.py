"""Gradient-based optimizer for differentiable sequence design."""

import copy
import logging
from collections.abc import Callable
from typing import Any, Literal, final

import numpy as np

from proto_language.base_config import BaseOptimizerConfig, ConfigField
from proto_language.language.core import Constraint, Construct, Generator, Optimizer, Segment
from proto_language.language.core.constraint import GradientResult
from proto_language.language.generator import PositionWeightGenerator
from proto_language.language.optimizer.optimizer_registry import optimizer
from proto_language.utils.gradients import MERGERS, GradientMergerName, adam_step, align_norms, normalize_gradient
from proto_language.utils.scheduling import SCHEDULES, ScheduleName

logger = logging.getLogger(__name__)


class GradientOptimizerConfig(BaseOptimizerConfig):
    """Configuration for gradient-based sequence optimization.

    Each GradientOptimizer runs one mode (fixed or ramping soft, with optional
    temperature annealing). Chain multiple in a ``Program`` for multi-phase
    pipelines like Germinal (logit phase → softmax phase).

    Attributes:
        num_results (int | None): Parallel optimization trajectories.
        num_steps (int): Number of gradient steps.
        lr (float): Base learning rate.
        beta1 (float): Adam first moment decay (0 = SGD).
        beta2 (float): Adam second moment decay (0 = SGD).
        initial_logit_bias (float): One-time logit bias at starting positions.
        soft_start (float): Soft blending at step 1 (0=hard, 1=softmax).
        soft_end (float): Soft blending at final step.
        temperature_start (float): Temperature at step 1.
        temperature_end (float): Temperature at final step.
        schedule (ScheduleName): Temperature decay schedule.
        merger (GradientMergerName): Gradient merging strategy.
        norm_alignment (Literal["none", "unit", "match_first"]): Per-constraint
            gradient normalization before merging.
        normalize_gradients (bool): Normalize merged gradient before update.
        normalize_mode (Literal["unit", "sqrt_length"]): Normalization formula.
            ``"unit"`` = L2 to magnitude 1. ``"sqrt_length"`` = Germinal-compatible
            ``g * sqrt(eff_L) / ||g||``.
        fixed_positions (list[int] | None): Positions to freeze during optimization.
        scale_lr_by_temperature (bool): Multiply LR by soft/temperature blending.
        min_lr_scale (float): Floor for effective LR scale.
        tracking_interval (int): Steps between progress snapshots.
        track_proposals (bool): Record per-proposal results in history.

    Note:
        Ramps use ``progress = step / num_steps`` with ``step`` starting at 1,
        so step 1 evaluates to ``start + (end - start) / num_steps`` (not exactly
        ``start``); step ``num_steps`` evaluates exactly to ``end``.
    """

    num_results: int | None = ConfigField(
        default=None,
        ge=1,
        title="Design Candidates",
        description="Candidate designs for this optimizer. Overrides program-level count.",
        advanced=True,
    )
    num_steps: int = ConfigField(
        default=1,
        ge=1,
        title="Num Steps",
        description="Number of gradient descent steps.",
    )
    lr: float = ConfigField(
        default=0.05,
        gt=0.0,
        title="Learning Rate",
        description="Base learning rate for gradient updates.",
    )
    beta1: float = ConfigField(
        default=0.9,
        ge=0.0,
        lt=1.0,
        title="Adam Beta1",
        description="Adam first moment decay. Set to 0 for SGD.",
        advanced=True,
    )
    beta2: float = ConfigField(
        default=0.999,
        ge=0.0,
        lt=1.0,
        title="Adam Beta2",
        description="Adam second moment decay. Set to 0 for SGD.",
        advanced=True,
    )
    initial_logit_bias: float = ConfigField(
        default=0.0,
        ge=0.0,
        title="Initial Logit Bias",
        description="One-time logit bias at starting-sequence positions.",
        advanced=True,
    )
    soft_start: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Soft Start",
        description="Soft blending at step 1 (0=hard logits, 1=full softmax).",
        advanced=True,
    )
    soft_end: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        title="Soft End",
        description="Soft blending at final step.",
        advanced=True,
    )
    temperature_start: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature Start",
        description="Softmax temperature at step 1.",
    )
    temperature_end: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature End",
        description="Softmax temperature at final step.",
    )
    schedule: ScheduleName = ConfigField(
        default="constant",
        title="Temperature Schedule",
        description="Temperature decay schedule across steps.",
    )
    merger: GradientMergerName = ConfigField(
        default="weighted_sum",
        title="Gradient Merger",
        description="Strategy for merging gradients from multiple constraints.",
    )
    norm_alignment: Literal["none", "unit", "match_first"] = ConfigField(
        default="none",
        title="Norm Alignment",
        description="Per-constraint gradient normalization before merging.",
        advanced=True,
    )
    normalize_gradients: bool = ConfigField(
        default=True,
        title="Normalize Gradients",
        description="Normalize the merged gradient before each update.",
        advanced=True,
    )
    normalize_mode: Literal["unit", "sqrt_length"] = ConfigField(
        default="unit",
        title="Normalize Mode",
        description="'unit' = L2 to 1.0. 'sqrt_length' = Germinal: g*sqrt(eff_L)/||g||.",
        advanced=True,
    )
    fixed_positions: list[int] | None = ConfigField(
        default=None,
        title="Fixed Positions",
        description="Sequence positions to freeze during optimization.",
        advanced=True,
    )
    scale_lr_by_temperature: bool = ConfigField(
        default=False,
        title="Scale LR by Temperature",
        description="Multiply LR by soft/temperature blending factor.",
        advanced=True,
    )
    min_lr_scale: float = ConfigField(
        default=0.0,
        ge=0.0,
        title="Min LR Scale",
        description="Floor for effective LR scale when temperature scaling is enabled.",
        advanced=True,
    )

    @classmethod
    def germinal_logit_preset(cls) -> "GradientOptimizerConfig":
        """Germinal VHH logit phase (Phase 1): soft ramps 0→1, temp fixed.

        Source: ``germinal/design/design.py`` — ``design_logits(iters=65, soft=0, e_soft=1)``.
        """
        return cls(
            num_steps=65,
            lr=0.1,
            beta1=0.0,
            beta2=0.0,
            soft_start=0.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=1.0,
            schedule="constant",
            merger="pcgrad",
            norm_alignment="match_first",
            normalize_mode="sqrt_length",
        )

    @classmethod
    def germinal_softmax_preset(cls) -> "GradientOptimizerConfig":
        """Germinal VHH softmax phase (Phase 2): soft=1, temp anneals 1→0.01.

        Source: ``germinal/design/design.py`` — ``design_soft(iters=35, e_temp=1e-2)``.
        """
        return cls(
            num_steps=35,
            lr=0.1,
            beta1=0.0,
            beta2=0.0,
            soft_start=1.0,
            soft_end=1.0,
            temperature_start=1.0,
            temperature_end=0.01,
            schedule="quadratic",
            merger="pcgrad",
            norm_alignment="match_first",
            normalize_mode="sqrt_length",
            scale_lr_by_temperature=True,
            min_lr_scale=0.01,
        )


@optimizer(
    key="gradient",
    label="Gradient Optimizer",
    config=GradientOptimizerConfig,
    description="Gradient-based sequence optimization via differentiable constraints",
)
@final
class GradientOptimizer(Optimizer):
    """Gradient-based optimizer for differentiable sequence design.

    Updates ``seq.logits`` directly on proposal sequences via gradient descent
    through differentiable constraints. Uses ``PositionWeightGenerator``
    to discretize logits into sequences for tracking and handoff.

    Chain multiple GradientOptimizers in a ``Program`` for multi-phase
    pipelines (e.g., Germinal logit phase → softmax phase).

    Attributes:
        config (GradientOptimizerConfig): Optimizer configuration.
    """

    # Class attribute required by OptimizerRegistry
    config_class = GradientOptimizerConfig

    def __init__(
        self,
        constructs: list[Construct],
        generators: list[Generator],
        constraints: list[Constraint],
        config: GradientOptimizerConfig,
        custom_logging: Callable[..., Any] | None = None,
        clear_tool_cache: int | bool | list[str] = 100 * 1024 * 1024,
    ) -> None:
        """Initialize the gradient optimizer.

        Args:
            constructs (list[Construct]): Constructs to optimize.
            generators (list[Generator]): Must contain exactly one PositionWeightGenerator.
            constraints (list[Constraint]): Must include at least one gradient-capable constraint.
            config (GradientOptimizerConfig): Configuration.
            custom_logging (Callable[..., Any] | None): Optional callback called at tracked steps
                (governed by ``tracking_interval``).
            clear_tool_cache (int | bool | list[str]): (int) Maximum size of cache in bytes, defaults to 100 MB.
                              (bool) Whether to clear the tool cache on each iteration.
                              (List[str]) Restrict clearing cache to a list of tool names.

        Raises:
            ValueError: If no gradient-capable constraints, generator validation fails,
                or a constraint's inputs do not include the target segment.
        """
        self.config = config
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            num_results=config.num_results,
            proposals_per_result=1,
            clear_tool_cache=clear_tool_cache,
            custom_logging=custom_logging,
            verbose=config.verbose,
            tracking_interval=config.tracking_interval,
            track_proposals=config.track_proposals,
            seed=config.seed,
        )

        self.num_steps: int = config.num_steps

        # Extract gradient-capable constraints
        self._gradient_constraints = [c for c in constraints if c.supports_gradient]
        if not self._gradient_constraints:
            raise ValueError("GradientOptimizer requires at least one gradient-capable constraint")

        # Validate generator
        pwg_generators = [g for g in generators if isinstance(g, PositionWeightGenerator)]
        if len(pwg_generators) != 1:
            raise ValueError(
                f"GradientOptimizer requires exactly one PositionWeightGenerator, got {len(pwg_generators)}"
            )
        self._generator = pwg_generators[0]

        # Validate fixed_positions are within the target segment's sequence length.
        if config.fixed_positions:
            seq_len = self._generator.segment.sequence_length
            out_of_bounds = [p for p in config.fixed_positions if p < 0 or p >= seq_len]
            if out_of_bounds:
                raise ValueError(f"fixed_positions {out_of_bounds} out of bounds for segment length {seq_len}")

        # Resolve which gradient index corresponds to the target segment in each constraint
        target_seg = self._generator.segment
        self._gradient_indices: list[int] = []
        for c in self._gradient_constraints:
            if target_seg not in c.inputs:
                raise ValueError(f"Constraint '{c.label}' inputs do not include the target segment")
            self._gradient_indices.append(c.inputs.index(target_seg))

        # Warn about non-gradient constraints that will be ignored
        skipped = [c.label for c in constraints if not c.supports_gradient]
        if skipped:
            logger.warning(f"GradientOptimizer ignoring non-gradient constraints: {skipped}")

        # Build merger and schedule
        self._merger = MERGERS[config.merger]()
        self._temperature_schedule = SCHEDULES[config.schedule](config.temperature_start, config.temperature_end)

        # Adam state (initialized in run)
        self._adam_m: list[np.ndarray] = []
        self._adam_v: list[np.ndarray] = []
        self._adam_t: list[int] = []

    def run(self) -> None:
        """Execute gradient optimization.

        Each step:
        1. Compute soft and temperature from linear/scheduled interpolation
        2. Compute gradients from all gradient-capable constraints
        3. Align norms, apply weights, merge (PCGrad/MGDA/weighted sum)
        4. Normalize merged gradient, zero fixed positions
        5. Update ``seq.logits`` via Adam/SGD
        6. At tracked steps: discretize via generator, save snapshot
        """
        self._prepare_run()
        assert self.num_results is not None  # noqa: S101 -- mypy type narrowing
        assert self.num_proposals is not None  # noqa: S101 -- mypy type narrowing
        target = self._generator.segment

        # Skip if logits carried from a previous stage.
        vocab = self._generator._ordered_vocab()
        for seq in target.proposal_sequences:
            if seq.logits is None:
                seq.logits = _init_logits(target.sequence_length, seq.sequence, self.config.initial_logit_bias, vocab)

        # Initialize Adam state
        self._adam_m = [np.zeros_like(seq.logits) for seq in target.proposal_sequences]
        self._adam_v = [np.zeros_like(seq.logits) for seq in target.proposal_sequences]
        self._adam_t = [0] * self.num_results

        # Discretize initial state and snapshot.
        self._generator.sample()
        self._proposal_outcomes = ["accepted"] * self.num_proposals
        self._proposal_energy_scores = list(self.energy_scores)
        self._save_progress_snapshot(time_step=0)

        if self.verbose:
            logger.info(
                f"GradientOptimizer: {self.num_results} trajectories, {self.config.num_steps} steps, "
                f"soft {self.config.soft_start}→{self.config.soft_end}, "
                f"temp {self.config.temperature_start}→{self.config.temperature_end}"
            )

        all_results: list[list[GradientResult]]
        for step in range(1, self.config.num_steps + 1):
            # 1. Compute soft and temperature from linear/scheduled interpolation
            progress = step / self.config.num_steps
            soft = self.config.soft_start + (self.config.soft_end - self.config.soft_start) * progress
            temp = self._temperature_schedule(step, self.config.num_steps)
            lr = self._effective_lr(temp, soft)

            # 2. Compute gradients from all gradient-capable constraints
            all_results = []
            for constraint in self._gradient_constraints:
                results = constraint.compute_gradient(temperature=temp, soft=soft)
                all_results.append(results)

            # 3-5. Merge gradients and update logits for each trajectory
            for k in range(self.num_results):
                self._update_trajectory(k, all_results, lr, target)

            # Report the same weighted objective that gradient descent is actually minimizing.
            self.energy_scores = [
                sum(c.weight * all_results[i][k].loss for i, c in enumerate(self._gradient_constraints))
                for k in range(self.num_results)
            ]
            self._proposal_outcomes = ["accepted"] * self.num_proposals
            self._proposal_energy_scores = list(self.energy_scores)

            # 6. At tracked steps: discretize, sync proposals→results, snapshot
            if step % self.tracking_interval == 0 or step == self.config.num_steps:
                self._generator.sample()
                for k in range(self.num_results):
                    target.result_sequences[k] = copy.deepcopy(target.proposal_sequences[k])
                self._save_progress_snapshot(time_step=step)
                self._log_progress(step, temp, lr)

            self._clear_tool_cache()

    # =============================================================================
    # Private helpers
    # =============================================================================

    def _log_progress(self, step: int, temperature: float, lr: float) -> None:
        """Log optimization progress at tracked steps."""
        if self.verbose:
            best = min(self.energy_scores)
            mean = float(np.mean(self.energy_scores))
            logger.info(
                f"Step {step:4d}/{self.config.num_steps} | "
                f"T={temperature:.4f} | lr={lr:.6f} | best={best:.6f} | mean={mean:.6f}"
            )

        if self.custom_logging:
            self.custom_logging(step, self.segments)

    def _update_trajectory(self, k: int, all_results: list[list[GradientResult]], lr: float, target: Segment) -> None:
        """Align, merge, normalize, and apply one gradient step for trajectory *k*."""
        grads = [all_results[c][k].gradient[self._gradient_indices[c]] for c in range(len(self._gradient_constraints))]
        weights = [c.weight for c in self._gradient_constraints]

        grads = align_norms(grads, self.config.norm_alignment)
        grads = [g * w for g, w in zip(grads, weights, strict=True)]
        merged = self._merger.merge(grads)

        if self.config.normalize_gradients:
            merged = normalize_gradient(merged, self.config.normalize_mode)
        if self.config.fixed_positions:
            merged[self.config.fixed_positions] = 0.0

        seq = target.proposal_sequences[k]
        assert seq.logits is not None  # noqa: S101 -- guaranteed by initialization
        seq.logits = adam_step(
            seq.logits,
            merged,
            lr,
            self._adam_m,
            self._adam_v,
            self._adam_t,
            k,
            self.config.beta1,
            self.config.beta2,
        )

    def _capture_initial_state(self) -> None:
        """Capture initial state preserving logits.

        Overrides the base implementation to pass ``include_logits=True`` so that
        on multi-stage re-run, ``_restore_initial_state`` can hand the continuous
        logits back to ``run()`` instead of silently re-initializing from scratch.
        """
        self._initial_state = {
            "segments": [
                {
                    "result": [seq.to_dict(include_logits=True) for seq in seg.result_sequences],
                    "proposals": [seq.to_dict(include_logits=True) for seq in seg.proposal_sequences],
                }
                for seg in self.segments
            ],
            "energy_scores": self.energy_scores.copy(),
        }

    def _effective_lr(self, temperature: float, soft: float) -> float:
        """Effective LR, optionally scaled: ``lr * ((1 - soft) + soft * temp)``."""
        lr = self.config.lr
        if self.config.scale_lr_by_temperature:
            lr_scale = (1.0 - soft) + (soft * temperature)
            lr_scale = max(lr_scale, self.config.min_lr_scale)
            lr *= lr_scale
        return lr


def _init_logits(num_positions: int, sequence: str, bias: float, vocab: list[str]) -> np.ndarray:
    """Return ``(num_positions, len(vocab))`` logits; ``sequence`` may be empty and only drives the bias."""
    if sequence and len(sequence) != num_positions:
        logger.warning(f"Sequence length {len(sequence)} != segment length {num_positions}; bias applied to overlap.")
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    logits = np.zeros((num_positions, len(vocab)), dtype=np.float64)
    for pos, char in enumerate(sequence[:num_positions]):
        idx = char_to_idx.get(char)
        if idx is not None:
            logits[pos, idx] = bias
    return logits
