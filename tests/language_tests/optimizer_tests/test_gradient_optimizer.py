"""Tests for the GradientOptimizer, multi-stage pipelines, and GPU integration."""

import numpy as np
import pytest
from pydantic import BaseModel

from proto_language.language.core import Constraint, Construct, Program, Segment
from proto_language.language.core.constraint import GradientResult
from proto_language.language.generator import (
    PositionWeightGenerator,
    PositionWeightGeneratorConfig,
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)
from proto_language.language.optimizer import (
    GradientOptimizer,
    GradientOptimizerConfig,
    MCMCOptimizer,
    MCMCOptimizerConfig,
)


class _Cfg(BaseModel):
    """Empty config for mocks."""


def _backward(inputs: tuple, *, config: BaseModel, temperature: float = 1.0, **kwargs: object) -> GradientResult:
    """Gradient pushes logits toward alanine (column 0)."""
    logits = inputs[0].logits
    target = np.zeros_like(logits)
    target[:, 0] = 1.0
    grad = logits - target
    return GradientResult(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={"temperature": temperature})


def _backward_toward_D(inputs: tuple, *, config: BaseModel, **kwargs: object) -> GradientResult:
    """Gradient pushes logits toward aspartate (column 2 in ACDEF…) — conflicts with _backward."""
    logits = inputs[0].logits
    target = np.zeros_like(logits)
    target[:, 2] = 1.0
    grad = logits - target
    return GradientResult(gradient=(grad,), loss=float(np.mean(grad**2)), metrics={})


def _scorer(input_sequences: list[tuple], config: BaseModel) -> list[float]:
    return [sum(c != "A" for c in seq.sequence) / max(len(seq.sequence), 1) for (seq,) in input_sequences]


_scorer._constraint_supported_sequence_types = ["protein"]  # type: ignore[attr-defined]
_scorer._constraint_num_input_sequences_per_tuple = 1  # type: ignore[attr-defined]


def _make(num_steps: int = 5, num_results: int = 1, seed: int = 42, **kw: object) -> tuple[GradientOptimizer, Segment]:
    seg = Segment(sequence="EVQLV", sequence_type="protein")
    construct = Construct([seg])
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(
        inputs=[seg], backward=_backward, backward_config=_Cfg(), function=_scorer, function_config=_Cfg(), label="mock"
    )
    defaults: dict[str, object] = {"lr": 0.1, "beta1": 0.0, "beta2": 0.0}
    defaults.update(kw)
    cfg = GradientOptimizerConfig(num_results=num_results, num_steps=num_steps, seed=seed, **defaults)
    opt = GradientOptimizer(constructs=[construct], generators=[gen], constraints=[con], config=cfg)
    return opt, seg


def _make_gradient_stage(seg: Segment, construct: Construct, label: str, **kw: object) -> GradientOptimizer:
    """Build one GradientOptimizer stage with fresh generator/constraint."""
    gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
    gen.assign(seg)
    con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label=label)
    defaults: dict[str, object] = {"num_results": 1, "num_steps": 5, "lr": 0.1, "beta1": 0.0, "beta2": 0.0}
    defaults.update(kw)
    return GradientOptimizer(
        constructs=[construct], generators=[gen], constraints=[con], config=GradientOptimizerConfig(**defaults)
    )


class TestConfig:
    def test_germinal_logit_preset(self) -> None:
        cfg = GradientOptimizerConfig.germinal_logit_preset()
        assert cfg.num_steps == 65 and cfg.soft_start == 0.0 and cfg.soft_end == 1.0
        assert cfg.temperature_end == 1.0 and cfg.merger == "pcgrad" and cfg.normalize_mode == "sqrt_length"

    def test_germinal_softmax_preset(self) -> None:
        cfg = GradientOptimizerConfig.germinal_softmax_preset()
        assert cfg.num_steps == 35 and cfg.temperature_end == 0.01 and cfg.schedule == "quadratic"
        assert cfg.scale_lr_by_temperature is True and cfg.min_lr_scale == 0.01


class TestRun:
    def test_produces_results_with_logits(self) -> None:
        opt, seg = _make()
        opt.run()
        assert seg.result_sequences[0].sequence != ""
        assert seg.result_sequences[0].logits is not None
        assert seg.result_sequences[0].logits.shape == (5, 20)

    def test_loss_decreases(self) -> None:
        opt, _ = _make(num_steps=20, lr=0.05, tracking_interval=5)
        opt.run()
        early = opt.history[1]["results"][0]["energy_score"]
        last = opt.history[-1]["results"][0]["energy_score"]
        assert last < early

    def test_multiple_trajectories(self) -> None:
        opt, seg = _make(num_results=3)
        opt.run()
        assert len(seg.result_sequences) == 3
        assert len(opt.energy_scores) == 3

    def test_rerun_determinism(self) -> None:
        opt, seg = _make()
        opt.run()
        first = seg.result_sequences[0].sequence
        opt.run()
        assert seg.result_sequences[0].sequence == first

    def test_capture_initial_state_preserves_logits(self) -> None:
        """_capture_initial_state must round-trip logits so multi-stage re-run preserves the handoff."""
        opt, seg = _make()
        # Simulate the multi-stage handoff: proposals arrive with logits set.
        for s in seg.proposal_sequences:
            s.logits = np.full((len(s.sequence), 20), 1.5, dtype=np.float64)
        opt._capture_initial_state()
        captured = opt._initial_state["segments"][0]["proposals"][0]  # type: ignore[index]
        assert "logits" in captured, "logits missing from captured state — multi-stage re-run would lose the handoff"
        # Round-trip through restore must preserve the values.
        for s in seg.proposal_sequences:
            s.logits = None
        opt._restore_initial_state()
        restored = seg.proposal_sequences[0].logits
        assert restored is not None
        assert np.allclose(restored, 1.5)


class TestSchedules:
    def test_soft_and_temperature_forwarded(self) -> None:
        """Soft ramps linearly; temperature decays via schedule — both forwarded to backward."""
        received_soft: list[float] = []
        received_temp: list[float] = []

        def bwd(
            inputs: tuple, *, config: BaseModel, soft: float = 1.0, temperature: float = 1.0, **kwargs: object
        ) -> GradientResult:
            received_soft.append(soft)
            received_temp.append(temperature)
            return GradientResult(gradient=(np.zeros_like(inputs[0].logits),), loss=0.0, metrics={})

        seg = Segment(sequence="AA", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=bwd, backward_config=_Cfg(), label="t")
        opt = GradientOptimizer(
            constructs=[Construct([seg])],
            generators=[gen],
            constraints=[con],
            config=GradientOptimizerConfig(
                num_results=1,
                num_steps=4,
                soft_start=0.0,
                soft_end=1.0,
                temperature_start=1.0,
                temperature_end=0.1,
                schedule="linear",
            ),
        )
        opt.run()
        assert received_soft[0] == pytest.approx(0.25) and received_soft[-1] == pytest.approx(1.0)
        assert received_temp[0] > received_temp[-1]


class TestFixedPositions:
    def test_logits_unchanged_at_fixed(self) -> None:
        opt, seg = _make(num_steps=20, lr=1.0, initial_logit_bias=5.0, fixed_positions=[0, 4])
        opt.run()
        from proto_language.language.core.sequence import PROTEIN_AMINO_ACIDS

        aa_idx = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}
        result = seg.result_sequences[0].logits
        assert result is not None
        assert result[0, aa_idx["E"]] == pytest.approx(5.0)
        assert result[4, aa_idx["V"]] == pytest.approx(5.0)
        assert result[1, aa_idx["V"]] != pytest.approx(5.0)


class TestValidation:
    def test_no_gradient_constraints(self) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], function=_scorer, function_config=_Cfg())
        with pytest.raises(ValueError, match="gradient-capable"):
            GradientOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )

    def test_wrong_generator(self) -> None:
        seg = Segment(sequence="AA", sequence_type="protein")
        gen = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen.assign(seg)
        con = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="m")
        with pytest.raises(ValueError, match="PositionWeightGenerator"):
            GradientOptimizer(
                constructs=[Construct([seg])],
                generators=[gen],
                constraints=[con],
                config=GradientOptimizerConfig(num_steps=1),
            )


class TestHistory:
    def test_snapshot_count(self) -> None:
        opt, _ = _make(num_steps=10, tracking_interval=5)
        opt.run()
        assert len(opt.history) == 3  # step 0, 5, 10


# =============================================================================
# E2E: Multi-stage pipeline and multi-constraint merging
# =============================================================================


class TestMultiStage:
    def test_logit_handoff_across_stages(self) -> None:
        """Stage 2 reads logits produced by stage 1, not re-initialized zeros."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = _make_gradient_stage(seg, construct, "s1", soft_start=0.0, soft_end=1.0)
        opt2 = _make_gradient_stage(seg, construct, "s2", temperature_start=1.0, temperature_end=0.1, schedule="linear")

        program = Program(optimizers=[opt1, opt2], num_results=1)
        program.run_stage(0)
        stage1_logits = seg.result_sequences[0].logits
        assert stage1_logits is not None
        assert stage1_logits[0, 0] > stage1_logits[0, 5]  # optimized toward A

        program.run_stage(1)
        stage2_logits = seg.result_sequences[0].logits
        assert stage2_logits is not None
        assert stage2_logits[0, 0] > stage1_logits[0, 0]  # further optimized

    def test_gradient_then_mcmc(self) -> None:
        """GradientOptimizer → MCMCOptimizer pipeline completes with valid scores."""
        seg = Segment(sequence="EVQLV", sequence_type="protein")
        construct = Construct([seg])
        opt1 = _make_gradient_stage(seg, construct, "g", seed=42)

        gen2 = RandomProteinGenerator(RandomProteinGeneratorConfig())
        gen2.assign(seg)
        con2 = Constraint(inputs=[seg], function=_scorer, function_config=_Cfg(), label="m")
        opt2 = MCMCOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[con2],
            config=MCMCOptimizerConfig(num_results=1, num_steps=5),
        )

        program = Program(optimizers=[opt1, opt2], num_results=1)
        program.run()
        assert len(opt2.energy_scores) == 1
        assert opt2.energy_scores[0] < float("inf")


class TestMergers:
    def test_pcgrad_differs_from_weighted_sum(self) -> None:
        """PCGrad conflict projection produces different logits than naive weighted sum."""
        results = {}
        for merger in ("pcgrad", "weighted_sum"):
            seg = Segment(sequence="GGGGG", sequence_type="protein")
            construct = Construct([seg])
            gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
            gen.assign(seg)
            con_a = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="A")
            con_d = Constraint(inputs=[seg], backward=_backward_toward_D, backward_config=_Cfg(), label="D")
            opt = GradientOptimizer(
                constructs=[construct],
                generators=[gen],
                constraints=[con_a, con_d],
                config=GradientOptimizerConfig(
                    num_results=1,
                    num_steps=10,
                    lr=0.1,
                    beta1=0.0,
                    beta2=0.0,
                    merger=merger,
                    normalize_gradients=False,
                    seed=42,
                ),
            )
            opt.run()
            results[merger] = seg.result_sequences[0].logits
        assert not np.allclose(results["pcgrad"], results["weighted_sum"])

    def test_constraint_weight_dominance(self) -> None:
        """High-weight constraint dominates in logit space."""
        seg = Segment(sequence="GGGGG", sequence_type="protein")
        construct = Construct([seg])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con_a = Constraint(inputs=[seg], backward=_backward, backward_config=_Cfg(), label="A", weight=10.0)
        con_d = Constraint(inputs=[seg], backward=_backward_toward_D, backward_config=_Cfg(), label="D", weight=0.01)
        opt = GradientOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[con_a, con_d],
            config=GradientOptimizerConfig(
                num_results=1, num_steps=20, lr=0.1, beta1=0.0, beta2=0.0, merger="weighted_sum", seed=42
            ),
        )
        opt.run()
        logits = seg.result_sequences[0].logits
        assert logits is not None
        assert logits[:, 0].mean() > logits[:, 2].mean()  # A dominates D


class TestExport:
    def test_to_dataframe_and_fasta(self) -> None:
        opt, _ = _make()
        opt.run()
        df = opt.to_dataframe(table="sequences")
        assert len(df) > 0
        fasta = opt.to_fasta()
        assert fasta.startswith(">")


# =============================================================================
# GPU integration tests
# =============================================================================


@pytest.mark.uses_gpu
@pytest.mark.slow
class TestGradientOptimizerGPU:
    """GPU-dependent integration tests with real differentiable constraints."""

    def test_ablang_gradient_descent(self) -> None:
        """AbLang VHH gradient reduces naturalness loss over 10 steps."""
        from proto_language.language.constraint.differentiable import ablang_vhh_gradient_backward
        from proto_language.language.constraint.differentiable.ablang_naturalness_gradient_constraint import (
            AbLangGradientConstraintConfig,
        )

        seg = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein")
        construct = Construct([seg])
        gen = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen.assign(seg)
        con = Constraint(
            inputs=[seg],
            backward=ablang_vhh_gradient_backward,
            backward_config=AbLangGradientConstraintConfig(),
            label="ablang",
        )
        opt = GradientOptimizer(
            constructs=[construct],
            generators=[gen],
            constraints=[con],
            config=GradientOptimizerConfig(
                num_results=1,
                num_steps=10,
                lr=0.1,
                beta1=0.0,
                beta2=0.0,
                tracking_interval=5,
            ),
        )
        opt.run()

        early = opt.history[1]["results"][0]["energy_score"]
        final = opt.history[-1]["results"][0]["energy_score"]
        assert final < early
        assert seg.result_sequences[0].logits is not None

    def test_two_stage_germinal_preset(self) -> None:
        """Two-stage Germinal pipeline (logit -> softmax) with AbLang preserves logits across stages."""
        from proto_language.language.constraint.differentiable import ablang_vhh_gradient_backward
        from proto_language.language.constraint.differentiable.ablang_naturalness_gradient_constraint import (
            AbLangGradientConstraintConfig,
        )

        seg = Segment(sequence="EVQLVESGGGLVQPGGSLRL", sequence_type="protein")
        construct = Construct([seg])

        # Stage 1: logit phase (reduced steps for test speed)
        gen1 = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen1.assign(seg)
        con1 = Constraint(
            inputs=[seg],
            backward=ablang_vhh_gradient_backward,
            backward_config=AbLangGradientConstraintConfig(),
            label="ablang_s1",
        )
        cfg1 = GradientOptimizerConfig.germinal_logit_preset()
        cfg1.num_steps = 10  # reduced from 65
        opt1 = GradientOptimizer(
            constructs=[construct],
            generators=[gen1],
            constraints=[con1],
            config=cfg1,
        )

        # Stage 2: softmax phase
        gen2 = PositionWeightGenerator(PositionWeightGeneratorConfig())
        gen2.assign(seg)
        con2 = Constraint(
            inputs=[seg],
            backward=ablang_vhh_gradient_backward,
            backward_config=AbLangGradientConstraintConfig(),
            label="ablang_s2",
        )
        cfg2 = GradientOptimizerConfig.germinal_softmax_preset()
        cfg2.num_steps = 5  # reduced from 35
        opt2 = GradientOptimizer(
            constructs=[construct],
            generators=[gen2],
            constraints=[con2],
            config=cfg2,
        )

        program = Program(optimizers=[opt1, opt2], num_results=1)
        program.run()

        result = seg.result_sequences[0]
        assert result.logits is not None
        assert result.sequence != "EVQLVESGGGLVQPGGSLRL"
