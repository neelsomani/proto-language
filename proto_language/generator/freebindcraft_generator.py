"""FreeBindCraft generator: PyRosetta-free de novo protein binder design against a target.

A :class:`FreeBindCraftGenerator` produces full binder sequences from nothing, conditioned on a
target structure supplied via config. It declares ``input_type = STARTING_SEQUENCE`` with
``allows_empty_starting_sequence = True`` (the de-novo pattern, like ``RandomProteinGenerator``):
the binder is hallucinated against the frozen target rather than mutated from a starting sequence,
so the target is conditioning metadata, not a backbone to inverse-fold. Each ``sample()`` dispatches
the ``freebindcraft-design`` tool and writes one accepted design per proposal slot — the binder
sequence, its predicted target+binder complex, and the PyRosetta-free per-design metrics. Because
the predicted complex is a fresh output (not stale input), the generator overrides
``_preserve_structure_after_sample`` to keep it on each proposal.

Examples:
    >>> from proto_language.core import Segment
    >>> from proto_language.generator import FreeBindCraftGenerator, FreeBindCraftGeneratorConfig
    >>> gen = FreeBindCraftGenerator(
    ...     FreeBindCraftGeneratorConfig(target_structure="/path/target.pdb", target_chain="A")
    ... )
    >>> seg = Segment(length=80, sequence_type="protein")
    >>> gen.assign(seg)
    >>> gen.input_type.value  # 'starting_sequence'
"""

import logging
from typing import final

from proto_tools import (
    FreeBindCraftConfig,
    FreeBindCraftInput,
    Structure,
    run_freebindcraft_design,
)
from proto_tools.transforms.masking import MASK_TOKEN

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)


class FreeBindCraftGeneratorConfig(BaseConfig):
    """Configuration object for FreeBindCraftGenerator.

    Bundles the target specification with the full FreeBindCraft pipeline settings. The binder
    length and the number of designs are NOT set here — they are derived from the assigned segment
    (length = the segment's length; one design per proposal slot) at sample time.

    Attributes:
        target_structure (Structure): Target structure to design a binder against. Accepts a file path,
            raw PDB/CIF content string, or a ``Structure`` object.
        target_chain (str): Chain ID(s) of the frozen target (comma-separated for multi-chain).
        target_hotspot_residues (str | None): Comma-separated 1-indexed target residues the binder
            must contact (e.g. ``"1-10,56,78"``). ``None`` leaves the binder free on the surface.
        binder_name (str): Filename prefix recorded on each accepted design.
        design_config (FreeBindCraftConfig): Full FreeBindCraft pipeline settings (hallucination,
            ProteinMPNN, AlphaFold2, OpenMM relaxation, filter thresholds). The program seed
            overrides ``design_config.seed`` when the program is seeded.
    """

    target_structure: Structure = ConfigField(
        title="Target Structure",
        description="Target to design a binder against (file path, PDB/CIF content, or Structure).",
    )
    target_chain: str = ConfigField(
        default="A",
        title="Target Chain",
        description="Chain ID(s) of the frozen target (comma-separated for multi-chain).",
    )
    target_hotspot_residues: str | None = ConfigField(
        default=None,
        title="Target Hotspot Residues",
        description="Comma-separated 1-indexed target residues the binder must contact (e.g. '1-10,56').",
    )
    binder_name: str = ConfigField(
        default="binder",
        title="Binder Name",
        description="Filename prefix recorded on each accepted design.",
    )
    design_config: FreeBindCraftConfig = ConfigField(
        default_factory=FreeBindCraftConfig,
        title="Design Config",
        description="FreeBindCraft pipeline settings; the program seed overrides its seed field.",
    )


@generator(
    key="freebindcraft",
    label="FreeBindCraft Binder Design",
    config=FreeBindCraftGeneratorConfig,
    description="PyRosetta-free de novo protein binder design against a target (FreeBindCraft).",
    uses_gpu=True,
    tools_called=["freebindcraft-design"],
    supported_sequence_types=["protein"],
)
@final
class FreeBindCraftGenerator(Generator):
    """De novo protein binder generator using the PyRosetta-free FreeBindCraft pipeline.

    Hallucinates a binder against a frozen target with AlphaFold2, refines it with ProteinMPNN,
    re-validates with AlphaFold2, and scores the interface with OpenMM/FreeSASA/sc-rs. The assigned
    segment's length is the binder length, and its proposal count is the number of designs requested.
    Each accepted design's sequence, predicted complex, and metrics are written onto a proposal; if
    the pipeline returns fewer designs than requested, the proposal pool is truncated to what it
    produced, and an empty result raises.

    Example:
        >>> from proto_language.core import Segment
        >>> from proto_language.generator import FreeBindCraftGenerator, FreeBindCraftGeneratorConfig
        >>> config = FreeBindCraftGeneratorConfig(target_structure="/path/to/target.pdb", target_chain="A")
        >>> gen = FreeBindCraftGenerator(config)
        >>> segment = Segment(length=80, sequence_type="protein")
        >>> gen.assign(segment)
        >>> gen.sample()  # writes one accepted binder design per proposal
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE
    allows_empty_starting_sequence = True

    def __init__(self, config: FreeBindCraftGeneratorConfig) -> None:
        """Initialize the generator with the target spec and FreeBindCraft pipeline settings.

        Args:
            config (FreeBindCraftGeneratorConfig): Configuration object containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.target_structure = config.target_structure
        self.target_chain = config.target_chain
        self.target_hotspot_residues = config.target_hotspot_residues
        self.binder_name = config.binder_name
        self.design_config = config.design_config

    def _preserve_structure_after_sample(self) -> bool:
        """Keep each proposal's predicted complex — it is a fresh design output, not stale input."""
        return True

    def _sample(self) -> None:
        """Design binders against the target and write one accepted design per proposal.

        Seeds the (length-only) proposal pool with placeholder residues so the STARTING_SEQUENCE
        validation passes, then dispatches ``freebindcraft-design`` with ``binder_lengths`` fixed to
        the segment length and ``number_of_final_designs`` set to the proposal count.

        Raises:
            RuntimeError: If the pipeline returns no accepted designs.
        """
        segment = self.segment
        target_length = segment.sequence_length

        # De-novo: seed empty proposals so STARTING_SEQUENCE validation passes; overwritten below.
        if not any(seq.sequence for seq in segment.proposal_sequences):
            placeholder = MASK_TOKEN * target_length
            for sequence in segment.proposal_sequences:
                sequence.sequence = placeholder

        self._validate_generator()
        num_proposals = segment.num_proposals

        inputs = FreeBindCraftInput(
            target_pdb=self.target_structure,
            target_chain=self.target_chain,
            target_hotspot_residues=self.target_hotspot_residues,
            binder_lengths=(target_length, target_length),
            binder_name=self.binder_name,
            number_of_final_designs=num_proposals,
        )
        next_seed = self._next_seed()
        update = {"seed": next_seed} if next_seed is not None else {}
        tool_config = self.design_config.model_copy(update=update)

        result = run_freebindcraft_design(inputs=inputs, config=tool_config)
        designs = result.designs
        if not designs:
            raise RuntimeError(
                "FreeBindCraft produced no accepted designs. Raise design_config.max_trajectories "
                "or relax design_config.filter_overrides."
            )
        if len(designs) < num_proposals:
            logger.warning(
                "FreeBindCraft returned %d designs for %d requested proposals on segment %r; "
                "truncating the proposal pool to the designs produced.",
                len(designs),
                num_proposals,
                segment.label or "unlabeled",
            )
            segment.proposal_sequences = segment.proposal_sequences[: len(designs)]

        key = self._spec.key
        for proposal, design in zip(segment.proposal_sequences, designs, strict=True):
            proposal.sequence = design.binder_sequence
            proposal.structure = design.structure
            proposal._generator_metadata[key] = dict(design.metrics.items())
