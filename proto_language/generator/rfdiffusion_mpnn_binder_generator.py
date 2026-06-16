"""De novo protein binder design by chaining RFdiffusion3 and an inverse-folding model.

This generator designs a protein binder against a fixed target. Each ``sample()``
call (1) generates binder backbones docked to the target with RFdiffusion3 — using
the target structure, an auto-built contig that keeps the target chains fixed and
appends a binder of the segment's length, and optional epitope hotspots — then
(2) designs each backbone's binder-chain sequence with the selected inverse-folding
model (``inverse_folding``) while holding the target chains fixed as context.

The inverse-folding model is configurable: ``"proteinmpnn"`` (the default) sees only
the protein backbone, so it conditions on protein targets but is blind to non-protein
atoms; ``"ligandmpnn"`` additionally conditions on ligand, nucleotide (DNA/RNA), and
metal atoms in the target, so it can design binders against protein, DNA, or RNA targets
(and condition on any ligand/metal cofactors present in the structure). A small-molecule
*target* still needs RFdiffusion3's ``ligand`` field — see ``_build_contig``.
Each tool's knobs live in a nested ``RFdiffusion3Config`` and the inverse-folding
config selected by ``inverse_folding`` (``proteinmpnn_config`` or ``ligandmpnn_config``);
the inputs, seed, and backbone-count fields the generator owns are injected at sample
time.

The binder sequence is written to each proposal and the RFdiffusion3 target+binder
complex to ``proposal.structure``. Because the binder is created from a length-only
segment, this is a de-novo generator: it declares ``input_type = STARTING_SEQUENCE``
with ``allows_empty_starting_sequence = True`` (its registry category is therefore
``mutation``, like ``RandomProteinGenerator``).

Examples:
    >>> from proto_language.generator import (
    ...     RFdiffusionMPNNBinderGenerator,
    ...     RFdiffusionMPNNBinderGeneratorConfig,
    ... )
    >>> gen = RFdiffusionMPNNBinderGenerator(
    ...     RFdiffusionMPNNBinderGeneratorConfig(
    ...         target_structure="operator.pdb", target_chains=["A", "B"], inverse_folding="ligandmpnn"
    ...     )
    ... )
    >>> gen.input_type.value  # 'starting_sequence'
"""

import re
from math import ceil
from typing import Any, Literal, final

from proto_tools import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    LigandMPNNSampleConfig,
    ProteinMPNNSampleConfig,
    RFdiffusion3Config,
    RFdiffusion3DesignSpec,
    RFdiffusion3Input,
    Structure,
    run_ligandmpnn_sample,
    run_proteinmpnn_sample,
    run_rfdiffusion3,
)
from pydantic import field_validator, model_validator

from proto_language.core import Generator, GeneratorInputType
from proto_language.generator.generator_registry import generator
from proto_language.utils.base import BaseConfig, ConfigField

_HOTSPOT_PATTERN = re.compile(r"^[A-Za-z]\d+$")

InverseFoldingModel = Literal["proteinmpnn", "ligandmpnn"]


class RFdiffusionMPNNBinderGeneratorConfig(BaseConfig):
    """Configuration object for RFdiffusionMPNNBinderGenerator.

    Top-level fields describe *what* binder to design (the target, its chains, the
    epitope hotspots, and which inverse-folding model designs the sequence); the
    per-tool knobs live in nested ``rfdiffusion3_config`` and an inverse-folding config
    (``proteinmpnn_config`` or ``ligandmpnn_config``), matching how constraints nest
    tool configs (e.g. ``esmfold_config``). Only the config matching ``inverse_folding``
    is used; it defaults to its tool default when omitted. The binder length is taken
    from the assigned segment's length, not from this config.

    The generator injects the fields it owns at sample time and respects everything else:
    on ``rfdiffusion3_config`` it sets ``n_batches`` (to produce enough backbones for the
    proposal count) and ``seed``; on the inverse-folding config it sets ``seed`` and reads
    ``num_sequences_per_structure`` as the number of sequences designed per backbone.

    Attributes:
        target_structure (Structure | str): Target to design a binder against. Accepts a
            file path, PDB/CIF content string, or ``Structure`` object. May contain
            non-protein chains (DNA/RNA/ligand) — use ``inverse_folding="ligandmpnn"`` to
            condition the binder on them.
        target_chains (list[str]): Target chain IDs kept fixed during design; the binder
            backbone is emitted after them (so the binder is the last chain).
        hotspots (list[str] | None): Target epitope residues the binder should contact,
            as ``"<chain><resnum>"`` tokens (e.g. ``["A37", "A39"]``). ``None`` leaves the
            interface unrestricted. Each hotspot's chain must appear in ``target_chains``.
            When given, RFdiffusion3's generation origin is centered on them.
        inverse_folding (InverseFoldingModel): Which model designs the binder sequence.
            ``"proteinmpnn"`` (default) is protein-backbone only; ``"ligandmpnn"`` also
            conditions on ligand/nucleotide/metal atoms in the target.
        rfdiffusion3_config (RFdiffusion3Config): Advanced RFdiffusion3 backbone-generation
            settings (``n_batches`` and ``seed`` are managed by the generator).
        proteinmpnn_config (ProteinMPNNSampleConfig | None): Advanced ProteinMPNN
            sequence-design settings; used when ``inverse_folding="proteinmpnn"``.
            ``num_sequences_per_structure`` is the per-backbone count and ``seed`` is
            managed by the generator.
        ligandmpnn_config (LigandMPNNSampleConfig | None): Advanced LigandMPNN
            sequence-design settings; used when ``inverse_folding="ligandmpnn"``.
            ``num_sequences_per_structure`` is the per-backbone count and ``seed`` is
            managed by the generator.
    """

    target_structure: Structure | str = ConfigField(
        title="Target Structure",
        description="Target to bind (file path, PDB/CIF content, or Structure); may include DNA/RNA/ligand chains.",
    )
    target_chains: list[str] = ConfigField(
        default_factory=lambda: ["A"],
        title="Target Chains",
        description="Target chain IDs kept fixed; the binder is emitted after these chains.",
    )
    hotspots: list[str] | None = ConfigField(
        default=None,
        title="Hotspots",
        description="Target hotspot residues as '<chain><resnum>' (e.g. ['A37', 'A39']).",
    )
    inverse_folding: InverseFoldingModel = ConfigField(
        default="proteinmpnn",
        title="Inverse Folding Model",
        description="Sequence-design model: 'proteinmpnn' (protein only) or 'ligandmpnn' (ligand/DNA/RNA/metal-aware).",
    )
    rfdiffusion3_config: RFdiffusion3Config = ConfigField(
        default_factory=RFdiffusion3Config,
        title="RFdiffusion3 Config",
        description="Advanced RFdiffusion3 backbone-generation configuration.",
    )
    proteinmpnn_config: ProteinMPNNSampleConfig | None = ConfigField(
        default=None,
        title="ProteinMPNN Config",
        description="Advanced ProteinMPNN settings; used when inverse_folding='proteinmpnn'.",
    )
    ligandmpnn_config: LigandMPNNSampleConfig | None = ConfigField(
        default=None,
        title="LigandMPNN Config",
        description="Advanced LigandMPNN settings; used when inverse_folding='ligandmpnn'.",
    )

    @field_validator("hotspots", mode="after")
    @classmethod
    def _validate_hotspot_format(cls, v: list[str] | None) -> list[str] | None:
        """Each hotspot must match '<chain_letter><resnum>' (e.g. 'A37')."""
        if v is None:
            return None
        bad = [h for h in v if not _HOTSPOT_PATTERN.match(h)]
        if bad:
            raise ValueError(f"Hotspots must be '<chain><resnum>' (e.g. 'A37'). Bad: {bad}")
        return v

    @model_validator(mode="after")
    def _validate_hotspot_chains(self) -> "RFdiffusionMPNNBinderGeneratorConfig":
        """Every hotspot's chain must be one of ``target_chains``."""
        if self.hotspots:
            unknown = sorted({h[0] for h in self.hotspots} - set(self.target_chains))
            if unknown:
                raise ValueError(f"Hotspot chains {unknown} not in target_chains {self.target_chains}.")
        return self

    @model_validator(mode="after")
    def _default_inverse_folding_config(self) -> "RFdiffusionMPNNBinderGeneratorConfig":
        """Default the inverse-folding config matching ``inverse_folding`` when omitted."""
        if self.inverse_folding == "proteinmpnn" and self.proteinmpnn_config is None:
            self.proteinmpnn_config = ProteinMPNNSampleConfig()
        elif self.inverse_folding == "ligandmpnn" and self.ligandmpnn_config is None:
            self.ligandmpnn_config = LigandMPNNSampleConfig()
        return self


@generator(
    key="rfdiffusion-mpnn-binder",
    label="RFdiffusion3 + MPNN Binder Design",
    config=RFdiffusionMPNNBinderGeneratorConfig,
    description="De novo binder design: RFdiffusion3 backbones + ProteinMPNN/LigandMPNN sequences",
    uses_gpu=True,
    tools_called=["rfdiffusion3-design", "proteinmpnn-sample", "ligandmpnn-sample"],
    supported_sequence_types=["protein"],
)
@final
class RFdiffusionMPNNBinderGenerator(Generator):
    """De-novo protein binder generator chaining RFdiffusion3 and an inverse-folding model.

    For each ``sample()`` call the generator diffuses binder backbones docked to a fixed
    target (RFdiffusion3), then designs each backbone's binder-chain sequence with the
    selected inverse-folding model (``proteinmpnn`` or ``ligandmpnn``) while keeping the
    target chains fixed as structural context. With ``ligandmpnn`` the binder conditions on
    the target's ligand/nucleotide/metal atoms, so the target may be a protein, DNA, or RNA
    chain (with ligand/metal cofactors as context). The designed binder sequence is written
    to ``proposal.sequence`` and the RFdiffusion3 target+binder complex to
    ``proposal.structure`` (its binder chain carries RFdiffusion3's co-designed sequence, so
    downstream structure-prediction constraints should re-fold ``proposal.sequence``).

    The binder length is the assigned segment's length. The generator fills a length-only
    segment, so its category is ``"mutation"`` despite being a de-novo designer (same
    convention as ``RandomProteinGenerator``).

    Attributes:
        batch_size (int): Number of sequences to generate per batch (always 1; the backbone
            count is derived from the proposal count and the inverse-folding config's
            ``num_sequences_per_structure``).

    Example:
        Build a two-segment binder program: a length-only ``binder`` (designed) plus a fixed
        ``target`` segment derived from the same structure, so a scoring constraint can fold
        the complex via ``inputs=[binder, target]``. The generator is assigned only to the
        binder; the target reaches it through config. See
        ``examples/scripts/binder_design_rfdiffusion_mpnn.py`` for the full program.

        >>> from proto_tools import Structure
        >>> from proto_language.core import Construct, Segment
        >>> from proto_language.generator import (
        ...     RFdiffusionMPNNBinderGenerator,
        ...     RFdiffusionMPNNBinderGeneratorConfig,
        ... )
        >>> target_structure = Structure.from_file("target.pdb")
        >>> target_seq = target_structure.get_chain_sequence("A", remove_non_standard=True)
        >>> binder = Segment(length=80, sequence_type="protein", label="binder")
        >>> target = Segment(sequence=target_seq, sequence_type="protein", label="target")
        >>> construct = Construct([binder, target])  # target is fixed: no generator
        >>> gen = RFdiffusionMPNNBinderGenerator(
        ...     RFdiffusionMPNNBinderGeneratorConfig(
        ...         target_structure=target_structure, target_chains=["A"], hotspots=["A37"]
        ...     )
        ... )
        >>> gen.assign(binder)  # generator touches only the binder
        >>> gen.sample()  # fills num_proposals binders; a constraint scores [binder, target]
    """

    input_type = GeneratorInputType.STARTING_SEQUENCE
    allows_empty_starting_sequence = True
    batch_size: int = 1

    def __init__(self, config: RFdiffusionMPNNBinderGeneratorConfig) -> None:
        """Initialize the binder generator.

        Args:
            config (RFdiffusionMPNNBinderGeneratorConfig): Configuration object
                containing all generator parameters.
        """
        super().__init__()
        self.config = config
        self.target_structure = config.target_structure
        self.target_chains = config.target_chains
        self.hotspots = config.hotspots
        self.inverse_folding = config.inverse_folding
        self.rfdiffusion3_config = config.rfdiffusion3_config
        # The active inverse-folding config; the validator guarantees the selected one is set.
        if_config = config.ligandmpnn_config if config.inverse_folding == "ligandmpnn" else config.proteinmpnn_config
        assert if_config is not None  # noqa: S101 -- guaranteed by _default_inverse_folding_config
        self.if_config: ProteinMPNNSampleConfig | LigandMPNNSampleConfig = if_config

    def _preserve_structure_after_sample(self) -> bool:
        """Keep the RFdiffusion3 target+binder complex written onto each proposal."""
        return True

    def _sample(self) -> None:
        """Design binders against the target and write them onto proposal sequences.

        Diffuses enough binder backbones to cover the proposal count (at the
        inverse-folding config's ``num_sequences_per_structure`` sequences per backbone),
        designs each backbone's binder-chain sequence with the selected inverse-folding
        model, then writes the binder sequence, the target+binder complex, and per-design
        metrics onto each proposal.

        Raises:
            RuntimeError: If RFdiffusion3 returns no backbones, or if the pipeline yields
                fewer designs than the number of proposals.
        """
        # A staged upload, path, or content string arrives as str; materialize before use.
        target_structure = self.target_structure
        if isinstance(target_structure, str):
            target_structure = Structure(structure=target_structure)

        segment = self.segment
        binder_len = segment.sequence_length
        num_proposals = segment.num_proposals

        # De-novo: the binder may be a length-only segment. Seed 'X' so the
        # STARTING_SEQUENCE validator passes; the real sequence is designed below.
        if not any(seq.sequence for seq in segment.proposal_sequences):
            for seq in segment.proposal_sequences:
                seq.sequence = "X" * binder_len
        self._validate_generator()

        contig = self._build_contig(binder_len, target_structure)

        # One inverse-folding run per backbone (num_sequences_per_structure seqs each);
        # n_batches fans out enough backbones within diffusion_batch_size.
        seqs_per_backbone = self.if_config.num_sequences_per_structure
        num_backbones = ceil(num_proposals / seqs_per_backbone)
        n_batches = ceil(num_backbones / self.rfdiffusion3_config.diffusion_batch_size)

        rfd_config = self.rfdiffusion3_config.model_copy(update={"n_batches": n_batches, "seed": self._next_seed()})
        rfd_output = run_rfdiffusion3(
            inputs=RFdiffusion3Input(
                design_specs=[
                    RFdiffusion3DesignSpec(
                        input_structure=target_structure,
                        contig=contig,
                        select_hotspots=",".join(self.hotspots) if self.hotspots else None,
                        # RFdiffusion3 centers the origin on the input COM by default; for
                        # hotspot-directed binder design, center it on the epitope instead.
                        infer_ori_strategy="hotspots" if self.hotspots else None,
                    )
                ]
            ),
            config=rfd_config,
        )
        # RFdiffusion3 fans out n_batches * diffusion_batch_size designs; keep what we need.
        backbones = list(rfd_output.designed_structures[0])[:num_backbones]
        if not backbones:
            raise RuntimeError(f"RFdiffusion3 produced no binder backbones for contig {contig!r}.")

        # _build_contig always emits the binder last, and RFdiffusion3 relabels output
        # chains by emission order, so the binder is the last output chain.
        structure_inputs = [
            InverseFoldingStructureInput(
                structure=backbone.structure,
                chains_to_redesign=[backbone.structure.get_chain_ids()[-1]],
            )
            for backbone in backbones
        ]

        if_config = self.if_config.model_copy(update={"seed": self._next_seed()})
        run_inverse_folding = run_ligandmpnn_sample if self.inverse_folding == "ligandmpnn" else run_proteinmpnn_sample
        if_output = run_inverse_folding(
            inputs=InverseFoldingInput(inputs=structure_inputs),
            config=if_config,
        )

        records = self._collect_designs(backbones, structure_inputs, if_output.design_sets)
        if len(records) < num_proposals:
            raise RuntimeError(
                f"Binder design produced {len(records)} sequences, fewer than the {num_proposals} "
                f"requested. Raise {self.inverse_folding}_config.num_sequences_per_structure or check "
                "the RFdiffusion3 output."
            )
        records = records[:num_proposals]

        key = self._spec.key
        for proposal, record in zip(segment.proposal_sequences, records, strict=True):
            proposal.sequence = record["binder_sequence"]
            proposal.structure = record["structure"]
            proposal._generator_metadata[key] = {
                "perplexity": record["perplexity"],
                "sequence_recovery": record["sequence_recovery"],
                "ligand_interface_sequence_recovery": record["ligand_interface_sequence_recovery"],
                "contig": contig,
                "full_complex_sequence": record["full_complex_sequence"],
            }

    def _build_contig(self, binder_len: int, target_structure: Structure) -> str:
        """Build the RFdiffusion3 binder contig from the target chains and binder length.

        Keeps each target chain over its full residue span, adds a chain break, then appends
        the binder length — e.g. ``"A1-100,/0,80"``. The binder is always emitted last, which
        is what lets ``_sample`` take the last output chain as the binder to redesign.

        Args:
            binder_len (int): Length of the binder to design (the segment length).
            target_structure (Structure): Materialized target structure to read chain spans from.

        Returns:
            str: The contig string passed to RFdiffusion3.

        Raises:
            ValueError: If a target chain is absent from ``target_structure``.
        """
        segments: list[str] = []
        for chain_id in self.target_chains:
            positions = target_structure.get_chain_positions(chain_id)
            if not positions:
                raise ValueError(f"Target chain {chain_id!r} not found in target_structure.")
            segments.append(f"{chain_id}{min(positions)}-{max(positions)}")
        return ",/0,".join(segments) + f",/0,{binder_len}"

    @staticmethod
    def _collect_designs(
        backbones: list[Any],
        structure_inputs: list[InverseFoldingStructureInput],
        design_sets: list[Any],
    ) -> list[dict[str, Any]]:
        """Flatten the backbone x designed-sequence grid into ordered design records.

        Metrics are read model-agnostically via the shared ``Metrics`` mapping access:
        ``sequence_recovery`` is always present, ``perplexity`` only for ProteinMPNN, and
        ``ligand_interface_sequence_recovery`` only for LigandMPNN — absent metrics are
        ``None``.

        Args:
            backbones (list[Any]): RFdiffusion3 backbone structures, one per inverse-folding input.
            structure_inputs (list[InverseFoldingStructureInput]): Inverse-folding inputs aligned
                to ``backbones``; each names the binder chain to redesign.
            design_sets (list[Any]): Inverse-folding design sets aligned to ``backbones``.

        Returns:
            list[dict[str, Any]]: One record per designed sequence with ``binder_sequence``,
                ``structure`` (target+binder complex), ``perplexity``, ``sequence_recovery``,
                ``ligand_interface_sequence_recovery``, and ``full_complex_sequence``.
        """
        records: list[dict[str, Any]] = []
        for backbone, struct_input, design_set in zip(backbones, structure_inputs, design_sets, strict=True):
            binder_chain_id = struct_input.chain_ids_to_redesign[0]
            for design in design_set.complexes:
                binder_sequence = next(
                    chain.sequence
                    for chain, was_designed in zip(design.chains, design.designed, strict=True)
                    if was_designed and chain.id == binder_chain_id
                )
                metrics = design.metrics
                records.append(
                    {
                        "binder_sequence": binder_sequence,
                        "structure": backbone.structure,
                        "perplexity": metrics.get("perplexity"),
                        "sequence_recovery": metrics.get("sequence_recovery"),
                        "ligand_interface_sequence_recovery": metrics.get("ligand_interface_sequence_recovery"),
                        # design.chains are protein chains only; a ligandmpnn non-protein target is
                        # conditioning context (carried in proposal.structure), not joined here.
                        "full_complex_sequence": "/".join(chain.sequence for chain in design.chains),
                    }
                )
        return records
