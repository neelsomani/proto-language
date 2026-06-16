"""tests/language_tests/generator_tests/test_rfdiffusion_mpnn_binder_generator.py."""

import copy
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
from proto_tools import LigandMPNNSampleConfig, ProteinMPNNSampleConfig, Structure

from proto_language.core import Segment
from proto_language.generator import (
    RFdiffusionMPNNBinderGenerator,
    RFdiffusionMPNNBinderGeneratorConfig,
)

KEY = "rfdiffusion-mpnn-binder"
_BINDER_AAS = "ACDEFGHIKLMNPQRSTVWY"
_PROTEINMPNN_METRICS = {"perplexity": 2.5, "sequence_recovery": 0.4}
_LIGANDMPNN_METRICS = {"sequence_recovery": 0.4, "ligand_interface_sequence_recovery": 0.6}


def _relabel_chain(pdb: str, chain_id: str) -> str:
    """Return the ATOM records of ``pdb`` with their chain column set to ``chain_id``."""
    return "\n".join(
        (line[:21] + chain_id + line[22:]) if line.startswith("ATOM") else line
        for line in pdb.splitlines()
        if line not in ("END", "")
    )


def _multi_chain_pdb(base_pdb: str, chain_ids: list[str]) -> str:
    """Build a PDB whose chains are copies of ``base_pdb`` relabeled to ``chain_ids``."""
    return "\n".join(_relabel_chain(base_pdb, cid) for cid in chain_ids) + "\nEND\n"


def _make_rfd_mock(
    captured: dict[str, Any], *, output_chain_ids: list[str], base_pdb: str, n_backbones: int | None = None
):
    """Fake ``run_rfdiffusion3`` returning one bundle of multi-chain backbones."""

    def fake_rfd(*, inputs, config):
        captured["rfd_inputs"] = inputs
        captured["rfd_config"] = config
        count = n_backbones if n_backbones is not None else config.n_batches * config.diffusion_batch_size
        backbones = [
            SimpleNamespace(structure=Structure(structure=_multi_chain_pdb(base_pdb, output_chain_ids)))
            for _ in range(count)
        ]
        return SimpleNamespace(designed_structures=[backbones])

    return fake_rfd


def _make_if_mock(captured: dict[str, Any], *, tool_name: str, metrics: dict[str, float]):
    """Fake an inverse-folding sample run designing a distinct binder per backbone/sequence.

    Records which tool was dispatched and attaches ``metrics`` (a dict, like the real
    ``Metrics`` mapping) to each design.
    """

    def fake_if(*, inputs, config):
        captured["if_inputs"] = inputs
        captured["if_config"] = config
        captured["if_tool"] = tool_name
        counter = 0
        design_sets = []
        for struct_input in inputs.inputs:
            chain_ids = struct_input.structure.get_chain_ids()
            binder_id = struct_input.chain_ids_to_redesign[0]
            complexes = []
            for _ in range(config.num_sequences_per_structure):
                binder_seq = _BINDER_AAS[counter % len(_BINDER_AAS)] * 5
                counter += 1
                chains = [
                    SimpleNamespace(id=cid, sequence=binder_seq if cid == binder_id else "AGSVL") for cid in chain_ids
                ]
                complexes.append(
                    SimpleNamespace(
                        chains=chains,
                        designed=[cid == binder_id for cid in chain_ids],
                        metrics=dict(metrics),
                    )
                )
            design_sets.append(SimpleNamespace(complexes=complexes))
        return SimpleNamespace(design_sets=design_sets)

    return fake_if


def _patch_tools(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    output_chain_ids: list[str],
    base_pdb: str,
    n_backbones: int | None = None,
) -> None:
    """Patch RFdiffusion3 and both inverse-folding runners; only the selected one is called."""
    module = import_module("proto_language.generator.rfdiffusion_mpnn_binder_generator")
    monkeypatch.setattr(
        module,
        "run_rfdiffusion3",
        _make_rfd_mock(captured, output_chain_ids=output_chain_ids, base_pdb=base_pdb, n_backbones=n_backbones),
    )
    monkeypatch.setattr(
        module, "run_proteinmpnn_sample", _make_if_mock(captured, tool_name="proteinmpnn", metrics=_PROTEINMPNN_METRICS)
    )
    monkeypatch.setattr(
        module, "run_ligandmpnn_sample", _make_if_mock(captured, tool_name="ligandmpnn", metrics=_LIGANDMPNN_METRICS)
    )


def _binder_segment(length: int = 5, num_proposals: int = 1) -> Segment:
    """Length-only protein segment with the requested number of empty proposals."""
    segment = Segment(length=length, sequence_type="protein")
    if num_proposals > 1:
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(num_proposals)]
    return segment


class TestRFdiffusionMPNNBinderGeneratorConfig:
    """Config-level construction and validation (no tool calls)."""

    def test_init_stores_config(self, sample_pdb_content: str) -> None:
        config = RFdiffusionMPNNBinderGeneratorConfig(
            target_structure=sample_pdb_content,
            target_chains=["A"],
            hotspots=["A3"],
            proteinmpnn_config=ProteinMPNNSampleConfig(temperature=0.2, num_sequences_per_structure=2),
        )
        gen = RFdiffusionMPNNBinderGenerator(config)
        assert gen.target_chains == ["A"]
        assert gen.hotspots == ["A3"]
        assert gen.inverse_folding == "proteinmpnn"
        # The active inverse-folding config is the proteinmpnn one for the default model.
        assert gen.if_config.temperature == 0.2
        assert gen.if_config.num_sequences_per_structure == 2

    def test_proteinmpnn_config_defaulted_by_default(self, sample_pdb_content: str) -> None:
        config = RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content)
        assert isinstance(config.proteinmpnn_config, ProteinMPNNSampleConfig)
        assert config.ligandmpnn_config is None

    def test_ligandmpnn_config_defaulted_when_selected(self, sample_pdb_content: str) -> None:
        config = RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, inverse_folding="ligandmpnn")
        assert isinstance(config.ligandmpnn_config, LigandMPNNSampleConfig)
        assert config.proteinmpnn_config is None
        gen = RFdiffusionMPNNBinderGenerator(config)
        assert isinstance(gen.if_config, LigandMPNNSampleConfig)

    def test_accepts_deferred_upload_reference(self) -> None:
        # An upload reference isn't valid PDB; Structure | str keeps it as a string (staged at run time).
        config = RFdiffusionMPNNBinderGeneratorConfig(target_structure="user_upload:asset_deadbeef")
        gen = RFdiffusionMPNNBinderGenerator(config)
        assert gen.target_structure == "user_upload:asset_deadbeef"

    def test_bad_hotspot_format_rejected(self, sample_pdb_content: str) -> None:
        with pytest.raises(ValueError, match="Hotspots must be"):
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, hotspots=["3"])

    def test_hotspot_chain_not_in_target_rejected(self, sample_pdb_content: str) -> None:
        with pytest.raises(ValueError, match="not in target_chains"):
            RFdiffusionMPNNBinderGeneratorConfig(
                target_structure=sample_pdb_content, target_chains=["A"], hotspots=["B3"]
            )

    def test_contig_override_not_accepted(self, sample_pdb_content: str) -> None:
        # The contig is always auto-built (binder last) so the binder is reliably the last
        # output chain; an explicit override could put the binder first and silently redesign
        # the target, so it is rejected.
        with pytest.raises(ValueError, match="contig"):
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, contig="50-120,/0,A17-131")


class TestRFdiffusionMPNNBinderGeneratorAssign:
    """Segment-assignment validation."""

    @pytest.mark.parametrize("sequence_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, sample_pdb_content: str, sequence_type: str) -> None:
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        with pytest.raises(ValueError, match="does not support sequence type"):
            gen.assign(Segment(length=50, sequence_type=sequence_type))

    def test_rejects_ligand_segment(self, sample_pdb_content: str) -> None:
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))


class TestRFdiffusionMPNNBinderGeneratorSample:
    """End-to-end sampling behavior with both tools mocked."""

    def test_contig_auto_build_single_target(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, target_chains=["A"])
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        assert captured["rfd_inputs"].design_specs[0].contig == "A1-5,/0,5"

    def test_contig_auto_build_multi_target(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        target_pdb = _multi_chain_pdb(sample_pdb_content, ["A", "B"])
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B", "C"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=target_pdb, target_chains=["A", "B"])
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        assert captured["rfd_inputs"].design_specs[0].contig == "A1-5,/0,B1-5,/0,5"
        assert captured["if_inputs"].inputs[0].chain_ids_to_redesign == ["C"]

    def test_binder_chain_is_last_and_only_redesigned(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        assert captured["if_inputs"].inputs[0].chain_ids_to_redesign == ["B"]
        assert segment.proposal_sequences[0].sequence != "AGSVL"
        assert len(segment.proposal_sequences[0].sequence) == 5

    def test_proteinmpnn_is_default_model(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        assert captured["if_tool"] == "proteinmpnn"
        assert isinstance(captured["if_config"], ProteinMPNNSampleConfig)
        metadata = segment.proposal_sequences[0]._generator_metadata[KEY]
        assert metadata["perplexity"] == 2.5
        assert metadata["ligand_interface_sequence_recovery"] is None

    def test_ligandmpnn_path_dispatches_and_flows_ligand_metrics(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, inverse_folding="ligandmpnn")
        )
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        # LigandMPNN is dispatched (not ProteinMPNN) and the binder chain is held to redesign.
        assert captured["if_tool"] == "ligandmpnn"
        assert isinstance(captured["if_config"], LigandMPNNSampleConfig)
        assert captured["if_inputs"].inputs[0].chain_ids_to_redesign == ["B"]
        metadata = segment.proposal_sequences[0]._generator_metadata[KEY]
        assert metadata["sequence_recovery"] == 0.4
        assert metadata["ligand_interface_sequence_recovery"] == 0.6
        assert metadata["perplexity"] is None

    def test_ligandmpnn_non_protein_target_contig_and_dispatch(self, monkeypatch, sample_pdb_content: str) -> None:
        # The headline capability: a non-protein target chain (chain B stands in for DNA) is held
        # fixed via the auto-built contig and conditions the binder; LigandMPNN is dispatched and
        # only the appended binder chain (C) is redesigned.
        captured: dict[str, Any] = {}
        target_pdb = _multi_chain_pdb(sample_pdb_content, ["A", "B"])
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B", "C"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(
                target_structure=target_pdb, target_chains=["A", "B"], inverse_folding="ligandmpnn"
            )
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        assert captured["if_tool"] == "ligandmpnn"
        assert captured["rfd_inputs"].design_specs[0].contig == "A1-5,/0,B1-5,/0,5"
        assert captured["if_inputs"].inputs[0].chain_ids_to_redesign == ["C"]

    def test_hotspots_forwarded_and_center_origin(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content, hotspots=["A3", "A5"])
        )
        gen.assign(_binder_segment(length=5))
        gen.sample()
        spec = captured["rfd_inputs"].design_specs[0]
        assert spec.select_hotspots == "A3,A5"
        # Origin strategy is derived from hotspots, not configured: hotspots -> centered on epitope.
        assert spec.infer_ori_strategy == "hotspots"

    def test_no_hotspots_leaves_origin_unset(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        gen.assign(_binder_segment(length=5))
        gen.sample()
        spec = captured["rfd_inputs"].design_specs[0]
        assert spec.select_hotspots is None
        assert spec.infer_ori_strategy is None

    def test_num_proposals_drives_backbone_count(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5, num_proposals=3)
        gen.assign(segment)
        gen.sample()
        # 3 proposals / 1 seq-per-backbone -> 3 backbones designed by the inverse-folding model.
        assert len(captured["if_inputs"].inputs) == 3
        assert captured["if_config"].num_sequences_per_structure == 1
        assert len({seq.sequence for seq in segment.proposal_sequences}) == 3

    def test_sequences_per_backbone_truncates_to_num_proposals(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(
            RFdiffusionMPNNBinderGeneratorConfig(
                target_structure=sample_pdb_content,
                proteinmpnn_config=ProteinMPNNSampleConfig(num_sequences_per_structure=2),
            )
        )
        segment = _binder_segment(length=5, num_proposals=3)
        gen.assign(segment)
        gen.sample()
        # ceil(3 / 2) = 2 backbones x 2 seqs = 4 designs, truncated to the 3 proposals.
        assert len(captured["if_inputs"].inputs) == 2
        assert captured["if_config"].num_sequences_per_structure == 2
        assert all(len(seq.sequence) == 5 for seq in segment.proposal_sequences)
        assert segment.num_proposals == 3

    def test_structure_preserved_after_sample(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        proposal = segment.proposal_sequences[0]
        assert proposal.structure is not None
        assert proposal.structure.get_chain_ids() == ["A", "B"]

    def test_metadata_written(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5)
        gen.assign(segment)
        gen.sample()
        metadata = segment.proposal_sequences[0]._generator_metadata[KEY]
        assert metadata["perplexity"] == 2.5
        assert metadata["sequence_recovery"] == 0.4
        assert metadata["contig"] == "A1-5,/0,5"
        assert metadata["full_complex_sequence"] == f"AGSVL/{segment.proposal_sequences[0].sequence}"

    def test_de_novo_empty_segment_runs(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        segment = _binder_segment(length=5)
        assert not segment.proposals_populated
        gen.assign(segment)
        gen.sample()
        sequence = segment.proposal_sequences[0].sequence
        assert len(sequence) == 5
        assert sequence != "XXXXX"

    def test_no_backbones_raises(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content, n_backbones=0)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        gen.assign(_binder_segment(length=5))
        with pytest.raises(RuntimeError, match="no binder backbones"):
            gen.sample()

    def test_fewer_designs_than_requested_raises(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content, n_backbones=1)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        gen.assign(_binder_segment(length=5, num_proposals=3))
        with pytest.raises(RuntimeError, match="fewer than"):
            gen.sample()

    def test_seed_advances_across_tools(self, monkeypatch, sample_pdb_content: str) -> None:
        captured: dict[str, Any] = {}
        _patch_tools(monkeypatch, captured, output_chain_ids=["A", "B"], base_pdb=sample_pdb_content)
        gen = RFdiffusionMPNNBinderGenerator(RFdiffusionMPNNBinderGeneratorConfig(target_structure=sample_pdb_content))
        gen._set_program_seed(123)
        gen.assign(_binder_segment(length=5))
        gen.sample()
        rfd_seed = captured["rfd_config"].seed
        if_seed = captured["if_config"].seed
        assert isinstance(rfd_seed, int)
        assert isinstance(if_seed, int)
        assert rfd_seed != if_seed
