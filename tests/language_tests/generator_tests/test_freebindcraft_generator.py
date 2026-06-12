"""tests/language_tests/generator_tests/test_freebindcraft_generator.py."""

import copy
from importlib import import_module
from types import SimpleNamespace

import pytest
from proto_tools import FreeBindCraftConfig, Structure

from proto_language.core import Segment
from proto_language.generator import FreeBindCraftGenerator, FreeBindCraftGeneratorConfig

_MODULE = "proto_language.generator.freebindcraft_generator"


def _fake_design(length: int, pdb_content: str, *, iptm: float = 0.8) -> SimpleNamespace:
    """A stand-in FreeBindCraftDesign: binder sequence, predicted complex, PyRosetta-free metrics."""
    return SimpleNamespace(
        binder_sequence="A" * length,
        structure=Structure(structure=pdb_content),
        metrics={"avg_iptm": iptm, "shape_complementarity": 0.6, "dSASA": 900.0},
    )


def _install_fake_run(monkeypatch, captured, designs):
    """Monkeypatch the module's run_freebindcraft_design to capture args and return ``designs``."""
    module = import_module(_MODULE)

    def fake_run(*, inputs, config):
        captured["inputs"] = inputs
        captured["config"] = config
        return SimpleNamespace(designs=list(designs))

    monkeypatch.setattr(module, "run_freebindcraft_design", fake_run)


class TestFreeBindCraftGeneratorConfig:
    """Config/registration unit tests (no GPU, no tool dispatch)."""

    def test_init_stores_target_and_design_config(self, sample_pdb_content: str) -> None:
        config = FreeBindCraftGeneratorConfig(
            target_structure=sample_pdb_content,
            target_chain="B",
            target_hotspot_residues="10,20",
            binder_name="proj",
            design_config=FreeBindCraftConfig(soft_iterations=12),
        )
        gen = FreeBindCraftGenerator(config)
        assert gen.target_chain == "B"
        assert gen.target_hotspot_residues == "10,20"
        assert gen.binder_name == "proj"
        assert gen.design_config.soft_iterations == 12
        # Stored verbatim; the Structure | str field materializes lazily at sample time.
        assert gen.target_structure == sample_pdb_content

    def test_accepts_deferred_upload_reference(self) -> None:
        # An upload reference isn't valid PDB; Structure | str keeps it as a string (staged at run time).
        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure="user_upload:asset_deadbeef"))
        assert gen.target_structure == "user_upload:asset_deadbeef"

    def test_design_config_defaults_to_freebindcraft_config(self, sample_pdb_content: str) -> None:
        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        assert isinstance(gen.design_config, FreeBindCraftConfig)

    def test_input_type_and_de_novo_flag(self, sample_pdb_content: str) -> None:
        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        assert gen.input_type.value == "starting_sequence"
        assert gen.allows_empty_starting_sequence is True
        # The predicted complex is a fresh output, so it survives sample().
        assert gen._preserve_structure_after_sample() is True


class TestFreeBindCraftGeneratorValidation:
    """Segment-assignment validation (no GPU)."""

    @pytest.mark.parametrize("sequence_type", ["dna", "rna"])
    def test_rejects_non_protein_segment(self, sample_pdb_content: str, sequence_type: str) -> None:
        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        with pytest.raises(ValueError, match="does not support sequence type"):
            gen.assign(Segment(length=50, sequence_type=sequence_type))

    def test_rejects_ligand_segment(self, sample_pdb_content: str) -> None:
        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        with pytest.raises(ValueError, match="Cannot assign generator to ligand segment"):
            gen.assign(Segment(sequence="CCC", sequence_type="ligand"))


class TestFreeBindCraftGeneratorSampling:
    """Sampling behavior with a mocked tool dispatch (no GPU)."""

    def test_de_novo_sampling_writes_sequence_structure_metadata(
        self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str
    ) -> None:
        length = 12
        captured: dict[str, object] = {}
        _install_fake_run(monkeypatch, captured, [_fake_design(length, sample_pdb_content)])

        gen = FreeBindCraftGenerator(
            FreeBindCraftGeneratorConfig(
                target_structure=sample_pdb_content, target_chain="A", target_hotspot_residues="56", binder_name="b"
            )
        )
        segment = Segment(length=length, sequence_type="protein")
        gen.assign(segment)
        gen.sample()

        proposal = segment.proposal_sequences[0]
        assert proposal.sequence == "A" * length
        assert proposal.structure is not None
        assert proposal._generator_metadata["freebindcraft"] == {
            "avg_iptm": 0.8,
            "shape_complementarity": 0.6,
            "dSASA": 900.0,
        }
        # Segment length is the binder length; one design is requested per proposal slot.
        inputs = captured["inputs"]
        assert inputs.binder_lengths == (length, length)
        assert inputs.number_of_final_designs == 1
        assert inputs.target_chain == "A"
        assert inputs.target_hotspot_residues == "56"
        assert inputs.binder_name == "b"

    def test_requests_one_design_per_proposal(self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str) -> None:
        length = 10
        captured: dict[str, object] = {}
        designs = [_fake_design(length, sample_pdb_content, iptm=0.5 + 0.1 * i) for i in range(3)]
        _install_fake_run(monkeypatch, captured, designs)

        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        segment = Segment(length=length, sequence_type="protein")
        gen.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(3)]
        gen.sample()

        assert captured["inputs"].number_of_final_designs == 3
        assert segment.num_proposals == 3
        assert all(p.sequence == "A" * length for p in segment.proposal_sequences)

    def test_fewer_designs_than_requested_truncates_pool(
        self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        length = 8
        captured: dict[str, object] = {}
        _install_fake_run(monkeypatch, captured, [_fake_design(length, sample_pdb_content)])  # 1 design

        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        segment = Segment(length=length, sequence_type="protein")
        gen.assign(segment)
        segment.proposal_sequences = [copy.deepcopy(segment.original_sequence) for _ in range(4)]

        with caplog.at_level("WARNING"):
            gen.sample()

        assert segment.num_proposals == 1, "proposal pool should shrink to the designs produced"
        assert segment.proposal_sequences[0].sequence == "A" * length
        assert "truncating the proposal pool" in caplog.text

    def test_zero_designs_raises(self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str) -> None:
        captured: dict[str, object] = {}
        _install_fake_run(monkeypatch, captured, [])

        gen = FreeBindCraftGenerator(FreeBindCraftGeneratorConfig(target_structure=sample_pdb_content))
        gen.assign(Segment(length=10, sequence_type="protein"))
        with pytest.raises(RuntimeError, match="no accepted designs"):
            gen.sample()

    def test_program_seed_overrides_design_config_seed(
        self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str
    ) -> None:
        captured: dict[str, object] = {}
        _install_fake_run(monkeypatch, captured, [_fake_design(10, sample_pdb_content)])

        gen = FreeBindCraftGenerator(
            FreeBindCraftGeneratorConfig(
                target_structure=sample_pdb_content, design_config=FreeBindCraftConfig(seed=999)
            )
        )
        gen.assign(Segment(length=10, sequence_type="protein"))
        gen._set_program_seed(42)
        gen.sample()
        # Seeded program owns determinism: design_config.seed (999) is overridden.
        assert captured["config"].seed is not None
        assert captured["config"].seed != 999

    def test_unseeded_program_preserves_design_config_seed(
        self, monkeypatch: pytest.MonkeyPatch, sample_pdb_content: str
    ) -> None:
        captured: dict[str, object] = {}
        _install_fake_run(monkeypatch, captured, [_fake_design(10, sample_pdb_content)])

        gen = FreeBindCraftGenerator(
            FreeBindCraftGeneratorConfig(
                target_structure=sample_pdb_content, design_config=FreeBindCraftConfig(seed=999)
            )
        )
        gen.assign(Segment(length=10, sequence_type="protein"))
        gen.sample()  # no program seed set
        assert captured["config"].seed == 999
