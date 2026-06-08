"""Smoke test for the Germinal PD-L1 antibody design pipeline."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from proto_tools.entities.msa import MSA
from proto_tools.tools.sequence_alignment.mmseqs2.homology_search import (
    Mmseqs2HomologySearchOutput,
    Mmseqs2HomologySearchResult,
)

from proto_language.constraint.protein_structure.structure_confidence_constraint import (
    StructureBasedConstraintConfig,
    structure_composite_constraint,
)
from proto_language.core import Sequence

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "germinal" / "run_germinal_pipeline.py"
PDB_DIR = SCRIPT.parent / "pdbs"

# A real target with deep, easily-found homologs (ubiquitin) and a short binder, for the
# end-to-end target-only-MSA cofold test below.
_E2E_TARGET = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_E2E_BINDER = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"

_TARGET_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
_TARGET_B = "GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDG"


def _load_pipeline():
    """Import the Germinal example script as a module (presets validate on load)."""
    spec = importlib.util.spec_from_file_location("run_germinal_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_germinal_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def _search_result(*sequences: str, paired: bool) -> Mmseqs2HomologySearchOutput:
    """Build a one-group search output: a singleton (unpaired) or one taxonomy-paired group."""
    chain_msas = [MSA(aligned_sequences=[s, s, s]) for s in sequences]
    result = Mmseqs2HomologySearchResult(
        sequence_ids=[f"t{i}" for i in range(len(sequences))],
        msas=chain_msas,
        paired_msas=chain_msas if paired else [None] * len(sequences),
        datasets_searched=["uniref30-2302"],
        num_homologs_found=[2] * len(sequences),
    )
    return Mmseqs2HomologySearchOutput(results=[result])


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_germinal_vhh_smoke(tmp_path: Path) -> None:
    """Run the new Germinal VHH entrypoint end-to-end with smoke-sized overrides."""
    output_dir = tmp_path / "outputs" / "smoke_test"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            "--preset",
            "vhh",
            "--target-pdb",
            str(PDB_DIR / "pdl1.pdb"),
            "--target-chain",
            "A",
            "--target-hotspots",
            "A37,A39,A41,A96,A98",
            "--max-trajectories",
            "1",
            "--max-passing",
            "1",
            "--logits-steps",
            "3",
            "--softmax-steps",
            "2",
            "--search-steps",
            "3",
            "--num-seqs",
            "2",
            "--max-mpnn-sequences",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, f"VHH pipeline failed:\n{result.stderr[-2000:]}"
    run_dirs = sorted((output_dir / "germinal" / "pdl1").glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert any(run_dir.glob("*_binder.pdb"))

    summary_path = run_dir / "trajectory_summary.json"
    assert summary_path.exists(), "trajectory_summary.json not produced"
    summary = json.loads(summary_path.read_text())
    assert summary["num_trajectories"] == 1
    assert "trajectories" in summary
    assert len(summary["trajectories"]) == 1
    assert "stages" in summary["trajectories"][0]

    assert (run_dir / "trajectory_dynamics.png").exists(), "trajectory_dynamics.png not produced"

    variant_jsons = sorted(run_dir.glob("traj*_variant*_*.json"))
    for json_path in variant_jsons:
        assert json_path.with_suffix(".fasta").exists()
        assert json_path.with_suffix(".pdb").exists()


class TestCofoldMsaMode:
    """Fast unit coverage for the ``msa_mode="target"`` cofold path (no GPU / no network)."""

    def test_presets_declare_msa_mode(self):
        """Every preset carries an ``msa_mode`` the loader validated (drops nothing)."""
        pipeline = _load_pipeline()
        assert pipeline.GERMINAL_PRESETS, "no presets loaded"
        for geom in pipeline.GERMINAL_PRESETS.values():
            assert geom.msa_mode in pipeline._MSA_MODES

    def test_cofold_config_disables_auto_search(self):
        """The cofold tool config sets ``use_msa=False`` so the binder is never auto-searched."""
        pipeline = _load_pipeline()
        assert pipeline._cofold_config("chai1", 7)["chai1_config"]["use_msa"] is False
        assert pipeline._cofold_config("alphafold3", 7)["alphafold3_config"]["use_msa"] is False

    def test_target_cofold_msas_single_chain_omits_binder(self):
        """A single target chain → unpaired MSA keyed to index 1; the binder (index 0) is omitted."""
        pipeline = _load_pipeline()
        target_seqs = [Sequence(_TARGET_A, "protein")]
        with patch.object(
            pipeline, "run_mmseqs2_homology_search", return_value=_search_result(_TARGET_A, paired=False)
        ):
            bundle = pipeline._target_cofold_msas(target_seqs)
        assert set(bundle.per_chain) == {1}  # binder index 0 absent → single-sequence
        assert bundle.paired is False
        assert bundle.per_chain[1].original_sequences[0] == _TARGET_A

    def test_target_cofold_msas_multichain_pairs_targets(self):
        """Several target chains → one taxonomy-paired group keyed to indices 1..N (binder omitted)."""
        pipeline = _load_pipeline()
        target_seqs = [Sequence(_TARGET_A, "protein"), Sequence(_TARGET_B, "protein")]
        with patch.object(
            pipeline, "run_mmseqs2_homology_search", return_value=_search_result(_TARGET_A, _TARGET_B, paired=True)
        ):
            bundle = pipeline._target_cofold_msas(target_seqs)
        assert set(bundle.per_chain) == {1, 2}  # targets at 1,2; binder index 0 omitted
        assert bundle.paired is True

    def test_target_cofold_msas_warns_on_empty_search(self, capsys):
        """A target with no homologs yields an empty bundle and a loud warning (not a silent single-seq)."""
        pipeline = _load_pipeline()
        empty = Mmseqs2HomologySearchOutput(
            results=[
                Mmseqs2HomologySearchResult(
                    sequence_ids=["t0"], msas=[None], paired_msas=[None], datasets_searched=[], num_homologs_found=[0]
                )
            ]
        )
        with patch.object(pipeline, "run_mmseqs2_homology_search", return_value=empty):
            bundle = pipeline._target_cofold_msas([Sequence(_TARGET_A, "protein")])
        assert bundle.per_chain == {}  # degrades to single-sequence...
        assert "no homologs" in capsys.readouterr().out  # ...but says so

    def test_pre_redesign_cofold_forwards_target_msa(self):
        """The pre-redesign gate threads its target MSA into the cofold (so it isn't single-sequence)."""
        pipeline = _load_pipeline()
        sentinel: list[object] = [object()]  # stand-in for the campaign's target MSA bundle
        captured: dict[str, tuple] = {}

        class _StopAfterCofold(Exception):
            """Abort once the cofold call is captured, before the (real) relax/interface steps."""

        def _capture(*args, **kwargs):
            captured["args"] = args
            raise _StopAfterCofold

        with patch.object(pipeline, "structure_composite_constraint", _capture):
            with pytest.raises(_StopAfterCofold):
                pipeline.run_pre_redesign_external_filters(
                    binder_sequence=_E2E_BINDER,
                    target_sequence=_TARGET_A,
                    cofold_tool="chai1",
                    cofold_hotspots="",
                    cdr_positions_1idx=set(),
                    cdr3_positions_1idx=set(),
                    trajectory_seed=0,
                    precomputed_msas=sentinel,
                )
        # The supplied MSA bundle is forwarded as the cofold's precomputed_msas (3rd positional arg).
        assert captured["args"][2] is sentinel


@pytest.mark.uses_gpu
@pytest.mark.integration
@pytest.mark.slow
def test_target_only_msa_cofold_end_to_end():
    """End-to-end target-only cofold: live target MSA search + real GPU fold via the feature's own code.

    Unmocked counterpart to ``TestCofoldMsaMode``. Runs ``_target_cofold_msas`` (a live ColabFold
    search on the fixed target) and feeds the bundle through ``structure_composite_constraint`` with
    the production cofold config (``use_msa=False``): the target is conditioned on its MSA while the
    binder stays single-sequence, and no per-variant search runs. Requires a GPU and network.
    """
    pipeline = _load_pipeline()
    target = Sequence(_E2E_TARGET, "protein")
    binder = Sequence(_E2E_BINDER, "protein")

    # Step 1 — search the fixed target once; key its MSA to cofold chain index 1 (binder = 0, omitted).
    bundle = pipeline._target_cofold_msas([target])
    assert set(bundle.per_chain) == {1}, "target MSA must be keyed to chain 1, binder (0) omitted"
    assert len(bundle.per_chain[1].aligned_sequences) > 5, "expected a real, deep target MSA from the live search"
    assert bundle.per_chain[1].aligned_sequences[0].replace("-", "") == _E2E_TARGET

    # Step 2 — cofold the (binder, target) complex with the supplied target-only MSA on a real predictor.
    config = StructureBasedConstraintConfig.model_validate(pipeline._cofold_config("chai1", 0))
    [out] = structure_composite_constraint([(binder, target)], config, precomputed_msas=[bundle])

    assert out.structures[0] is not None, "cofold must populate the predicted structure"
    assert 0.0 <= out.score <= 1.0
    for key in ("composite_avg_plddt", "composite_iptm", "composite_ptm", "composite_avg_pae"):
        assert 0.0 <= out.metadata[key] <= 1.0, f"{key} out of range: {out.metadata[key]}"
