"""
Program symmetric proteins with configurable parameters.

Usage:
    python program_symmetric_proteins.py --monomer-length 100 --n-symmetric-units 3 --n-steps 10000 --output-dir ./outputs
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any

from proto_language.language.constraint import (
    protein_globularity_constraint,
    protein_symmetry_ring_constraint,
    structure_plddt_constraint,
    structure_ptm_constraint,
)
from proto_language.language.core import (
    Constraint,
    Construct,
    Program,
    Segment,
    Sequence,
)
from proto_language.language.generator import (
    RandomProteinGenerator,
    RandomProteinGeneratorConfig,
)
from proto_language.language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
from proto_language.storage import get_file_content


def parse_args():
    parser = argparse.ArgumentParser(description="Design symmetric proteins using MCMC optimization")
    parser.add_argument(
        "--monomer-length",
        type=int,
        default=100,
        help="Length of each monomer unit (default: 100)",
    )
    parser.add_argument(
        "--n-symmetric-units",
        type=int,
        default=3,
        help="Number of symmetric units in the assembly (default: 3)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10000,
        help="Number of MCMC optimization steps (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Base output directory (default: ./outputs)",
    )
    return parser.parse_args()


def setup_output_dir(base_output_dir: str, monomer_length: int, n_symmetric_units: int, n_steps: int) -> str:
    """Create timestamped output directory with parameter info."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(
        base_output_dir, f"sym{n_symmetric_units}_len{monomer_length}_steps{n_steps}/run_{run_timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class LogCapture:
    """Capture logging output to both console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def _get_constraints_metadata(sequence: Sequence) -> dict[str, Any]:
    """Return constraint metadata across old/new sequence metadata layouts."""
    metadata_view = getattr(sequence, "metadata", None)
    if isinstance(metadata_view, dict):
        constraints = metadata_view.get("constraints")
        if isinstance(constraints, dict):
            return constraints

    constraints_metadata = getattr(sequence, "_constraints_metadata", None)
    if isinstance(constraints_metadata, dict):
        return constraints_metadata

    raw_metadata = getattr(sequence, "_metadata", None)
    if isinstance(raw_metadata, dict):
        constraints = raw_metadata.get("constraints")
        if isinstance(constraints, dict):
            return constraints

    return {}


def run_optimization(
    monomer_length: int,
    n_symmetric_units: int,
    n_steps: int,
    output_dir: str,
) -> None:
    """Run the symmetric protein optimization."""

    # Setup output directory
    run_dir = setup_output_dir(output_dir, monomer_length, n_symmetric_units, n_steps)
    print(f"Output directory: {run_dir}")

    # Setup logging
    log_path = os.path.join(run_dir, "optimization.log")
    log_capture = LogCapture(log_path)
    sys.stdout = log_capture

    try:
        # Log parameters
        print("=" * 60)
        print("Symmetric Protein Design")
        print("=" * 60)
        print("Parameters:")
        print(f"  Monomer length: {monomer_length}")
        print(f"  N symmetric units: {n_symmetric_units}")
        print(f"  N steps: {n_steps}")
        print(f"  Output directory: {run_dir}")
        print("=" * 60)

        #######################
        ## Segments ##
        #######################

        protomer_segments = [
            Segment(
                length=monomer_length,
                sequence_type="protein",
                label=f"protomer_{idx + 1}",
            )
            for idx in range(n_symmetric_units)
        ]

        ################
        ## Constructs ##
        ################

        protomer_constructs = [
            Construct([segment], label=f"protomer_{idx + 1}") for idx, segment in enumerate(protomer_segments)
        ]

        ################
        ## Generators ##
        ################

        uniform_gen_config = RandomProteinGeneratorConfig()
        uniform_gen = RandomProteinGenerator(uniform_gen_config)
        uniform_gen.assign(protomer_segments)

        #################
        ## Constraints ##
        #################

        symmetric_complex = protomer_segments

        esmfold_plddt = Constraint(
            inputs=symmetric_complex,
            function=structure_plddt_constraint,
            function_config={"structure_tool": "esmfold"},
            weight=1,
        )

        esmfold_ptm = Constraint(
            inputs=symmetric_complex,
            function=structure_ptm_constraint,
            function_config={"structure_tool": "esmfold"},
            weight=1,
        )

        symmetry = Constraint(
            inputs=symmetric_complex,
            function=protein_symmetry_ring_constraint,
            function_config={
                "all_to_all_protomer_symmetry": True,
            },
            weight=1,
        )

        globularity = Constraint(
            inputs=symmetric_complex,
            function=protein_globularity_constraint,
            function_config={},
            weight=5,
        )

        #################
        ## Custom Logging ##
        #################

        def custom_logging(step: int, outputs: tuple[Segment]) -> None:
            output_sequences = [segment.result_sequences[0] for segment in outputs]
            constraints = _get_constraints_metadata(output_sequences[0])

            # Get pLDDT from structure_plddt_constraint
            plddt = constraints.get("structure_plddt_constraint", {}).get("data", {}).get("avg_plddt")

            # Get pTM from structure_ptm_constraint
            ptm = constraints.get("structure_ptm_constraint", {}).get("data", {}).get("ptm")

            # Get esmfolded_sequence from symmetry constraint
            folded_sequence = (
                constraints.get("protein_symmetry_ring_constraint", {})
                .get("data", {})
                .get(
                    "esmfolded_sequence",
                    "N/A",
                )
            )

            print(
                f"Iteration {step} | \n"
                f"\tsequence (monomer): {output_sequences[0].sequence}, \n"
                f"\tsequence (folded complex): {folded_sequence}, \n"
                f"\tpLDDT: {plddt}, \n"
                f"\tpTM: {ptm}"
            )

        #############
        ## Program ##
        #############

        mcmc_optimizer_config = MCMCOptimizerConfig(
            num_steps=n_steps,
            max_temperature=1.0,
            min_temperature=0.0001,
            verbose=True,
        )

        # Create optimizer
        optimizer = MCMCOptimizer(
            constructs=protomer_constructs,
            generators=[uniform_gen],
            constraints=[
                esmfold_plddt,
                esmfold_ptm,
                symmetry,
                globularity,
            ],
            config=mcmc_optimizer_config,
            custom_logging=custom_logging,
        )

        # Create program with optimizer
        program = Program(optimizers=[optimizer], num_results=1)

        program.run()

        #################
        ## Save Outputs ##
        #################

        # Get sequences from the protomer segments (where constraint metadata is stored)
        protomer_sequences: list[Sequence] = [segment.result_sequences[0] for segment in protomer_segments]
        constraints = _get_constraints_metadata(protomer_sequences[0])
        if not constraints:
            raise RuntimeError("No constraint metadata found on final protomer sequences.")

        # Save PDB (stored as file reference, need to retrieve content)
        symmetry_data = constraints.get("protein_symmetry_ring_constraint", {}).get("data", {})
        pdb_ref = symmetry_data.get("pdb_output")
        if not pdb_ref:
            raise RuntimeError("Missing PDB output reference in protein_symmetry_ring_constraint metadata.")
        pdb_content = get_file_content(pdb_ref)
        pdb_path = os.path.join(run_dir, "design.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_content)
        print(f"Saved PDB to: {pdb_path}")

        # Extract final metrics
        final_plddt = constraints.get("structure_plddt_constraint", {}).get("data", {}).get("avg_plddt")
        final_ptm = constraints.get("structure_ptm_constraint", {}).get("data", {}).get("ptm")
        folded_sequence = symmetry_data.get("esmfolded_sequence")

        # Save final sequence and metrics
        seq_path = os.path.join(run_dir, "final_sequence.txt")
        with open(seq_path, "w") as f:
            f.write("# Symmetric Protein Design Result\n")
            f.write(f"# Monomer length: {monomer_length}\n")
            f.write(f"# N symmetric units: {n_symmetric_units}\n")
            f.write(f"# N steps: {n_steps}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write("\n")
            f.write("Protomer sequences:\n")
            for idx, protomer_sequence in enumerate(protomer_sequences):
                f.write(f"Protomer {idx + 1}: {protomer_sequence.sequence}\n")
            f.write(f"\nFolded sequence (complex):\n{folded_sequence}\n")
            f.write(f"\nFinal pLDDT: {final_plddt}\n")
            f.write(f"Final pTM: {final_ptm}\n")
        print(f"Saved sequence to: {seq_path}")

        print(f"\n{'=' * 60}")
        print("Optimization complete!")
        print(f"Results saved to: {run_dir}")
        print(f"{'=' * 60}")

    finally:
        # Restore stdout and close log
        sys.stdout = log_capture.stdout
        log_capture.close()
        print(f"Log saved to: {log_path}")


def main():
    args = parse_args()
    run_optimization(
        monomer_length=args.monomer_length,
        n_symmetric_units=args.n_symmetric_units,
        n_steps=args.n_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
