"""Generate intron_design_alphagenome.json with real DNA sequences baked in.

Reads data files (plasmid contexts, gene sequences, AlphaGenome genomic contexts)
and builds a complete JSON program specification that can be imported directly
into the client or run via the API parser.

Usage:
    python examples/scripts/generate_intron_design_alphagenome_json.py \
        --output examples/jsons/intron_design_alphagenome.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from examples.scripts.program_intron_design import (
    get_initial_intron,
    process_splice_transformer_input,
)

# Data file paths (relative to repo root).
PLASMID_CONTEXT_PATHS = [
    "examples/data/plasmid_context_cmv_20260308.txt",
    "examples/data/plasmid_context_Ef1a.txt",
    "examples/data/plasmid_context_sffv.txt",
]
GENE_SEQUENCE_PATH = "examples/data/mscarlet_ires_zsgreen.txt"
GENOMIC_CONTEXT_PATHS = [
    "examples/data/alphagenome_context_aavs1.txt",
    "examples/data/alphagenome_context_ccr5.txt",
    "examples/data/alphagenome_context_clybl.txt",
    "examples/data/alphagenome_context_hrosa26.txt",
]

# Design defaults.
INTRON_LENGTH = 301
GENE_INSERTION_POS = 159 * 3
INITIALIZATION = "random"


def _read_sequence(path: str) -> str:
    return Path(path).read_text().strip().upper()


def _build_json() -> Dict[str, Any]:
    """Build the complete JSON program specification."""
    from types import SimpleNamespace

    # Use a fixed seed intron for reproducible JSON generation.
    intron_args = SimpleNamespace(initialization=INITIALIZATION, intron_length=INTRON_LENGTH)
    initial_intron = get_initial_intron(intron_args)

    # We'll use the first plasmid context to define the construct segments.
    first_plasmid = PLASMID_CONTEXT_PATHS[0]
    splice_args = SimpleNamespace(
        plasmid_context_path=first_plasmid,
        gene_sequence_path=GENE_SEQUENCE_PATH,
        gene_insertion_pos=GENE_INSERTION_POS,
    )
    (
        _left_ctx, _right_ctx, target_seq,
        _gene_start, _gene_end,
        donor_start_pos, acceptor_end_pos,
    ) = process_splice_transformer_input(initial_intron, splice_args)

    # Segment boundaries within the 1kb target.
    left_flank_seq = target_seq[:donor_start_pos + 2]
    intron_core_seq = target_seq[donor_start_pos + 2:acceptor_end_pos - 1]
    right_flank_seq = target_seq[acceptor_end_pos - 1:]

    # Splice evaluation positions (target-relative).
    donor_eval_pos = donor_start_pos - 1
    acceptor_eval_pos = acceptor_end_pos + 1
    splice_pos = [donor_eval_pos, acceptor_eval_pos]

    # Build constructs.
    constructs = [
        {
            "id": "intron_construct",
            "type": "dna",
            "label": "Intron in expression cassette (1kb SpliceTransformer target)",
            "segments": [
                {"id": "left_flank", "label": "Left flank", "sequence": left_flank_seq},
                {"id": "intron", "label": "Intron core (optimizable)", "sequence": intron_core_seq},
                {"id": "right_flank", "label": "Right flank", "sequence": right_flank_seq},
            ],
        }
    ]

    # Build constraints across all plasmid x genomic context combinations.
    constraints: List[Dict[str, Any]] = []

    for context_idx, plasmid_path in enumerate(PLASMID_CONTEXT_PATHS):
        plasmid_label = Path(plasmid_path).stem
        splice_args = SimpleNamespace(
            plasmid_context_path=plasmid_path,
            gene_sequence_path=GENE_SEQUENCE_PATH,
            gene_insertion_pos=GENE_INSERTION_POS,
        )
        (
            left_context, right_context, _target_seq,
            _gene_start, _gene_end,
            _donor_start, _acceptor_end,
        ) = process_splice_transformer_input(initial_intron, splice_args)

        # SpliceTransformer boundary constraint.
        constraints.append({
            "id": f"splice_boundary_{plasmid_label}_{context_idx}",
            "key": "splice-transformer-intron-boundary",
            "label": f"splice_boundary__{plasmid_label}__{context_idx}",
            "targets": ["left_flank", "intron", "right_flank"],
            "config": {
                "left_context": left_context,
                "right_context": right_context,
                "donor_pos": [donor_eval_pos],
                "acceptor_pos": [acceptor_eval_pos],
            },
        })

        # SpliceTransformer specificity constraints (brain max, blood min).
        for tissue, direction in [("BRAIN", "max"), ("BLOOD", "min")]:
            constraints.append({
                "id": f"splice_specificity_{tissue.lower()}_{direction}_{plasmid_label}_{context_idx}",
                "key": "splice-transformer-specificity",
                "label": f"splice_specificity_{tissue.lower()}_{direction}__{plasmid_label}__{context_idx}",
                "targets": ["left_flank", "intron", "right_flank"],
                "config": {
                    "left_context": left_context,
                    "right_context": right_context,
                    "splice_pos": splice_pos,
                    "tissue": tissue,
                    "direction": direction,
                },
            })

        # AlphaGenome SSU constraints across genomic contexts.
        for genomic_idx, genomic_path in enumerate(GENOMIC_CONTEXT_PATHS):
            genomic_label = Path(genomic_path).stem.replace("alphagenome_context_", "")
            genomic_context = _read_sequence(genomic_path)

            for tissue_label, ontology, direction in [
                ("brain", "CL:0002319", "max"),
                ("blood", "EFO:0002067", "min"),
            ]:
                constraints.append({
                    "id": f"ag_ssu_{tissue_label}_{direction}_{plasmid_label}_{context_idx}_{genomic_label}_{genomic_idx}",
                    "key": "alphagenome-splice-site-usage",
                    "label": (
                        f"alphagenome_ssu_{tissue_label}_{direction}__"
                        f"{plasmid_label}__{genomic_label}__{context_idx}_{genomic_idx}"
                    ),
                    "targets": ["left_flank", "intron", "right_flank"],
                    "weight": 1.0,
                    "config": {
                        "genomic_context": genomic_context,
                        "cassette_left_context": left_context,
                        "cassette_right_context": right_context,
                        "ontology_terms": [ontology],
                        "splice_pos": splice_pos,
                        "direction": direction,
                        "strand": "positive",
                        "model_version": "all_folds",
                        "organism": "human",
                    },
                })

    program = {
        "name": "intron_design_alphagenome",
        "description": (
            "Intron design with SpliceTransformer and AlphaGenome constraints "
            "for tissue-specific splicing optimization."
        ),
        "version": "1.0",
        "num_results": 1,
        "constructs": constructs,
        "optimization_stages": [
            {
                "optimizer": {
                    "method": "mcmc",
                    "config": {
                        "num_steps": 5000,
                        "num_results": 1,
                        "max_temperature": 1.0,
                        "min_temperature": 0.001,
                        "verbose": True,
                    },
                },
                "generators": [
                    {
                        "id": "intron_mutator",
                        "key": "uniform-mutation",
                        "target": "intron",
                        "config": {"num_mutations": 1},
                    }
                ],
                "constraints": constraints,
            }
        ],
    }
    return program


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate intron_design_alphagenome.json")
    parser.add_argument(
        "--output",
        default="examples/jsons/intron_design_alphagenome.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    program_json = _build_json()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(program_json, indent=2) + "\n")
    print(f"Generated {output_path} ({output_path.stat().st_size:,} bytes)")
