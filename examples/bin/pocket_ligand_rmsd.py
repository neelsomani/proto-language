"""
Compute pocket-aligned RMSD of ATP ligands using PyMOL's CEAlign.

Usage:
    python pocket_ligand_rmsd.py reference.pdb design1.pdb design2.pdb
    python pocket_ligand_rmsd.py reference.pdb designs/*.pdb --ligand ATP --pocket-distance 8.0
    python pocket_ligand_rmsd.py reference.pdb -g "designs/round*/*.pdb" --ligand ANP
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pymol
from pymol import cmd


def init_pymol():
    """Initialize PyMOL in headless mode."""
    pymol.finish_launching(["pymol", "-qc"])


def get_pocket_selection(obj_name, ligand_name, distance):
    """Return a selection string for pocket residues near ligand."""
    return f"({obj_name} and polymer.protein) within {distance} of ({obj_name} and resn {ligand_name})"


def compute_ligand_rmsd_pymol(ref_pdb, design_pdb, ligand_name, pocket_distance):
    """
    Align design to reference using CEAlign on pocket residues,
    then compute ligand RMSD.
    """
    cmd.delete("all")

    cmd.load(ref_pdb, "ref")
    cmd.load(design_pdb, "design")

    # Check ligands exist
    ref_lig_count = cmd.count_atoms(f"ref and resn {ligand_name}")
    design_lig_count = cmd.count_atoms(f"design and resn {ligand_name}")

    if ref_lig_count == 0:
        raise ValueError(f"No {ligand_name} found in reference")
    if design_lig_count == 0:
        raise ValueError(f"No {ligand_name} found in design")

    # Define pocket selections
    ref_pocket = get_pocket_selection("ref", ligand_name, pocket_distance)
    design_pocket = get_pocket_selection("design", ligand_name, pocket_distance)

    cmd.select("ref_pocket", ref_pocket)
    cmd.select("design_pocket", design_pocket)

    ref_pocket_count = cmd.count_atoms("ref_pocket and name CA")
    design_pocket_count = cmd.count_atoms("design_pocket and name CA")

    print(f"  Reference pocket: {ref_pocket_count} CA atoms")
    print(f"  Design pocket: {design_pocket_count} CA atoms")

    # CEAlign design pocket onto reference pocket
    cealign_result = cmd.cealign("ref_pocket", "design_pocket")

    alignment_rmsd = cealign_result.get("RMSD", None)
    alignment_length = cealign_result.get("alignment_length", None)

    print(f"  CEAlign RMSD: {alignment_rmsd:.3f} Å over {alignment_length} residues")

    # Get ligand atom coordinates after alignment
    ref_atoms = {}
    cmd.iterate_state(1, f"ref and resn {ligand_name}", "ref_atoms[name] = (x, y, z)", space={"ref_atoms": ref_atoms})

    design_atoms = {}
    cmd.iterate_state(
        1, f"design and resn {ligand_name}", "design_atoms[name] = (x, y, z)", space={"design_atoms": design_atoms}
    )

    # Compute ligand RMSD
    common_atoms = set(ref_atoms.keys()) & set(design_atoms.keys())

    if not common_atoms:
        raise ValueError("No common atoms between ligands")

    squared_dists = []
    for atom_name in common_atoms:
        ref_coord = np.array(ref_atoms[atom_name])
        design_coord = np.array(design_atoms[atom_name])
        squared_dists.append(np.sum((ref_coord - design_coord) ** 2))

    ligand_rmsd = np.sqrt(np.mean(squared_dists))

    return {
        "ligand_rmsd": ligand_rmsd,
        "alignment_rmsd": alignment_rmsd,
        "alignment_length": alignment_length,
        "n_ligand_atoms": len(common_atoms),
        "ref_pocket_size": ref_pocket_count,
        "design_pocket_size": design_pocket_count,
    }


def expand_paths(path_args, glob_patterns=None):
    """Expand paths from direct arguments and glob patterns."""
    paths = []

    # Direct paths (shell may have already expanded wildcards)
    for p in path_args:
        if "*" in p or "?" in p:
            # Shell didn't expand, do it manually
            expanded = glob.glob(p, recursive=True)
            paths.extend(expanded)
        else:
            paths.append(p)

    # Additional glob patterns from -g/--glob
    if glob_patterns:
        for pattern in glob_patterns:
            expanded = glob.glob(pattern, recursive=True)
            paths.extend(expanded)

    # Remove duplicates, preserve order
    seen = set()
    unique_paths = []
    for p in paths:
        p_resolved = str(Path(p).resolve())
        if p_resolved not in seen:
            seen.add(p_resolved)
            unique_paths.append(p)

    return sorted(unique_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pocket-aligned ligand RMSD using PyMOL CEAlign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s reference.pdb design1.pdb design2.pdb
    %(prog)s reference.pdb designs/*.pdb --ligand ATP
    %(prog)s reference.pdb -g "designs/**/*.pdb" --pocket-distance 8.0
    %(prog)s ref.pdb -g "round1/*.pdb" -g "round2/*.pdb" --ligand ANP
        """,
    )

    parser.add_argument("reference", help="Reference PDB file with ligand")
    parser.add_argument("designs", nargs="*", help="Design PDB files (supports shell wildcards)")
    parser.add_argument(
        "-g",
        "--glob",
        action="append",
        dest="glob_patterns",
        metavar="PATTERN",
        help="Glob pattern for design PDBs (can be used multiple times, supports **)",
    )
    parser.add_argument("-l", "--ligand", default="ATP", help="Ligand residue name (default: ATP)")
    parser.add_argument(
        "-d",
        "--pocket-distance",
        type=float,
        default=10.0,
        help="Distance cutoff for pocket residues in Angstroms (default: 10.0)",
    )
    parser.add_argument("-o", "--output", default="pocket_ligand_rmsd_output.csv", help="Output CSV file for results")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-structure output")

    args = parser.parse_args()

    # Expand design paths
    design_pdbs = expand_paths(args.designs or [], args.glob_patterns)

    if not design_pdbs:
        parser.error("No design PDB files specified. Use positional arguments or -g/--glob.")

    # Verify reference exists
    if not Path(args.reference).exists():
        parser.error(f"Reference file not found: {args.reference}")

    print(f"Reference: {args.reference}")
    print(f"Ligand: {args.ligand}")
    print(f"Pocket distance: {args.pocket_distance} Å")
    print(f"Design files: {len(design_pdbs)}")
    print()

    init_pymol()

    results = []

    for design_path in design_pdbs:
        if not args.quiet:
            print(f"Processing: {design_path}")

        try:
            result = compute_ligand_rmsd_pymol(args.reference, design_path, args.ligand, args.pocket_distance)
            result["design"] = design_path
            result["status"] = "success"

            if not args.quiet:
                print(f"  Ligand RMSD: {result['ligand_rmsd']:.3f} Å ({result['n_ligand_atoms']} atoms)\n")

            results.append(result)

        except Exception as e:
            if not args.quiet:
                print(f"  ERROR: {e}\n")
            results.append(
                {
                    "design": design_path,
                    "ligand_rmsd": None,
                    "alignment_rmsd": None,
                    "alignment_length": None,
                    "n_ligand_atoms": None,
                    "ref_pocket_size": None,
                    "design_pocket_size": None,
                    "status": f"error: {e}",
                }
            )

    # Summary table
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Design':<35} {'Lig RMSD':>10} {'Align RMSD':>11} {'Align Len':>10}")
    print("-" * 75)

    successful = 0
    for r in results:
        name = Path(r["design"]).name
        if r["status"] == "success":
            print(f"{name:<35} {r['ligand_rmsd']:>10.3f} {r['alignment_rmsd']:>11.3f} {r['alignment_length']:>10}")
            successful += 1
        else:
            print(f"{name:<35} {'FAILED':>10}")

    print("-" * 75)
    print(f"Processed: {len(results)} | Successful: {successful} | Failed: {len(results) - successful}")

    # Write CSV if requested
    if args.output:
        import csv

        with open(args.output, "w", newline="") as f:
            fieldnames = [
                "design",
                "ligand_rmsd",
                "alignment_rmsd",
                "alignment_length",
                "n_ligand_atoms",
                "ref_pocket_size",
                "design_pocket_size",
                "status",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {args.output}")

    cmd.quit()
    return results


if __name__ == "__main__":
    main()
