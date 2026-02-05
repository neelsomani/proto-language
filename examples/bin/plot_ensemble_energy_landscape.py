"""
Generate a 2D free energy landscape from MD ensemble data.

Usage:
    python plot_ensemble_energy_landscape.py <pdb1> <pdb2> <topology.pdb> <samples.xtc> [output.svg]

Example:
    python plot_ensemble_energy_landscape.py 1ake.pdb 4ake.pdb topology.pdb samples.xtc landscape.svg
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from proto_language.language.constraint.protein_structure.structure_ensemble_similarity_constraint import (
    _compute_ensemble_rmsds,
    _extract_chain_from_pdb,
)

mpl.rcParams['font.family'] = 'Liberation Sans'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2


def extract_pdb_id(filename: str) -> str:
    """Extract and uppercase PDB ID from filename like '1ake.pdb'."""
    return Path(filename).stem.upper()


def load_ensemble_frames(topology_path: str, trajectory_path: str) -> list[str]:
    """Load trajectory and return list of PDB strings for each frame."""
    import mdtraj as md
    import tempfile

    traj = md.load(trajectory_path, top=topology_path)
    frames = []
    for i in range(traj.n_frames):
        frame = traj[i]
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            frame.save_pdb(tmp.name)
            with open(tmp.name, 'r') as f:
                frames.append(f.read())
            Path(tmp.name).unlink()

    return frames


def free_energy_landscape(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    xlabel: str,
    ylabel: str,
    bins: int = 100,
    kT: float = 0.596,
    smoothing: float = 1.0,
    vmax: float = 5.0,
    output_path: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Generate a 2D free energy landscape plot.

    Parameters
    ----------
    x : array-like
        First collective variable (e.g., RMSD to state 1).
    y : array-like
        Second collective variable (e.g., RMSD to state 2).
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    bins : int
        Number of histogram bins per dimension.
    kT : float
        Thermal energy in kcal/mol (0.596 at 300K).
    smoothing : float
        Gaussian smoothing sigma for the histogram.
    vmax : float
        Maximum energy value for colorbar (kcal/mol).
    output_path : str or None
        If provided, save SVG to this path.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    xmax = 25#max(np.max(x), 30)
    ymax = 25#max(np.max(y), 30)

    # 2D histogram
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0, xmax], [0, ymax]],
        density=True,
    )
    H = gaussian_filter(H, sigma=smoothing)

    # Free energy
    H = np.where(H > 0, H, np.nan)
    F = -kT * np.log(H)
    F = F - np.nanmin(F)

    # Figure size: 19mm x 17mm = 0.748 x 0.669 inches plus some buffer.
    fig, ax = plt.subplots(figsize=(1.2, 0.8))

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Pastel colormap
    colors = [
        '#4A90A4', '#7FB5C5', '#B5D8E0', '#F5F5DC',
        '#F8D9C4', '#F4A582', '#D6604D'
    ]
    pastel_cmap = mpl.colors.LinearSegmentedColormap.from_list('pastel_energy', colors)

    levels = np.linspace(0, vmax, 20)
    cf = ax.contourf(X, Y, F.T, levels=levels, cmap=pastel_cmap, extend='max')
    ax.contour(X, Y, F.T, levels=levels[::4], colors='white', linewidths=0.3, alpha=0.6)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, aspect=10, pad=0.02)
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)

    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(labelsize=6)

    plt.tight_layout(pad=0.1)

    if output_path:
        fig.savefig(
            output_path,
            format='svg',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.01,
            transparent=False,
        )
        print(f"Saved to {output_path}")

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2D free energy landscape from MD ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdb1", help="First reference PDB file")
    parser.add_argument("pdb2", help="Second reference PDB file")
    parser.add_argument("topology", help="Topology PDB file for ensemble")
    parser.add_argument("trajectory", help="Trajectory XTC file")
    parser.add_argument(
        "output",
        nargs="?",
        default="free_energy_landscape.svg",
        help="Output SVG path (default: free_energy_landscape.svg)",
    )
    parser.add_argument(
        "--chain1",
        default=None,
        help="Chain ID to extract from first PDB (e.g., 'A'). If not specified, uses entire PDB.",
    )
    parser.add_argument(
        "--chain2",
        default=None,
        help="Chain ID to extract from second PDB (e.g., 'A'). If not specified, uses entire PDB.",
    )
    parser.add_argument(
        "--chain-ensemble",
        default=None,
        help="Chain ID to extract from ensemble frames. If not specified, uses entire structure.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins")
    parser.add_argument("--smoothing", type=float, default=1.5, help="Gaussian smoothing sigma")
    parser.add_argument("--vmax", type=float, default=4.5, help="Max energy for colorbar")
    parser.add_argument(
        "--selection",
        default="name CA",
        help="PyMOL selection for RMSD (default: 'name CA')",
    )

    args = parser.parse_args()

    # Extract PDB IDs for labels
    pdb1_id = extract_pdb_id(args.pdb1)
    pdb2_id = extract_pdb_id(args.pdb2)

    print(f"Reference structures: {pdb1_id}, {pdb2_id}")
    print(f"Loading ensemble from {args.topology} + {args.trajectory}...")

    # Load reference PDBs
    with open(args.pdb1, 'r') as f:
        pdb1_text = f.read()
    with open(args.pdb2, 'r') as f:
        pdb2_text = f.read()

    # Extract chains if specified
    if args.chain1:
        print(f"Extracting chain {args.chain1} from {pdb1_id}")
        pdb1_text = _extract_chain_from_pdb(pdb1_text, args.chain1)
    if args.chain2:
        print(f"Extracting chain {args.chain2} from {pdb2_id}")
        pdb2_text = _extract_chain_from_pdb(pdb2_text, args.chain2)

    # Load ensemble frames
    ensemble_frames = load_ensemble_frames(args.topology, args.trajectory)
    print(f"Loaded {len(ensemble_frames)} frames")

    # Extract chain from ensemble if specified
    if args.chain_ensemble:
        print(f"Extracting chain {args.chain_ensemble} from ensemble frames...")
        ensemble_frames = [
            _extract_chain_from_pdb(frame, args.chain_ensemble)
            for frame in ensemble_frames
        ]

    # Compute RMSDs to both references
    print(f"Computing RMSDs to {pdb1_id}...")
    rmsds_to_pdb1 = _compute_ensemble_rmsds(
        target_pdb_text=pdb1_text,
        ensemble_pdb_frames=ensemble_frames,
        target_selection=args.selection,
        mobile_selection=args.selection,
        verbose=True,
    )
    print(f"Min RMSD to {pdb1_id}: {min(rmsds_to_pdb1)}")

    print(f"Computing RMSDs to {pdb2_id}...")
    rmsds_to_pdb2 = _compute_ensemble_rmsds(
        target_pdb_text=pdb2_text,
        ensemble_pdb_frames=ensemble_frames,
        target_selection=args.selection,
        mobile_selection=args.selection,
        verbose=True,
    )
    print(f"Min RMSD to {pdb2_id}: {min(rmsds_to_pdb2)}")

    # Generate plot
    print("Generating free energy landscape...")
    fig, ax = free_energy_landscape(
        x=rmsds_to_pdb1,
        y=rmsds_to_pdb2,
        xlabel=f"RMSD to {pdb1_id} (Å)",
        ylabel=f"RMSD to {pdb2_id} (Å)",
        bins=args.bins,
        smoothing=args.smoothing,
        vmax=args.vmax,
        output_path=args.output,
    )

    plt.show()


if __name__ == "__main__":
    main()
