"""
Structure and geometry utilities for protein/molecular structures.

This module provides utilities for working with atomic coordinates,
PDB files, and molecular geometry calculations.
"""

from io import StringIO
from typing import Union

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    """Convert a PDB file to a Biotite AtomArray."""
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    """Extract atoms within a specific residue range."""
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]


def _is_Nx3(array: np.ndarray) -> bool:
    """Check if array is Nx3 shaped."""
    return len(array.shape) == 2 and array.shape[1] == 3


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances between all coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]


def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    """Calculate distances between adjacent coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)


def get_centroid(coordinates: np.ndarray) -> np.ndarray:
    """Calculate the centroid of coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Computes the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    centroid = get_centroid(coordinates)
    m = coordinates - centroid
    return np.linalg.norm(m, axis=-1)


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    """Extract backbone atoms (CA, N, C) from an AtomArray."""
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]

