"""
Utility modules for proto-language.

This package provides various utility functions organized by category:
- metadata: Metadata propagation utilities
- structure: Structure and geometry utilities for molecular structures
- array: Array manipulation utilities
- compute: GPU and compute resource utilities
- file_resolution: File resolution and cloud storage utilities
"""


# Metadata utilities
from .metadata import propagate_metadata

# Structure utilities
from .structure import (
    pdb_file_to_atomarray,
    get_atomarray_in_residue_range,
    pairwise_distances,
    adjacent_distances,
    get_centroid,
    distances_to_centroid,
    get_backbone_atoms,
)

# Array utilities
from .array import top_k_indices

# Compute utilities
from .compute import (
    use_cloud_gpu,
    is_gpu_available,
)

# File resolution utilities
from .file_resolution import (
    resolve_file,
    resolve_paths,
    VOLUME_PATH,
    get_cache_path,
    download_gcs_file,
)

__all__ = [
    # Metadata
    "propagate_metadata",
    # Structure
    "pdb_file_to_atomarray",
    "get_atomarray_in_residue_range",
    "pairwise_distances",
    "adjacent_distances",
    "get_centroid",
    "distances_to_centroid",
    "get_backbone_atoms",
    # Array
    "top_k_indices",
    # Compute
    "use_cloud_gpu",
    "is_gpu_available",
    # File resolution
    "resolve_file",
    "resolve_paths",
    "VOLUME_PATH",
    "get_cache_path",
    "download_gcs_file",
]

