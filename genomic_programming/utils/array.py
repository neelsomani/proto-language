"""
Array manipulation utilities for proto-language.

This module provides utilities for working with NumPy arrays,
including sorting and selection operations.
"""

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-k values in the scores vector.

    Args:
        scores (np.ndarray): 1D array of scores.
        k (int): number of top elements to return.

    Returns:
        np.ndarray: Array of indices of the top-k scores.
    """
    # np.argpartition is more efficient than sorting the entire array
    # when we only need the top-k elements
    if k >= len(scores):
        # If k is larger than the array length, return all indices in sorted order
        return np.argsort(scores)[::-1]

    # Get indices of top-k elements
    # The negative sign is because we want the largest values (descending order)
    top_k_idx = np.argpartition(scores, -k)[-k:]

    # Sort these top-k indices by their corresponding values (highest first)
    top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]

    return top_k_idx

