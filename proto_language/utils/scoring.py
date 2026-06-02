"""Constraint scoring math: range deviation, GC content, sigmoid/softmax transforms."""

import math

import numpy as np

MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
LOG_BASE = 2

MIN_GC_CONTENT = 0.0
MAX_GC_CONTENT = 100.0


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value falls within the specified range.

    Args:
        value (float): The value to validate.
        min_val (float): Minimum acceptable value (inclusive).
        max_val (float): Maximum acceptable value (inclusive).
        name (str): Name of the parameter for error messages.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def calculate_range_deviation(actual: float, min_val: float, max_val: float, epsilon: float = 1) -> float:
    """Calculate deviation from acceptable range for general constraints.

    Args:
        actual (float): The actual measured value.
        min_val (float): Minimum acceptable value.
        max_val (float): Maximum acceptable value.
        epsilon (float): Floor for the denominator to avoid division by zero.
            Use 1 (default) for integer-scale values, 1e-9 for fractional values.

    Returns:
        float: Range deviation score where 0.0 indicates the value is within range
            and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    if actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, epsilon))
    return min(MAX_ENERGY, (actual - max_val) / max(max_val, epsilon))


def calculate_percentage_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """Calculate deviation from acceptable range for percentage-based constraints (0-100%).

    Args:
        actual (float): The actual measured percentage value.
        min_val (float): Minimum acceptable percentage.
        max_val (float): Maximum acceptable percentage.

    Returns:
        float: Percentage range deviation score where 0.0 indicates the value is within range
            and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    if actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content percentage of a DNA/RNA sequence.

    Args:
        sequence (str): DNA or RNA sequence string.

    Returns:
        float: GC content as a percentage (0-100).
    """
    if not sequence:
        return 0.0

    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count("G") + sequence_upper.count("C")
    return 100.0 * gc_count / len(sequence)


def calculate_normalized_deviation(actual: float, target: float) -> float:
    """Calculate normalized deviation from target value for target-based constraints.

    Args:
        actual (float): The actual measured value.
        target (float): The desired target value.

    Returns:
        float: Normalized deviation score where 0.0 indicates perfect match
            and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def softmax(matrix: np.ndarray) -> np.ndarray:
    """Compute numerically stable row-wise softmax."""
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(shifted)
    result = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    assert isinstance(result, np.ndarray)  # noqa: S101 -- narrows numpy scalar arithmetic for mypy
    return result


def sigmoid_score(
    metric: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """Squeezes a metric into a 0-1 score using a sigmoid function.

    Args:
        metric (float): A metric value.
        inflection (float): The value of the original metric where the transformed score
            would be 0.5.
        slope (float): The steepness of the curve. Default: 3.0.

    Returns:
        float: Score between 0.0 (good/low) and 1.0 (bad/high).
    """
    scaled = slope * (metric - inflection)
    if scaled >= 0.0:
        z = math.exp(-scaled)
        return 1.0 / (1.0 + z)
    z = math.exp(scaled)
    return z / (1.0 + z)


def inverse_sigmoid_score(
    score: float,
    inflection: float,
    slope: float = 3.0,
) -> float:
    """Invert ``sigmoid_score`` via the **logit function** to recover the original metric.

    Args:
        score (float): A score value strictly between 0.0 and 1.0.
        inflection (float): The value of the original metric where the transformed score
                    is 0.5.
        slope (float): The steepness of the curve. Default: 3.0.

    Returns:
        float: The recovered metric value.
    """
    if score <= 0.0 or score >= 1.0:
        raise ValueError(f"Input score must be strictly between 0 and 1, found {score}")

    if slope == 0:
        raise ValueError("Slope cannot be zero for inversion.")

    return float(inflection + (np.log(score / (1.0 - score)) / slope))
