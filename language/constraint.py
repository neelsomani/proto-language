import itertools
from io import StringIO
import numpy as np
import pandas as pd
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import warnings

from .base import ProgramConstraint, ProgramSequence


def sequence_length_constraint(
    inputs: List[ProgramSequence], config: Dict[str, Any]
) -> float:
    """
    Scoring function for the sequence length constraint.

    Args:
        inputs (List[ProgramSequence]): The input sequences.
        config (Dict[str, Any]): Configuration parameters including:
            - target_length (int): The targeted length.

    Returns:
        float: An energy score between 0.0 and 1.0
    """
    if "target_length" not in config:
        raise ValueError("target_length must be specified in config")
    target_length = config["target_length"]

    if len(inputs) > 1:
        warnings.warn(
            "Input is a list of sequences. Concatenating for length calculation."
        )

    for seq in inputs:
        seq._metadata["length"] = len(seq)

    # Calculate deviation based on total length.
    full_length = len("".join(str(seq) for seq in inputs))
    if full_length == target_length:
        return 0.0

    # Calculate normalized deviation from target length.
    deviation = abs(full_length - target_length) / target_length
    return min(1.0, deviation)


def gc_content_constraint(
    inputs: List[ProgramSequence], config: Dict[str, Any]
) -> float:
    """
    Evaluates a constraint on GC content to be within a target range.

    Args:
        inputs (List[ProgramSequence]): The input sequences.
        config (Dict[str, Any]): Configuration parameters including:
            - min_gc (float): Minimum acceptable GC content percentage (default: 30.0)
            - max_gc (float): Maximum acceptable GC content percentage (default: 60.0)

    Returns:
        float: An energy score between 0.0 and 1.0
    """
    if "min_gc" not in config:
        raise ValueError("min_gc must be specified in config")
    if "max_gc" not in config:
        raise ValueError("max_gc must be specified in config")
    min_gc = config["min_gc"]
    max_gc = config["max_gc"]

    # Validate range.
    if min_gc < 0 or max_gc > 100:
        raise ValueError("GC content range must be between 0 and 100 percent.")

    if len(inputs) > 1:
        warnings.warn(
            "Input is a list of sequences. Concatenating for GC content calculation."
        )

    sequence = "".join(str(sequence) for sequence in inputs)

    # Calculate GC content.
    gc_content = (
        100.0 * sum(nt in "GC" for nt in sequence.upper()) / max(len(sequence), 1)
    )

    if len(inputs) == 1:
        inputs[0]._metadata["gc_content"] = gc_content

    # Return 0.0 if GC content is within the range.
    if min_gc <= gc_content <= max_gc:
        return 0.0
    else:
        if gc_content < min_gc:
            deviation = (min_gc - gc_content) / min_gc
        else:
            deviation = (gc_content - max_gc) / (100 - max_gc)
        return min(1.0, deviation)


def max_homopolymer_constraint(
    inputs: List[ProgramSequence], config: Dict[str, Any]
) -> float:
    """
    Evaluates a constraint that penalizes homopolymers longer than the specified maximum length.

    Args:
        inputs (List[ProgramSequence]): The input sequences.
        config (Dict[str, Any]): Configuration parameters including:
            - max_length (int): Maximum allowed homopolymer length (default: 10)

    Returns:
        float: An energy score between 0.0 and 1.0
    """
    if "max_length" not in config:
        raise ValueError("max_length must be specified in config")
    max_length = config["max_length"]

    if len(inputs) > 1:
        warnings.warn(
            "Input is a list of sequences. Concatenating for homopolymer calculation."
        )

    sequence = "".join(str(sequence) for sequence in inputs)

    if len(sequence) <= 1:
        # Edge case.
        longest_homopolymer = len(sequence)
    else:
        # Find length of each homopolymer.
        homopolymer_lengths = [
            len(list(group)) for _, group in itertools.groupby(sequence)
        ]
        longest_homopolymer = max(homopolymer_lengths)

    if len(inputs) == 1:
        inputs[0]._metadata["max_homopolymer_length"] = longest_homopolymer

    # Return 0.0 if longest homopolymer is within range.
    if longest_homopolymer <= max_length:
        return 0.0
    else:
        # Use a logarithmic scale for scoring.
        excess_length = longest_homopolymer - max_length
        log_ratio = np.log(1 + excess_length / max_length) / np.log(2)
        return min(1.0, log_ratio)


def dinucleotide_frequency_constraint(
    inputs: List[ProgramSequence], config: Dict[str, Any]
) -> float:
    """
    Evaluates a constraint on dinucleotide frequencies to be within a target range.

    Args:
        inputs (List[ProgramSequence]): The input sequences.
        config (Dict[str, Any]): Configuration parameters including:
            - min_freq (float): Minimum acceptable frequency for each dinucleotide (default: 0.03)
            - max_freq (float): Maximum acceptable frequency for each dinucleotide (default: 0.08)

    Returns:
        float: An energy score between 0.0 and 1.0
    """
    if "min_freq" not in config:
        raise ValueError("min_freq must be specified in config")
    min_freq = config["min_freq"]

    if "max_freq" not in config:
        raise ValueError("max_freq must be specified in config")
    max_freq = config["max_freq"]

    assert len(inputs) == 1 and inputs[0].sequence_type in {
        "dna",
        "rna",
    }, "Input must be a DNA or RNA sequence"

    sequence = inputs[0]

    # Edge case.
    if len(sequence) < 2:
        inputs[0]._metadata["dinucleotide_freqs"] = {}
        return 1.0

    # Determine valid nucleotides.
    valid_nucleotides = "ATCG" if sequence.sequence_type == "dna" else "AUCG"

    # Precompute dinucleotides.
    dinucleotides = [
        "".join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)
    ]

    # Count dinucleotides.
    dinucleotide_counts = {}
    total_count = 0
    for i in range(len(sequence) - 1):
        dinuc = str(sequence)[i : i + 2]
        if all(nt in valid_nucleotides for nt in dinuc):
            dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
            total_count += 1

    # If no valid dinucleotides found.
    if total_count == 0:
        inputs[0]._metadata["dinucleotide_freqs"] = {}
        return 1.0

    # Calculate frequencies and check if they are in range.
    max_deviation = 0.0
    dinucleotide_freqs = {}

    # Score based on deviation from target dinucleotide frequencies.
    for dinuc in dinucleotides:
        freq = dinucleotide_counts.get(dinuc, 0) / total_count
        dinucleotide_freqs[dinuc] = freq

        # Calculate deviation if outside acceptable range.
        if freq < min_freq:
            deviation = (min_freq - freq) / min_freq
            max_deviation = max(max_deviation, deviation)
        elif freq > max_freq:
            deviation = (freq - max_freq) / (1.0 - max_freq)
            max_deviation = max(max_deviation, deviation)

    inputs[0]._metadata["dinucleotide_freqs"] = dinucleotide_freqs
    return min(1.0, max_deviation)


def tetranucleotide_usage_constraint(
    inputs: List[ProgramSequence], config: Dict[str, Any]
) -> float:
    """
    Evaluates a constraint on tetranucleotide usage deviation (TUD) to be within a target range.

    Args:
        inputs (List[ProgramSequence]): The input sequences.
        config (Dict[str, Any]): Configuration parameters including:
            - tetranucleotide (str): The specific 4-base sequence to analyze
            - min_tud (float): Minimum acceptable TUD value (default: 0.8)
            - max_tud (float): Maximum acceptable TUD value (default: 1.2)

    Returns:
        float: An energy score between 0.0 and 1.0
    """
    if "tetranucleotide" not in config:
        raise ValueError("tetranucleotide must be specified in config")
    tetranucleotide = config["tetranucleotide"].upper()

    if "min_tud" not in config:
        raise ValueError("min_tud must be specified in config")
    min_tud = config["min_tud"]

    if "max_tud" not in config:
        raise ValueError("max_tud must be specified in config")
    max_tud = config["max_tud"]

    # Validate tetranucleotide input.
    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

    assert len(inputs) == 1 and inputs[0].sequence_type in {
        "dna",
        "rna",
    }, "Input must be a DNA or RNA sequence"

    sequence = inputs[0]

    # Set appropriate nucleotide keys based on sequence type.
    nucleotide_keys = (
        ["A", "T", "C", "G"]
        if sequence.sequence_type == "dna"
        else ["A", "U", "C", "G"]
    )

    # Edge case.
    if len(sequence) < 4:
        inputs[0]._metadata[tetranucleotide + "_tud"] = 0.0
        return 0.0

    # Calculate nucleotide frequencies.
    nucleotide_freqs = {}
    seq_length = len(sequence)
    for nt in nucleotide_keys:
        nucleotide_freqs[nt] = str(sequence).count(nt) / seq_length

    # Count occurrences of tetranucleotide.
    tetra_count = 0
    for i in range(len(sequence) - 3):
        if str(sequence)[i : i + 4] == tetranucleotide:
            tetra_count += 1

    # Calculate expected frequency using zero-order Markov model.
    tetra_expected_freq = 1.0
    for nt in tetranucleotide:
        if nt in nucleotide_freqs:
            tetra_expected_freq *= nucleotide_freqs[nt]
        else:
            # If invalid nucleotide, set to 0
            tetra_expected_freq = 0
            break

    # Calculate expected occurrences and TUD.
    expected_occurrences = tetra_expected_freq * (seq_length - 3)
    tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
    inputs[0]._metadata[tetranucleotide + "_tud"] = tetra_tud

    # Score based on TUD range.
    if min_tud <= tetra_tud <= max_tud:
        return 0.0
    else:
        # Calculate normalized deviation.
        if tetra_tud < min_tud:
            deviation = (min_tud - tetra_tud) / min_tud
        else:
            deviation = (tetra_tud - max_tud) / max_tud
        return min(1.0, deviation)


def _run_esmfold(
    input_sequence: ProgramSequence,
    n_replications: int = 1,
    esmfold_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Runs ESMFold on a ProgramSequence and store result in metadata.

    Also ensure the protein sequence is replicated correctly before ESMFold evaluation.

    Raises:
        ValueError: If the ProgramSequence is not a protein.
    """
    from .tools.structure_prediction import esmfold_protein_sequence

    if input_sequence.sequence_type != "protein":
        raise ValueError("Can only run ESMFold on a protein sequence.")

    esmfolded_sequence = ":".join([input_sequence.sequence] * n_replications)

    if "esmfolded_sequence" not in input_sequence._metadata or \
       (esmfolded_sequence != input_sequence._metadata["esmfolded_sequence"]) or \
       "avg_plddt" not in input_sequence._metadata or \
       "ptm" not in input_sequence._metadata or \
       "pdb_output" not in input_sequence._metadata:
        folding_output = esmfold_protein_sequence(
            esmfolded_sequence,
            **esmfold_kwargs,
        )
        input_sequence._metadata.update(folding_output)
        input_sequence._metadata["esmfolded_sequence"] = esmfolded_sequence


def esmfold_plddt_constraint(
    inputs: List[ProgramSequence],
    config: Dict[str, Any],
) -> float:
    """
    Scores inputs based on the ESMFold pLDDT.

    Args:
        inputs (List[ProgramSequence]): A list with a single input sequence.
        config (Dict[str, Any]): Configuration parameters:
            - esmfold_kwargs (Dict[str, Any]): Arguments to pass to ESMFold.
            - n_replications (int): The number of times to replicate the sequence
                                    (e.g., for symmetric multimer design).

    Returns:
        float: 1 - ESMFold pLDDT.
    """
    input_sequence = inputs[0]
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["avg_plddt"]


def esmfold_ptm_constraint(
    inputs: List[ProgramSequence],
    config: Dict[str, Any],
) -> float:
    """
    Scores inputs based on the ESMFold pTM.

    Args:
        inputs (List[ProgramSequence]): A list with a single input sequence.
        config (Dict[str, Any]): Configuration parameters:
            - esmfold_kwargs (Dict[str, Any]): Arguments to pass to ESMFold.
            - n_replications (int): The number of times to replicate the sequence
                                    (e.g., for symmetric multimer design).

    Returns:
        float: 1 - ESMFold pTM.
    """
    input_sequence = inputs[0]
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))
    return 1.0 - input_sequence._metadata["ptm"]


def protein_symmetry_ring_constraint(
    inputs: List[ProgramSequence],
    config: Dict[str, Any],
) -> float:
    """
    Constrains a protein to form a symmetric ring.

    Args:
        inputs (List[ProgramSequence]): A list with a single input sequence.
        config (Dict[str, Any]): Configuration parameters:
            - esmfold_kwargs (Dict[str, Any]): Arguments to pass to ESMFold.
            - n_replications (int): The number of times to replicate the sequence
                                    (e.g., for symmetric multimer design).
            - all_to_all_protomer_symmetry (bool): Whether to compare all centroids (True)
                                                   or only adjacent centroids (False, default).

    Returns:
        float: The variance in distances of the centroids of adjacent chains.

    Raises:
        ValueError: If the provided protein sequence is not a multimer.
    """
    from biotite.structure import get_chains
    from .utils import (
        adjacent_distances,
        get_backbone_atoms,
        get_centroid,
        pairwise_distances,
        pdb_file_to_atomarray,
    )

    input_sequence = inputs[0]
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    centroids = []
    for chain_id in get_chains(atom_array):
        chain_backbone = get_backbone_atoms(
            atom_array[atom_array.chain_id == chain_id]
        ).coord
        centroids.append(get_centroid(chain_backbone))
    assert len(centroids) == n_replications

    centroids = np.vstack(centroids)

    return (
        float(np.std(pairwise_distances(centroids)))
        if config.get("all_to_all_protomer_symmetry", False) else
        float(np.std(adjacent_distances(centroids)))
    )


def protein_globularity_constraint(
    inputs: List[ProgramSequence],
    config: Dict[str, Any],
) -> float:
    """
    Encourages globular proteins.

    Args:
        inputs (List[ProgramSequence]): A list with a single input sequence.
        config (Dict[str, Any]): Configuration parameters:
            - esmfold_kwargs (Dict[str, Any]): Arguments to pass to ESMFold.
            - n_replications (int): The number of times to replicate the sequence
                                    (e.g., for symmetric multimer design).

    Returns:
        float: The variance in distances from all backbone atoms to the centroid.
    """
    from .utils import (
        distances_to_centroid,
        get_backbone_atoms,
        pdb_file_to_atomarray,
    )

    input_sequence = inputs[0]
    n_replications = config.get('n_replications', 1)
    _run_esmfold(input_sequence, n_replications, config.get("esmfold_kwargs", {}))

    atom_array = pdb_file_to_atomarray(StringIO(input_sequence._metadata["pdb_output"]))

    backbone = get_backbone_atoms(atom_array).coord

    return float(np.std(distances_to_centroid(backbone)))
