from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd
from language.sequence import ProgramSequence, ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence
from language.constraint import ProgramConstraint


def calculate_energy_score(
    program: Dict[ProgramSequence, List[ProgramConstraint]],
    weights: Dict[ProgramConstraint, float] = {}
) -> Tuple[float, Dict[ProgramSequence, Dict[str, Any]]]:
    """
    Calculate the energy score for a program with multiple sequences and constraints.
    
    Args:
        program: Dictionary mapping sequences to lists of constraints
        weights: Dictionary mapping constraints to their weights (default is 1.0)
        
    Returns:
        Tuple of (total_energy_score, results_dict) where results_dict maps sequences to their metadata
    """
    
    total_energy = 0.0
    results = {}
    
    # Process each sequence and its constraints
    for sequence, constraints in program.items():
        sequence_energy = 0.0
        
        # Apply each constraint to the sequence
        for constraint in constraints:
            weight = weights.get(constraint, 1.0)
            constraint_score = constraint(sequence)
            sequence_energy += weight * constraint_score
            
            # Store constraint details in sequence metadata
            constraint_name = type(constraint).__name__
            sequence._metadata[f"{constraint_name}_score"] = constraint_score
            sequence._metadata[f"{constraint_name}_weight"] = weight

        sequence._metadata["energy_score"] = sequence_energy
        total_energy += sequence_energy
        results[sequence] = deepcopy(sequence._metadata)
    
    return total_energy, results