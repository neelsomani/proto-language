from abc import ABC, abstractmethod
from typing import Any, List, Dict
from language.sequence import ProgramSequence, ProgramDNASequence, ProgramRNASequence, ProgramProteinSequence

import pandas as pd
import numpy as np
import re
import itertools

class ProgramConstraint(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def __call__(self, sequences: List[ProgramSequence]) -> List[float]:
        return self.evaluate(sequences)
    
class ValidCharactersConstraint(ProgramConstraint):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []

        for seq in sequences:
            if isinstance(seq, ProgramDNASequence):
                valid_chars = {'A', 'C', 'G', 'T'}
            elif isinstance(seq, ProgramRNASequence):
                valid_chars = {'A', 'U', 'G', 'C'}
            elif isinstance(seq, ProgramProteinSequence):
                valid_chars = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
            else:
                raise ValueError(f"Unknown sequence type: {type(seq)}")
                
            sequence = seq._sequence
            has_invalid = bool(re.search(f'[^{"".join(valid_chars)}]', sequence))
            
            # Add metrics to df
            seq.data['valid_nucleotides'] = not has_invalid
            
            # return 1.0 if sequence only contains valid nucleotides, otherwise 0.0
            if has_invalid:
                results.append(1.0)
            else:
                results.append(0.0)
                
        return results

class GenomeLengthConstraint(ProgramConstraint):
    def __init__(self, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length

    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []
            
        for seq in sequences:
            genome_length = len(seq._sequence)
            
            # Add metrics to df
            seq.data['genome_length'] = genome_length
            
            # Calculate deviation from target length
            if genome_length == self.target_length:
                results.append(0.0)
                continue
            
            # Calculate normalized deviation (similar to other energy terms)
            # Scale between 0.0 and 1.0 based on how far we are from target
            deviation = abs(genome_length - self.target_length) / self.target_length
            
            # Cap at 1.0 for large deviations
            results.append(min(1.0, deviation))
            
        return results

class GCContent(ProgramConstraint):
    def __init__(self, target_range: tuple = (30, 60)) -> None:
        super().__init__()
        self.min_gc = min(target_range)
        self.max_gc = max(target_range)
        
        # Validate range
        if self.min_gc < 0 or self.max_gc > 100:
            raise ValueError("GC content range must be between 0 and 100 percent.")

    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []
            
        for seq in sequences:
            sequence = seq._sequence
            seq_len = len(sequence)
            
            # edge case
            if seq_len == 0:
                seq.data['gc_content'] = 0.0
                results.append(1.0)
                continue
                
            # Count G and C nucleotides directly in one pass
            gc_count = 0
            for nt in sequence:
                if nt in 'GC':
                    gc_count += 1
                    
            # Calculate GC content
            gc_content = (gc_count / seq_len) * 100
            
            # Add metrics to seq.data
            seq.data['gc_content'] = gc_content
            
            # return 0.0 if GC content is within the desired range
            if self.min_gc <= gc_content <= self.max_gc:
                results.append(0.0)
            else:
                # return a normalized score based on distance from acceptable range
                if gc_content < self.min_gc:
                    deviation = (self.min_gc - gc_content) / self.min_gc
                else:
                    deviation = (gc_content - self.max_gc) / (100 - self.max_gc)
                
                # Return a score that approaches 1 as deviation increases -- does this scoring make sense?
                results.append(min(1.0, deviation))
                
        return results

class NucleotideHomopolymer(ProgramConstraint):
    def __init__(self, max_length: int = 10) -> None:
        super().__init__()
        self.max_length = max_length

    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []
        
        for seq in sequences:
            sequence = seq._sequence
            
            # Use a single regex pattern to find all homopolymers of any nucleotide
            # This matches any consecutive repeated nucleotide (A, C, G, or T)
            matches = re.findall(r'(A+|C+|G+|T+)', sequence)
            
            # Find the length of the longest homopolymer
            longest_homopolymer = max((len(match) for match in matches), default=0)
            
            # Add homopolymer info to the sequence data
            seq.data['longest_homopolymer_length'] = longest_homopolymer
            
            # Return 0.0 if the longest homopolymer is within acceptable range
            if longest_homopolymer <= self.max_length:
                results.append(0.0)
            else:
                # Use a logarithmic scale for scoring rather than assuming a fixed extreme value
                # This approach increases more slowly as homopolymer length grows
                # ln(excess/max_allowed)/ln(2) gives a score of 1.0 when excess length = 2*max_allowed
                excess_length = longest_homopolymer - self.max_length
                log_ratio = np.log(1 + excess_length/self.max_length) / np.log(2)
                results.append(min(1.0, log_ratio))
                
        return results

class DinucleotideFrequency(ProgramConstraint):
    def __init__(self, freq_range: tuple = (0.03, 0.08)) -> None:
        super().__init__()
        self.min_freq = min(freq_range)
        self.max_freq = max(freq_range)

        # Precompute all dinucleotides
        self.dinucleotides = [''.join(pair) for pair in itertools.product('ACGT', repeat=2)]
        
    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []
            
        for seq in sequences:
            sequence = seq._sequence
            seq_len = len(sequence)
            
            # Edge case
            if seq_len < 2:
                # Add empty dinucleotide frequencies to sequence data
                seq.data['dinucleotide_freqs'] = {}
                results.append(1.0)
                continue
            
            dinucleotide_counts = {}
            total_count = 0
            for i in range(seq_len - 1):
                dinuc = sequence[i:i+2]
                if all(nt in 'ACGT' for nt in dinuc):  # Only count valid dinucleotides
                    dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
                    total_count += 1
            
            # If no valid dinucleotides found
            if total_count == 0:
                seq.data['dinucleotide_freqs'] = {}
                results.append(1.0)
                continue
                
            # Calculate frequencies and check if they're in range
            max_deviation = 0.0
            dinucleotide_freqs = {}
            
            # Score based on deviation from target dinucleotide frequencies
            for dinuc in self.dinucleotides:
                freq = dinucleotide_counts.get(dinuc, 0) / total_count
                dinucleotide_freqs[dinuc] = freq
                
                # Calculate deviation if outside acceptable range
                if freq < self.min_freq:
                    deviation = (self.min_freq - freq) / self.min_freq
                    max_deviation = max(max_deviation, deviation)
                elif freq > self.max_freq:
                    deviation = (freq - self.max_freq) / (1.0 - self.max_freq)
                    max_deviation = max(max_deviation, deviation)
            
            # Add dinucleotide frequencies to sequence data
            seq.data['dinucleotide_freqs'] = dinucleotide_freqs
            
            # Return max deviation
            results.append(min(1.0, max_deviation))
            
        return results

class TetranucleotideUsage(ProgramConstraint):
    def __init__(self, tetranucleotide: str = "GATC", tud_range: tuple = (0.8, 1.2)) -> None:
        super().__init__()
        self.tetranucleotide = tetranucleotide.upper()
        self.min_tud = min(tud_range)
        self.max_tud = max(tud_range)
        
        # Validate tetranucleotide input
        if len(self.tetranucleotide) != 4:
            raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")
    
    def evaluate(self, sequences: List[ProgramSequence]) -> List[float]:
        results = []
        
        for seq in sequences:
            sequence = seq._sequence
            tetra = self.tetranucleotide
            
            # edge case
            total_bases = len(sequence)
            if total_bases < 4:
                seq.data['tetra_tud'] = 0.0
                results.append(1.0)
                continue
            
            # Calculate frequencies of each nucleotide in the whole sequence
            A_freq = sequence.count("A") / total_bases
            C_freq = sequence.count("C") / total_bases
            G_freq = sequence.count("G") / total_bases
            T_freq = sequence.count("T") / total_bases
            
            # Calculate tetranucleotide occurrences
            tetra_count = sequence.count(tetra)
            
            # Calculate expected frequency using zero-order Markov method
            nucleotide_freqs = {'A': A_freq, 'C': C_freq, 'G': G_freq, 'T': T_freq}
            
            # Build expected frequency product based on nucleotides in the tetranucleotide
            tetra_expected_freq = 1.0
            for nt in tetra:
                if nt in nucleotide_freqs:
                    tetra_expected_freq *= nucleotide_freqs[nt]
                else:
                    # If invalid nucleotide, set to 0
                    tetra_expected_freq = 0
                    break
            tetra_expected_freq *= total_bases
            
            # Calculate TUD
            tetra_tud = tetra_count / tetra_expected_freq if tetra_expected_freq != 0 else 0
            
            # Add tetranucleotide usage data to sequence data
            seq.data['tetra_tud'] = tetra_tud
            
            # Score based on TUD range
            if self.min_tud <= tetra_tud <= self.max_tud:
                results.append(0.0)
            else:
                # Calculate normalized deviation
                if tetra_tud < self.min_tud:
                    deviation = (self.min_tud - tetra_tud) / self.min_tud
                else:
                    deviation = (tetra_tud - self.max_tud) / self.max_tud
                
                results.append(min(1.0, deviation))
                
        return results