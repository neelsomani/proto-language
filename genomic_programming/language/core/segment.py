"""
Segment class for the proto-language.

Represents building blocks for biological constructs.
"""
from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Set, Union
import copy
from .sequence import Sequence, SequenceType


class Segment:
    """
    Building block for biological constructs with two sequence pools: candidate (work space) and selected (results space):
    - candidate_sequences: Working space for optimizer proposals (mutations, offspring, rollouts)
    - selected_sequences: Results space containing current best sequences (user-facing)

    Examples:
        Creating a Segment:
        >>> promoter = Segment(sequence="TATA", sequence_type=SequenceType.DNA, label="promoter")
        >>> promoter.label  # "promoter"
        >>> promoter.sequence_length  # 4 (inferred from sequence)
        >>> promoter.selected_sequences  # [Sequence("TATA")]
    """

    def __init__(
        self,
        sequence: Optional[str] = None,
        sequence_length: Optional[int] = None,
        sequence_type: Optional[Union[SequenceType, str]] = SequenceType.DNA,
        valid_chars: Optional[Set[str]] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        constant: bool = False,
    ) -> None:
        """
        Initialize a Segment with dual sequence pools.

        Args:
            sequence: Biological sequence string. If provided, length is inferred unless sequence_length is also provided.
            sequence_length: Target length for sequences. Computed from sequence if not provided
            sequence_type: Type of biological sequence (DNA, RNA, or PROTEIN). Defaults to DNA.
            valid_chars: Optional custom set of valid characters for sequence validation.
            label: Optional label for this segment (e.g., "promoter", "coding_region").
            metadata: Additional data associated with this sequence.
            constant: If True, the sequence is constant and cannot be mutated.
        """
        # Validate and determine sequence_length
        self.sequence_length = self._validate_segment(sequence, sequence_length)
        
        seq = Sequence(
            sequence=sequence if sequence is not None else "", # Use empty string if sequence is None
            sequence_type=sequence_type,
            metadata=metadata,
            valid_chars=valid_chars,
        )
        self.original_sequence = seq
        # Dual pools: candidates (work space) and selected (results space)
        self.candidate_sequences: List[Sequence] = [seq]
        self.selected_sequences: List[Sequence] = [seq]

        self.sequence_type: SequenceType = SequenceType(seq.sequence_type)
        self._valid_chars: Optional[Set[str]] = seq._valid_chars
        self.label: Optional[str] = label

        if constant and not self.original_sequence.sequence:
            raise ValueError("Constant segment must be initialized with a non-empty sequence.")

        # Constant segment is assigned by default
        self.constant = constant
        self._is_assigned: bool = True if constant else False

    @property
    def num_selected(self) -> int:
        """Number of sequences in selected pool (solution space)."""
        return len(self.selected_sequences)

    @property
    def num_candidates(self) -> int:
        """Number of sequences in candidate pool (proposal space)."""
        return len(self.candidate_sequences)

    def __iter__(self) -> Iterator[Sequence]:
        """Iterate over selected sequences (user-facing results)."""
        return iter(self.selected_sequences)

    def __getitem__(self, index: int) -> Sequence:
        """Index into selected sequences (user-facing results)."""
        return self.selected_sequences[index]

    def create_candidates(self, num_candidates: int) -> None:
        """Create a new candidate pool of the given size."""
        self.candidate_sequences = [copy.deepcopy(self.original_sequence) for _ in range(num_candidates)]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Segment to dictionary for cloud/API communication."""
        return {
            "original_sequence": self.original_sequence.to_dict(),
            "sequence_length": self.sequence_length,
            "candidate_sequences": [seq.to_dict() for seq in self.candidate_sequences],
            "selected_sequences": [seq.to_dict() for seq in self.selected_sequences],
            "sequence_type": self.sequence_type.value,
            "valid_chars": list(self._valid_chars) if self._valid_chars else None,
            "label": self.label,
            "constant": self.constant,
            "_is_assigned": self._is_assigned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        """Deserialize Segment from dictionary."""
        # Reconstruct original sequence
        original_seq = Sequence.from_dict(data["original_sequence"])

        # Create segment with original sequence data
        valid_chars = set(data["valid_chars"]) if data.get("valid_chars") else None
        segment = cls(
            sequence=original_seq.sequence,
            sequence_length=data["sequence_length"],
            sequence_type=data["sequence_type"],
            valid_chars=valid_chars,
            label=data.get("label"),
            metadata=original_seq._metadata,
            constant=data.get("constant", False),
        )

        # Restore sequence pools
        segment.original_sequence = original_seq
        segment.candidate_sequences = [Sequence.from_dict(seq_data) for seq_data in data["candidate_sequences"]]
        segment.selected_sequences = [Sequence.from_dict(seq_data) for seq_data in data["selected_sequences"]]
        segment._is_assigned = data.get("_is_assigned", False)

        return segment

    @staticmethod
    def _validate_segment(sequence: Optional[str], sequence_length: Optional[int]) -> int:
        """Validate and infer sequence_length. At least one parameter required; both must match if provided."""
        # Either sequence or sequence_length must be provided
        if sequence is None and sequence_length is None:
            raise ValueError("Either 'sequence' or 'sequence_length' must be provided")
        
        # Ensure sequence_length matches length of sequence if provided
        if sequence is not None:
            # Sequence_length is not provided, set sequence_length to len(sequence)
            if sequence_length is None:
                sequence_length = len(sequence)
            # Both sequence and sequence_length provided but don't match
            elif len(sequence) != sequence_length:
                raise ValueError(f"Provided sequence length ({len(sequence)}) must match sequence_length ({sequence_length})")
        
        # Validate sequence_length
        if sequence_length < 0:
            raise ValueError(f"sequence_length must be 0 or positive, got {sequence_length}")
        
        return sequence_length
