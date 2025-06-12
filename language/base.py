"""
Defines the base classes for the high-level programming language framework.

This module provides the core abstractions for sequence programming:
- ProgramSequence: Individual sequence variables with validation and metadata
- BatchedProgramSequence: Collections of sequences for batch processing  
- ProgramConstraint: Scoring functions that evaluate sequence quality
- ProgramGenerator: Base class for sequence generation algorithms
- ProgramIterativeGenerator: Specialized generator for iterative optimization

These classes work together to enable flexible, constraint-driven sequence design
using various generation strategies like MCMC, autoregressive models, etc.
"""
from abc import ABC, abstractmethod
from typing import (
    Callable, List, Tuple, Dict, Any, Set, Optional, Iterator, Iterable
)
from enum import Enum
from itertools import zip_longest
import numpy as np

class SequenceType(Enum):
    """Enumeration of supported biological sequence types."""
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"

class ConstraintType(Enum):
    """Enumeration of constraint evaluation strategies for multiple inputs."""
    CONTIGUOUS = "contiguous"  # Concatenate sequences before evaluation
    DISJOINT = "disjoint"      # Evaluate sequences separately as a group

class ProgramSequence:
    """
    A biological sequence variable with type validation and metadata tracking.
    
    This is the fundamental unit for sequence programming, representing a single
    DNA, RNA, or protein sequence with automatic validation and rich metadata
    support. Sequences can be empty initially and filled by generators.
    
    The class enforces sequence type constraints and maintains metadata that
    gets automatically updated when the sequence changes.
    
    Examples:
        Creating a DNA sequence:
        >>> seq = ProgramSequence("ATCG", SequenceType.DNA)
        >>> print(len(seq))  # 4
        >>> print(str(seq))  # "ATCG"
        
        Creating an empty sequence to be filled later:
        >>> seq = ProgramSequence(sequence_type=SequenceType.PROTEIN)
        >>> seq.sequence = "MVLSPADKTNVK"
    """
    
    def __init__(
        self,
        sequence: Optional[str] = None,
        sequence_type: Optional[SequenceType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_chars: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize a ProgramSequence with optional sequence data and metadata.

        Args:
            sequence: The biological sequence string. Can be None for empty sequences.
            sequence_type: Type of biological sequence (SequenceType.DNA, SequenceType.RNA, or SequenceType.PROTEIN).
                          Required for validation if sequence is provided.
            metadata: Additional data to associate with this sequence. Will be 
                     automatically updated to track the current sequence value.
            valid_chars: Optional custom set of valid characters for sequence validation.
                        If provided, overrides the default character set for the sequence_type.

        Raises:
            ValueError: If sequence_type is not one of the valid SequenceType values.
        """
        if sequence_type and sequence_type not in [SequenceType.DNA, SequenceType.RNA, SequenceType.PROTEIN]:
                raise ValueError(f"sequence_type must be one of {[SequenceType.DNA, SequenceType.RNA, SequenceType.PROTEIN]}, got {sequence_type}")

        self.sequence_type: Optional[SequenceType] = sequence_type
            
        # Set up character validation based on sequence type or custom valid_chars
        if valid_chars is not None:
            self._valid_chars: Optional[Set[str]] = valid_chars
        elif self.sequence_type == SequenceType.DNA:
            self._valid_chars = set('ACGT- ')
        elif self.sequence_type == SequenceType.RNA:
            self._valid_chars = set('ACGU- ')
        elif self.sequence_type == SequenceType.PROTEIN:
            self._valid_chars = set('ACDEFGHIKLMNPQRSTVWY*-: ')
        else:
            self._valid_chars = None

        # Truncate sequence at the first space character (EOS/space token) if present
        if sequence is not None:
            space_index = sequence.find(' ')
            if space_index != -1:
                sequence = sequence[:space_index]

        self._validate_sequence(sequence)
        self._sequence: Optional[str] = sequence

        if metadata is not None:
            self._metadata: Dict[str, Any] = metadata.copy()
        else:
            self._metadata: Dict[str, Any] = {}
        self._metadata['sequence'] = sequence
        self._metadata['sequence_length'] = len(sequence) if sequence is not None else 0

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate that sequence contains only allowed characters for its type.

        Args:
            sequence: The sequence string to validate.

        Raises:
            ValueError: If sequence contains invalid characters for this sequence type.
        """
        if self._valid_chars is None or sequence is None:
            return

        invalid_chars = set(sequence) - self._valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}. "
                             f"Valid characters are: {', '.join(sorted(self._valid_chars))}")

    @property
    def sequence(self) -> Optional[str]:
        """
        Get the current sequence string.

        Returns:
            The sequence string, or None if no sequence has been set.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, new_sequence: str) -> None:
        """
        Set a new sequence string with automatic validation and metadata updates.
        
        Automatically truncates the sequence at the first space character (" ") if present,
        removing everything from the first space onwards. This handles EOS/space tokens
        that may be generated by language models.

        Args:
            new_sequence: The new sequence string to set.

        Raises:
            ValueError: If the new sequence contains invalid characters.
        """
        if new_sequence is not None:
            # Truncate at the first space character (EOS/space token)
            space_index = new_sequence.find(' ')
            if space_index != -1:
                new_sequence = new_sequence[:space_index]
        
        self._validate_sequence(new_sequence)
        self._sequence = new_sequence
        # Always keep metadata in sync with the actual sequence
        self._metadata["sequence"] = new_sequence
        self._metadata["sequence_length"] = len(new_sequence) if new_sequence is not None else 0

    def __len__(self) -> int:
        """
        Get the length of the sequence.

        Returns:
            Number of characters in the sequence, or 0 if sequence is None.
        """
        if self._sequence is None:
            return 0
        return len(self._sequence)

    def __str__(self) -> str:
        """
        Get the sequence as a string.

        Returns:
            The sequence string, or empty string if sequence is None.
        """
        if self._sequence is None:
            return ""
        return self._sequence
    

class BatchedProgramSequence:
    """
    A collection of ProgramSequence objects representing multiple sequence variants.
    
    This is the primary data structure that generators work with internally.
    Each generator produces one or more BatchedProgramSequence objects, where
    each contains multiple sequence samples/variants that can be evaluated
    and compared during optimization.
    
    Examples:
        Creating a batch of DNA sequences:
        >>> sequences = [
        ...     ProgramSequence(seq, SequenceType.DNA) 
        ...     for seq in ["ATCG", "GCTA", "TTAA"]
        ... ]
        >>> batch = BatchedProgramSequence(sequences)
        >>> print(len(batch))  # 3
        >>> print(batch[0].sequence)  # "ATCG"
    """
    
    def __init__(self, sequences: Iterable[ProgramSequence]) -> None:
        """
        Initialize a batch with a collection of ProgramSequence objects.

        Args:
            sequences: An iterable of ProgramSequence objects. All sequences must
                      have the same sequence_type and valid_chars for consistency.
                      
        Raises:
            ValueError: If sequences have inconsistent sequence_type or valid_chars values.
        """
        self.sequences: Tuple[ProgramSequence] = tuple(sequences)
        self.sequence_type: Optional[SequenceType] = None
        self.valid_chars: Optional[Set[str]] = None
        
        if self.sequences:
            self.sequence_type: SequenceType = self.sequences[0].sequence_type
            self.valid_chars: Optional[Set[str]] = self.sequences[0]._valid_chars
            self._validate_sequences()

    def _validate_sequences(self) -> None:
        """
        Validate that all sequences in the batch have consistent properties.
        
        Raises:
            ValueError: If sequences have inconsistent sequence_type or valid_chars values.
        """
        # Validate consistent sequence_type
        if not all(seq.sequence_type == self.sequence_type for seq in self.sequences):
            all_types = {seq.sequence_type for seq in self.sequences}
            raise ValueError(f"All sequences in a batch must have the same sequence_type. Found: {all_types}")
        
        # Validate consistent valid_chars
        if not all(seq._valid_chars == self.valid_chars for seq in self.sequences):
            raise ValueError(f"All sequences in a batch must have the same valid_chars.")
    
    def __len__(self) -> int:
        """
        Get the number of sequences in this batch.

        Returns:
            Number of ProgramSequence objects in the batch.
        """
        return len(self.sequences)
    
    def __getitem__(self, index: int) -> ProgramSequence:
        """
        Get a specific sequence from the batch by index.

        Args:
            index: Zero-based index of the sequence to retrieve.

        Returns:
            The ProgramSequence at the specified index.
        """
        return self.sequences[index]
    
    def __iter__(self) -> Iterator[ProgramSequence]:
        """
        Iterate over all sequences in the batch.

        Returns:
            Iterator over ProgramSequence objects in this batch.
        """
        return iter(self.sequences)
    

class ProgramConstraint:
    """
    A constraint function that evaluates sequence quality.
    
    Constraints define the objective function for sequence optimization by
    scoring how well sequences satisfy biological or design requirements.
    They can operate on single sequences or multiple sequences simultaneously,
    with different combination strategies.
    
    The constraint evaluates BatchedProgramSequence inputs and returns scores
    where lower values indicate better satisfaction of the constraint.
    
    Examples:
        Creating a simple length constraint:
        >>> def length_constraint(seq, config):
        ...     target = config['target_length']
        ...     return abs(len(seq) - target) / target
        >>> 
        >>> constraint = ProgramConstraint(
        ...     inputs=(sequence_batch,),
        ...     scoring_function=length_constraint,
        ...     scoring_function_config={'target_length': 100}
        ... )
        
        Creating a constraint with a unique name to avoid metadata collisions:
        >>> gc_constraint1 = ProgramConstraint(
        ...     inputs=(sequence_batch1,),
        ...     name="gc_constraint1",
        ...     scoring_function=gc_content_constraint,
        ...     scoring_function_config={'min_gc': 40, 'max_gc': 60}
        ... )
    """
    
    def __init__(
        self,
        inputs: Tuple[BatchedProgramSequence],
        scoring_function: Callable[[ProgramSequence | Tuple[ProgramSequence], Dict[str, Any]], float],
        scoring_function_config: Dict[str, Any] = {},
        constraint_type: ConstraintType = ConstraintType.CONTIGUOUS,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize a constraint with its inputs and scoring function.

        Args:
            inputs: The BatchedProgramSequence objects this constraint evaluates.
                   These should be outputs from registered generators. Sequences
                   are evaluated by corresponding batch indices (inputs[0][i] 
                   with inputs[1][i], etc.). If batches have different sizes,
                   all sequences that have corresponding elements are evaluated.
            scoring_function: Function that scores sequences. Takes either a single
                            ProgramSequence (CONTIGUOUS) or tuple of ProgramSequences
                            (DISJOINT) plus config dict, returns a float score.
            scoring_function_config: Configuration parameters passed to scoring_function.
            constraint_type: How to process multiple inputs:
                           - CONTIGUOUS: Concatenate sequences before evaluation
                           - DISJOINT: Evaluate sequences separately as a group
            name: Optional unique name for this constraint. If provided, metadata keys
                  will be prefixed with this name to avoid collisions when multiple
                  constraints of the same type are used (e.g., "gc_content1_gc_content").

        Note:
            The scoring_function should return values where lower is better within the range [0.0, 1.0].
        """
        
        self.scoring_function: Callable[[ProgramSequence | Tuple[ProgramSequence], Dict[str, Any]], float] = scoring_function
        self.scoring_function_config: Dict[str, Any] = scoring_function_config
        self.inputs: Tuple[BatchedProgramSequence] = inputs
        self.constraint_type: ConstraintType = constraint_type
        self.name: Optional[str] = name

    def _process_inputs(self, inputs: Tuple[BatchedProgramSequence], constraint_type: ConstraintType) -> List[ProgramSequence] | List[Tuple[ProgramSequence]]:
        """
        Transform batched inputs into the format expected by the scoring function.
        
        Processes sequences by corresponding indices across batches. If batches have
        different sizes, evaluates all positions where at least one batch has a sequence.
        
        Args:
            inputs: Tuple of BatchedProgramSequence objects to process.
            constraint_type: Strategy for combining inputs:
                           - CONTIGUOUS: Concatenate corresponding sequences
                           - DISJOINT: Group corresponding sequences as tuples
            
        Returns:
            For CONTIGUOUS: List of ProgramSequence objects with concatenated sequences.
            For DISJOINT: List of tuples, each containing corresponding sequences.
            
        Raises:
            ValueError: If constraint_type is not recognized.
        """
        if constraint_type == ConstraintType.CONTIGUOUS:
            sequence_type = inputs[0].sequence_type if inputs else None
            valid_chars = inputs[0].valid_chars if inputs else None
            
            result = []
            for group in zip_longest(*inputs, fillvalue=None):
                concatenated_sequence = ''.join(seq.sequence or '' for seq in group if seq is not None)
                #TODO: weird behavior if two sequences have same metadata keys but different values
                merged_metadata = {k: v for seq in group if seq is not None for k, v in seq._metadata.items()}
                
                result.append(ProgramSequence(
                    sequence=concatenated_sequence,
                    sequence_type=sequence_type,
                    metadata=merged_metadata,
                    valid_chars=valid_chars
                ))
            return result
        
        elif constraint_type == ConstraintType.DISJOINT:
            return [tuple(group) for group in zip_longest(*inputs, fillvalue=None)]
        else:
            raise ValueError(f"Invalid constraint type: {constraint_type}")

    def evaluate(self) -> List[float]:
        """
        Evaluate this constraint on all sequences in the input batches.

        The constraint is applied to each corresponding set of sequences across
        all input batches, producing one score per batch element.

        Returns:
            List of constraint scores, one per batch element. Lower scores
            indicate better constraint satisfaction. Returns float('inf')
            for invalid/None sequences.
            
        Note:
            The length of the returned list equals the batch size of the input
            BatchedProgramSequence objects.
        """
        scoring_function_inputs = self._process_inputs(self.inputs, self.constraint_type)
        
        scores = []
        for i, input in enumerate(scoring_function_inputs):
            # Check for invalid input scenarios
            if  ((isinstance(input, tuple) and None in input) or
                (isinstance(input, tuple) and any(isinstance(seq, ProgramSequence) and seq.sequence is None for seq in input))):
                scores.append(float('inf'))
            else:
                # Store original metadata to identify new keys added by the scoring function
                if isinstance(input, ProgramSequence):
                    original_metadata_keys = set(input._metadata.keys())
                
                scores.append(self.scoring_function(input, self.scoring_function_config))
                
                # Copy only newly added metadata back to original sequences
                if isinstance(input, ProgramSequence):
                    for batch in self.inputs:
                        original_seq = batch[i]
                        # Find keys that were added by the scoring function
                        new_keys = set(input._metadata.keys()) - original_metadata_keys
                        for key in new_keys:
                            if key != "sequence":  # Don't overwrite original sequence content
                                value = input._metadata[key]
                                # Prefix metadata key with constraint name if provided
                                prefixed_key = f"{self.name}_{key}" if self.name else key
                                original_seq._metadata[prefixed_key] = value
        
        return scores


class ProgramGenerator(ABC):
    """
    Abstract base class for sequence generation algorithms and samplers.

    Generators are responsible for creating and modifying sequences during
    the optimization process. They can implement various strategies like
    random mutations, autoregressive language models, or other sampling methods.

    Key concepts:
    - _generator_outputs: Internal BatchedProgramSequence objects that algorithms modify
    - Batch processing: Generate multiple sequence variants simultaneously
    - In-place updates: Sequences are modified directly for efficiency

    Subclasses must implement register() to initialize sequences and sample()
    to propose new sequence variants.
    
    Note: 
        - sample() must contain logic to update self._generator_outputs with the new sequences.
          see UniformMutationGenerator.sample() for an example.
          
    Examples:
        Implementing a simple random mutation generator:
        >>> class RandomMutator(ProgramGenerator):
        ...     def register(self):
        ...         sequences = [ProgramSequence("ATCG", SequenceType.DNA) 
        ...                     for _ in range(self.batch_size)]
        ...         self._generator_outputs = (BatchedProgramSequence(sequences),)
        ...         self._is_initialized = True
        ...         return self._generator_outputs
        ...     
        ...     def sample(self):
        ...         # Modify sequences in self._generator_outputs
        ...         pass
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        **hyperparameters: Any,
    ) -> None:
        """
        Initialize the generator with configuration parameters.

        Args:
            batch_size: Number of sequence variants to generate simultaneously.
            **hyperparameters: Additional configuration specific to the generator type.
                              These are stored and can be accessed by subclasses.
        """
        self.batch_size: int = batch_size
        self.hyperparameters: Dict[str, Any] = hyperparameters
        self._is_initialized: bool = False
        self._generator_outputs: Optional[Tuple[BatchedProgramSequence]] = None

    @abstractmethod
    def register(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[BatchedProgramSequence]:
        """
        Create and initialize the _generator_outputs.

        This method creates the internal BatchedProgramSequence objects that
        the generator will modify during sampling. These _generator_outputs
        are what optimization algorithms operate on directly.

        Args:
            *args: Positional arguments for registration (subclass-specific).
            **kwargs: Keyword arguments for registration (subclass-specific).

        Returns:
            Tuple of BatchedProgramSequence objects that will be modified in-place throughout generation.
            
        Note:
            This method must set self._is_initialized = True and store the
            created sequences in self._generator_outputs.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the register method."
        )

    def __len__(self) -> int:
        """
        Get the number of _generator_outputs managed by this generator.

        Returns:
            Number of BatchedProgramSequence objects this generator manages.

        Raises:
            RuntimeError: If called before register() has been called.
        """
        if not self._is_initialized or self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call register() first."
            )
        return len(self._generator_outputs)

    def get_generator_outputs(self) -> Tuple[BatchedProgramSequence]:
        """
        Access the internal _generator_outputs that algorithms modify in-place.
        
        These are the raw sequence objects that generators manipulate directly.
        For user-facing outputs with metadata and concatenation, use the
        user_sequences property on ProgramIterativeGenerator instead.

        Returns:
            Tuple of BatchedProgramSequence objects that are modified during sampling.

        Raises:
            RuntimeError: If called before register() has been called.
        """
        if not self._is_initialized or self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call register() first."
            )
        return self._generator_outputs

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> None:
        """
        Sample new sequences by modifying _generator_outputs in-place.

        This is where the core generation logic happens. Implementations
        should directly modify the sequences within self._generator_outputs
        rather than creating new objects.

        Args:
            *args: Positional arguments for sampling (subclass-specific).
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Raises:
            RuntimeError: If called before register() has been called.
            
        Note:
            This method should modify sequences in-place for efficiency.
            It does not return anything - the changes are reflected in
            the _generator_outputs accessed via get_generator_outputs().
        """
        if not self._is_initialized or self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call register() first."
            )
        raise NotImplementedError("Subclasses must implement the sample method.")


class ProgramIterativeGenerator(ProgramGenerator):
    """
    Specialized generator for iterative optimization with energy-based evaluation.
    
    This class extends ProgramGenerator to support iterative algorithms like MCMC
    that require energy evaluation and state tracking. It bridges the gap between
    internal algorithm operations and user-facing results by providing:
    
    - Energy scoring via constraint combination
    - User-friendly sequence access with metadata
    - Automatic tracking of optimization progress
    - Flexible sequence concatenation for complex designs
    
    The class works with multiple sub-generators and constraints to implement
    sophisticated optimization strategies while maintaining clean interfaces.
    
    Examples:
        Using in MCMC optimization:
        >>> mcmc = ProgramMCMCGenerator(
        ...     generators=[mutation_gen, crossover_gen],
        ...     constraints=[gc_constraint, length_constraint],
        ...     sequence_order=((batch1, batch2), (batch3,))
        ... )
        >>> history = mcmc.sample()  # Returns optimization history
        >>> final_seqs = mcmc.user_sequences  # Clean user-facing results
    """
    
    def _check_constraint_attributes(self) -> None:
        """
        Validate that required constraint attributes are properly configured.
        
        Raises:
            ValueError: If constraints are missing or improperly configured.
        """
        if not hasattr(self, 'constraints'):
            raise ValueError("ProgramIterativeGenerator objects must have constraints.")

        for constraint in self.constraints:
            if not isinstance(constraint, ProgramConstraint):
                raise ValueError(f"Found type {type(constraint)}, expected a ProgramConstraint")

        if not hasattr(self, 'constraint_weights'):
            self.constraint_weights = [1.] * len(self.constraints)

    def _validate_init(self) -> None:
        """
        Validate that generators, constraints, and sequence order are properly configured.
        
        Raises:
            ValueError: If any validation checks fail.
        """
        # Check that all required attributes exist
        if not hasattr(self, 'generators'):
            raise ValueError("ProgramIterativeGenerator must have 'generators' attribute")
        if not hasattr(self, 'constraints'):
            raise ValueError("ProgramIterativeGenerator must have 'constraints' attribute")
        if not hasattr(self, 'sequence_order'):
            raise ValueError("ProgramIterativeGenerator must have 'sequence_order' attribute")
        if not hasattr(self, 'constraint_weights'):
            raise ValueError("ProgramIterativeGenerator must have 'constraint_weights' attribute")
        
        if len(self.constraints) != len(self.constraint_weights):
            raise ValueError("Constraint weights must match number of constraints.")

        # Generators must already be registered, since their variables are hooked up to constraints.
        variable_ids = set()
        for generator in self.generators:
            if not generator._is_initialized:
                raise ValueError("Not all generators have been registered.")
            generator_outputs = generator.get_generator_outputs()
            for sequence_batch in generator_outputs:
                variable_ids.add(id(sequence_batch))

        # All constraint inputs must be the same as generator outputs.
        for constraint in self.constraints:
            for input_ in constraint.inputs:
                if id(input_) not in variable_ids:
                    raise ValueError(
                        "Found a constraint not tied to a given generator."
                    )

        # Validate that all BatchedProgramSequence objects in sequence_order exist in generator outputs
        all_generator_outputs = {id(seq) for gen in self.generators for seq in gen.get_generator_outputs()}
        all_sequence_order_ids = {id(seq) for group in self.sequence_order for seq in group}
        
        if all_sequence_order_ids != all_generator_outputs:
            raise ValueError("sequence_order must contain exactly the same BatchedProgramSequence objects as generator outputs")

    def score_energy(self, operation: str = 'add') -> List[float]:
        """
        Compute energy scores by combining constraint evaluation scores.

        The energy function is computed as a weighted combination of all
        constraint scores. Lower energy values indicate better solutions.

        Args:
            operation: How to combine constraint scores across constraints:
                      - 'add': Sum weighted constraint scores (default)
                      - 'multiply': Multiply weighted constraint scores

        Returns:
            List of energy values, one per batch element. Lower values indicate
            better constraint satisfaction.
            
        Raises:
            ValueError: If operation is not 'add' or 'multiply'.
            
        Note:
            The returned list length equals the batch size of the sequences.
            Energy computation uses current sequence values, so it reflects
            the most recent state after any sampling operations.
        """
        self._check_constraint_attributes()
        assert len(self.constraints) == len(self.constraint_weights)

        # Get weighted scores from all constraints: shape (n_constraints, n_samples)
        constraint_scores = np.array([
            np.array(constraint.evaluate()) * weight
            for constraint, weight in zip(self.constraints, self.constraint_weights)
        ])

        # Combine across constraints for each sample
        if operation == 'multiply':
            energies = np.prod(constraint_scores, axis=0)
        elif operation == 'add':
            energies = np.sum(constraint_scores, axis=0)
        else:
            raise ValueError(f"Operation must be 'multiply' or 'add', got {operation}")

        return energies.tolist()

    def _propagate_best_sequence(self, best_idx: int) -> None:
        """
        Copy the best sequence to all positions within each BatchedProgramSequence.
        
        This helper method ensures that when a proposal is accepted, the sequence
        with the best energy is propagated to all positions within each batch.
        This is essential for maintaining consistency in user_sequences access.
        
        Args:
            best_idx: Index of the best sequence to propagate across all batches.
            
        Raises:
            ValueError: If any batch has fewer sequences than best_idx.
            
        Note:
            This method modifies sequences in-place for all _generator_outputs.
        """
        for sequence_batch in self._generator_outputs:
            if len(sequence_batch) > best_idx:
                best_sequence = sequence_batch[best_idx].sequence
                for program_seq in sequence_batch:
                    program_seq.sequence = best_sequence
            else:
                raise ValueError(f"Batch has only {len(sequence_batch)} sequences, cannot propagate best sequence at index {best_idx}")

    @property
    @abstractmethod
    def user_sequences(self) -> Tuple[ProgramSequence]:
        """Get user-facing sequences with appropriate metadata and proper concatenation."""
        raise NotImplementedError("Child classes must implement user_sequences property")

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> List[Tuple[ProgramSequence]]:
        """
        Run iterative generation and return optimization history.
        
        Unlike the base ProgramGenerator.sample() which returns None, this method
        returns a history of user_sequences snapshots taken at tracked intervals
        during the optimization process.

        Args:
            *args: Positional arguments for sampling (subclass-specific).
            **kwargs: Keyword arguments for sampling (subclass-specific).

        Returns:
            List of user_sequences snapshots taken at tracked steps. Each element
            is a tuple of ProgramSequence objects with energy and time metadata.

        Raises:
            RuntimeError: If called before register() has been called.
            
        Note:
            The returned history allows tracking optimization progress over time.
            The final state is also accessible via the user_sequences property.
        """
        if not self._is_initialized or self._generator_outputs is None:
            raise RuntimeError(
                f"Generator {self.__class__.__name__} has not been initialized. "
                "Call register() first."
            )
        raise NotImplementedError("Subclasses must implement the sample method.")
