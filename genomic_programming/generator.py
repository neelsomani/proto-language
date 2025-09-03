import copy
import os
import random
import sys
import time
import json
import itertools
import numpy as np
import requests
import torch
import heapq
from typing import Any, List, Optional, Tuple, Callable, Iterable, Dict, final, Generator as GeneratorType
from .base import *

# Maximum safe exponent for np.exp() to prevent overflow
MAX_EXP_ARG = 700.0

@final
class UniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations.

    This generator initializes with a random sequence and samples single-nucleotide
    or amino acid mutations on each call to sample().

    Examples:
        Creating a DNA mutation generator:
        >>> segment = ConstructSegment(sequence="ATCGG", sequence_type=SequenceType.DNA)
        >>> gen = UniformMutationGenerator(
        ...     batch_size=5,
        ...     sequence_length=5,
        ...     num_mutations=2
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Introduces 2 random mutations
        >>> outputs = gen.get_generator_outputs()
        >>> len(outputs[0])  # 5 (batch size)

        Note: Using a mutation scheduler. Define a function that takes the iteration count 
            and returns the number of mutations.
        
        >>> def mutation_scheduler(iteration: int) -> int:
        ...     return max(1, 10 - iteration // 10)  # Decrease mutations over time
        >>> gen = UniformMutationGenerator(
        ...     batch_size=5,
        ...     sequence_length=100,
        ...     mutation_scheduler=mutation_scheduler
        ... )
    """

    def __init__(
        self,
        batch_size: int = 1,
        sequence_length: int = 100,
        num_mutations: int = 1,
        mutation_scheduler: Optional[Callable[[int], int]] = None,
    ) -> None:
        """
        Initialize the uniform mutation generator.

        Args:
            batch_size: Number of sequence variants to maintain simultaneously.
            sequence_length: Length of the sequence to generate.
            num_mutations: Number of mutations to introduce per sequence per sample() call.
                          Ignored if mutation_scheduler is provided.
            mutation_scheduler: Optional callable that takes iteration count and returns
                              number of mutations. If provided, overrides num_mutations.
        """
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.num_mutations = num_mutations
        self.mutation_scheduler = mutation_scheduler

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "UniformMutationGenerator must be assigned exactly one ConstructSegment"
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True

        initial_sequence = self._generator_output.batch_sequences[0].sequence
        valid_chars = self._generator_output._valid_chars - set(" ")
        valid_chars_list = list(valid_chars)
        if initial_sequence == "":
            self._generator_output.batch_sequences[0].sequence = "".join(
                random.choice(valid_chars_list) for _ in range(self.sequence_length)
            )
        else:
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )
        self._generator_output.create_batch(self.batch_size)

        # No model initialization needed for this generator
        self._is_initialized = True

    def sample(self) -> None:
        """
        Introduce random point mutations in each sequence.

        For each sequence in the batch, selects random positions and replaces
        the characters with different random characters from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Determine number of mutations for this iteration
        if self.mutation_scheduler is not None:
            current_mutations = self.mutation_scheduler(self.iteration_count)
        else:
            current_mutations = self.num_mutations

        # Sample mutations for each output in the segment batch
        for sequence in self._generator_output.batch_sequences:
            current_sequence = sequence.sequence
            sequence_length = len(current_sequence)
            
            # Ensure we don't try to mutate more positions than available
            actual_mutations = min(current_mutations, sequence_length)
            
            # Select random positions to mutate (without replacement)
            positions_to_mutate = random.sample(range(sequence_length), actual_mutations)
            
            # Apply mutations
            for pos in positions_to_mutate:
                current_char = current_sequence[pos]
                
                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in self._generator_output._valid_chars if c != current_char
                ]
                mutated_char = random.choice(possible_mutations)
                current_sequence = (
                    current_sequence[:pos]
                    + mutated_char
                    + current_sequence[pos + 1 :]
                )
            
            sequence.sequence = current_sequence

        # Increment iteration count (shared helper on base class)
        self._increment_iteration_count()


@final
class TwoSegmentUniformMutationGenerator(Generator):
    """
    A sequence generator that proposes random point mutations across two segments.

    This generator is specifically designed to work with exactly two ConstructSegment objects,
    randomly mutating each segment independently. This is a common pattern in bio models that model
    paired sequences (e.g., protein-ligand, protein-protein, or DNA-RNA pairs). The segments can have different lengths.

    Examples:
        Creating a two-segment mutation generator:
        >>> segment1 = ConstructSegment(sequence="ATCGG", sequence_type=SequenceType.DNA)
        >>> segment2 = ConstructSegment(sequence="GCTAA", sequence_type=SequenceType.DNA)
        >>> gen = TwoSegmentUniformMutationGenerator(batch_size=5)
        >>> gen.assign([segment1, segment2])
        >>> gen.sample()  # Introduces random mutations in each segment
        >>> outputs = gen.get_generator_outputs()
        >>> len(outputs)  # 2 (number of segments)
    """

    def __init__(
        self,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the two-segment mutation generator.

        Args:
            batch_size: Number of sequence variants to maintain simultaneously.
        """
        super().__init__(batch_size=batch_size)

    def assign(
        self, assigned_segments: Iterable[ConstructSegment]
    ) -> None:
        """
        Assign exactly two ConstructSegment objects to this generator.

        Args:
            assigned_segments: An iterable of exactly two ConstructSegment objects.

        Raises:
            ValueError: If not exactly two segments are provided.
        """
        segments = list(assigned_segments)
        
        if len(segments) != 2:
            raise ValueError(f"TwoSegmentUniformMutationGenerator requires exactly 2 segments, got {len(segments)}")

        # Segments can have different lengths (common in bio models like BindCraft and RF diffusion)
        # Initialize _generator_outputs for two segments
        self._generator_outputs = tuple(segments)
        for segment in self._generator_outputs:
            segment._is_assigned = True
            
            # Validate that existing sequences are not empty
            initial_sequence = segment.batch_sequences[0].sequence
            if initial_sequence == "":
                raise ValueError("TwoSegmentUniformMutationGenerator requires segments with existing sequences (cannot be empty)")
            
            segment.create_batch(self.batch_size)

        # No model initialization needed for this generator
        self._is_initialized = True

    def sample(self) -> None:
        """
        Introduce a random point mutation in each sequence of each segment.

        For each sequence in each segment's batch, selects a random position and replaces
        the character with a different random character from the vocabulary.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Sample mutation for each segment
        for segment in self._generator_outputs:
            for sequence in segment.batch_sequences:
                if len(sequence.sequence) == 0:
                    continue  # Skip empty sequences
                    
                mutated_index = random.randint(0, len(sequence.sequence) - 1)
                current_sequence = sequence.sequence
                current_char = current_sequence[mutated_index]

                # Make sure the mutated character is different from the current one
                possible_mutations = [
                    c for c in segment._valid_chars if c != current_char
                ]
                if possible_mutations:
                    mutated_char = random.choice(possible_mutations)
                    sequence.sequence = (
                        current_sequence[:mutated_index]
                        + mutated_char
                        + current_sequence[mutated_index + 1 :]
                    )


@final
class Evo2Generator(Generator):
    """
    A sequence generator that uses the Evo2 genome language model for DNA sequence generation.

    This generator wraps the Evo2 model to provide autoregressive sequence generation
    from prompt sequences. The generator can handle single prompts (replicated across batch)
    or multiple prompts (one per batch element), with automatic model instance sharing
    between generators that use the same model configuration.

    Examples:
        Basic DNA generation:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     evo2_type="evo2_7b",
        ...     sequence_length=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences from prompts

        Custom model with local weights:
        >>> gen = Evo2Generator(
        ...     prompt_seqs=["+~GA", "+~GC"],
        ...     evo2_type="evo2_7b_phage",
        ...     evo2_local_path="/path/to/weights.pt",
        ...     batch_size=2
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()  # Uses local model weights
    """

    def __init__(
        self,
        prompt_seqs: List[str],
        evo2_type: str = "evo2_7b",
        evo2_local_path: Optional[str] = None,
        sequence_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: Optional[int] = None,
        batch_size: int = 1,
        prepend_prompt: bool = False,
        **sampling_kwargs: Any,
    ) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            prompt_seqs: List of prompt sequences to start generation from.
                Single prompt gets replicated batch_size times, or provide
                one prompt per batch element.
            evo2_type: Name of the Evo2 model variant to use.
            evo2_local_path: Optional path to local model weights file.
            sequence_length: Number of tokens to generate after each prompt.
            temperature: Sampling temperature for nucleus sampling.
            top_k: Top-k parameter for sampling.
            top_p: Top-p (nucleus) parameter for sampling.
            batched: Whether to use batched generation for efficiency.
            cached_generation: Whether to cache model states for faster sampling.
            verbose: Verbosity level for generation logging.
            force_prompt_threshold: Optional threshold for forcing prompt continuation.
            batch_size: Number of sequences to generate simultaneously.
            prepend_prompt: Whether to prepend the prompt to generated sequences.
            **sampling_kwargs: Additional arguments passed to Evo2 model sampling.

        Note:
            Model instances are automatically shared between generators with the same
            evo2_type, evo2_local_path, and sampling_kwargs to save memory and initialization time.
        """
        super().__init__(batch_size=batch_size)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert (
                len(prompt_seqs) == batch_size
            ), f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
            assert (
                len(set(len(seq) for seq in prompt_seqs)) == 1
            ), f"All prompts must have same length, got: {[len(seq) for seq in prompt_seqs]}"
            self.prompt_seqs = prompt_seqs

        self.batch_size = batch_size
        self.evo2_type = evo2_type
        self.evo2_local_path = evo2_local_path
        self.n_tokens = sequence_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batched = batched
        self.cached_generation = cached_generation
        self.verbose = verbose
        self.force_prompt_threshold = force_prompt_threshold
        self.prepend_prompt = prepend_prompt
        self.sampling_kwargs = sampling_kwargs

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as Evo2 performs autoregressive generation from prompt sequences.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "Evo2Generator must be assigned exactly one ConstructSegment"
            )

        # Warn user if existing sequences will be overwritten
        existing_sequences = [
            seq.sequence for seq in assigned_segments.batch_sequences if seq.sequence
        ]
        if existing_sequences:
            print(
                f"Warning: Evo2Generator will overwrite {len(existing_sequences)} existing sequence(s) "
                f"when sample() is called due to autoregressive generation."
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True

    def sample(self, prompt_seqs: Optional[List[str]] = None) -> None:
        """
        Generate sequences using the Evo2 model and update generator output.

        Uses the Evo2 model to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the ConstructSegment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        # Choose execution mode based on configuration
        from .utils import use_cloud_gpu
        
        if use_cloud_gpu():
            # Use cloud for cloud GPU execution
            print("Using cloud for Evo2 generation...")
            import cloud
            evo2_generate_cloud = cloud.Function.from_name('proto-language', 'evo2_generate_cloud')
            generated_sequences = evo2_generate_cloud.remote(
                prompt_seqs=prompts,
                evo2_type=self.evo2_type,
                evo2_local_path=self.evo2_local_path,
                n_tokens=self.n_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                batched=self.batched,
                cached_generation=self.cached_generation,
                verbose=self.verbose,
                force_prompt_threshold=self.force_prompt_threshold,
                **self.sampling_kwargs,
            )
        else:
            # Use local GPU execution
            print("Using local GPU for Evo2 generation...")
            generated_sequences = self._evo2_generate_gpu(
                prompt_seqs=prompts,
                evo2_type=self.evo2_type,
                evo2_local_path=self.evo2_local_path,
                n_tokens=self.n_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                batched=self.batched,
                cached_generation=self.cached_generation,
                verbose=self.verbose,
                force_prompt_threshold=self.force_prompt_threshold,
                **self.sampling_kwargs,
            )

        # Update sequences in the ConstructSegment
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence

    def _evo2_generate_gpu(
        self,
        prompt_seqs: List[str],
        evo2_type: str,
        evo2_local_path: Optional[str],
        n_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        batched: bool,
        cached_generation: bool,
        verbose: int,
        force_prompt_threshold: Optional[int],
        **sampling_kwargs
    ) -> List[str]:
        """
        Local GPU function for Evo2 generation.
        
        Returns:
            List of generated sequences
        """
        from evo2 import Evo2

        # Load and generate
        print(f"Loading Evo2 model: {evo2_type}")
        evo2_model = Evo2(model_name=evo2_type, local_path=evo2_local_path)
        
        output = evo2_model.generate(
            prompt_seqs=prompt_seqs,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batched=batched,
            cached_generation=cached_generation,
            verbose=verbose,
            force_prompt_threshold=force_prompt_threshold,
            **sampling_kwargs,
        )
        return output.sequences


@final
class NimEvo2Generator(Generator):
    """
    A sequence generator that uses the Nvidia NIM Evo2 API for DNA sequence generation.

    Users must provide a NVIDIA API key for authentication or set it in the NV_API_KEY environment variable.

    Examples:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen = NimEvo2Generator(
        ...     prompt_seqs=["+~GA"],
        ...     api_key="your_api_key",
        ...     n_tokens=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen.assign(segment)
        >>> gen.sample()
    """

    def __init__(
        self,
        prompt_seqs: List[str],
        nim_api_url: str = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate",
        api_key: Optional[str] = None,
        sequence_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        enable_sampled_probs: bool = False,
        verbose: int = 1,
        batch_size: int = 1,
        prepend_prompt: bool = False,
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize the NIM Evo2 generator with API configuration and sampling parameters.

        Args:
            prompt_seqs: List of prompt sequences to start generation from.
                Single prompt gets replicated batch_size times, or provide
                one prompt per batch element.
            nim_api_url: Full URL for the Nvidia NIM API endpoint.
            api_key: API key for authentication. If None, will try to get from NV_API_KEY environment variable.
            n_tokens: Number of tokens to generate after each prompt.
            temperature: Sampling temperature for nucleus sampling.
            top_k: Top-k parameter for sampling.
            top_p: Top-p (nucleus) parameter for sampling.
            enable_sampled_probs: Whether to enable sampled probabilities in API response.
            verbose: Verbosity level for generation logging.
            batch_size: Number of sequences to generate simultaneously.
            prepend_prompt: Whether to prepend the prompt to generated sequences.
            timeout: Request timeout in seconds.

        Note:
            The API key can be provided directly or set in the NV_API_KEY environment variable.
        """
        super().__init__(batch_size=batch_size)

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(prompt_seqs) == 1:
            self.prompt_seqs = prompt_seqs * batch_size
        else:
            assert (
                len(prompt_seqs) == batch_size
            ), f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
            self.prompt_seqs = prompt_seqs

        self.batch_size = batch_size
        self.api_endpoint = nim_api_url
        self.api_key = api_key or self._get_api_key_from_env()
        self.n_tokens = sequence_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.enable_sampled_probs = enable_sampled_probs
        self.verbose = verbose
        self.prepend_prompt = prepend_prompt
        self.timeout = timeout

    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get API key from environment variables.

        Returns:
            API key from NV_API_KEY environment variable, or None if not set.
        """
        return os.getenv("NV_API_KEY") or input("Paste the Run Key: ")

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as NIM Evo2 performs autoregressive generation from prompt sequences.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "NimEvo2Generator must be assigned exactly one ConstructSegment"
            )

        # Warn user if existing sequences will be overwritten
        existing_sequences = [
            seq.sequence for seq in assigned_segments.batch_sequences if seq.sequence
        ]
        if existing_sequences:
            print(
                f"Warning: NimEvo2Generator will overwrite {len(existing_sequences)} existing sequence(s) "
                f"when sample() is called due to autoregressive generation."
            )

        # Initialize _generator_output (singular) and create batch
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)

        # No model initialization needed for this generator (uses API calls)
        self._is_initialized = True

    def _make_api_request(self, prompt_seq: str) -> str:
        """
        Make API request to generate a sequence.

        Args:
            prompt_seq: The prompt sequence to generate from.

        Returns:
            Generated sequence from the API.

        Raises:
            RuntimeError: If API request fails or response cannot be parsed.
        """
        payload = {
            "sequence": prompt_seq,
            "num_tokens": self.n_tokens,
            "top_k": self.top_k,
            "enable_sampled_probs": self.enable_sampled_probs,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                self.api_endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            if self.verbose >= 1:
                print(f"API response: {result}")

            return result["sequence"]

        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse API response: {e}")

    def sample(self, prompt_seqs: Optional[List[str]] = None) -> None:
        """
        Generate sequences using the NIM Evo2 API and update generator output.

        Uses the NIM API to generate continuations from the provided prompt sequences
        or the default prompt sequences, updating the sequences in the ConstructSegment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign() or if API requests fail.
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        if self.verbose >= 1:
            print(f"Generating {len(prompts)} sequences via NIM API...")

        # Generate sequences for each prompt
        generated_sequences = []
        for i, prompt in enumerate(prompts):
            if self.verbose >= 1:
                print(f"Generating sequence {i+1}/{len(prompts)}")

            try:
                generated_seq = self._make_api_request(prompt)
                generated_sequences.append(generated_seq)
            except Exception as e:
                print(f"Failed to generate sequence {i+1}: {e}")
                # Use empty string as fallback
                generated_sequences.append("")

        # Update sequences in the ConstructSegment
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence





@final
class ESM2Generator(Generator):
    """
    A protein sequence generator using the ESM-2 protein language model.

    This generator uses the ESM-2 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen = ESM2Generator(
        ...     esm2_type="esm2_t33_650M_UR50D",
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(
        self,
        esm2_type: str = "esm2_t33_650M_UR50D",
        sequence_length: int = 100,
        temperature: float = 1.0,
        decoding_method: str = "entropy",
        top_k: int = 5,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            esm2_type: ESM-2 model variant to use. See Facebook ESM repository
                for available models.
            sequence_length: Length of protein sequences to generate.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Strategy for selecting positions to sample:
                - 'entropy': Choose positions with highest prediction entropy
                - 'max_logit': Choose positions with highest maximum logits
                - 'random': Choose positions randomly
            top_k: Number of positions to sample per iteration.
            batch_size: Number of sequences to generate simultaneously.
        """
        super().__init__(batch_size=batch_size)
        if top_k > sequence_length:
            raise ValueError(f"top_k ({top_k}) cannot exceed sequence_length ({sequence_length})")

        self.esm2_type = esm2_type
        self.sequence_length = sequence_length
        self.temperature = temperature
        self.decoding_method = decoding_method
        self.top_k = top_k
        self.batch_size = batch_size

    def assign(self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]) -> None:
        """
        Assign a ConstructSegment to this generator.

        Creates initial sequences by running ESM-2 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "ESM2Generator must be assigned exactly one ConstructSegment"
            )

        # Validate provided sequence length if not empty
        initial_sequence = assigned_segments.batch_sequences[0].sequence
        if initial_sequence != "":
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )

        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM-2 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        sequences = [self._generator_output.batch_sequences[i].sequence for i in range(self.batch_size)]

        # Choose execution mode based on configuration
        from .utils import use_cloud_gpu

        if use_cloud_gpu():
            # Use cloud for cloud GPU execution
            print("Using cloud for ESM2 sampling...")
            import cloud
            esm2_sample_cloud = cloud.Function.from_name('proto-language', 'esm2_sample_cloud')
            mutated_sequences = esm2_sample_cloud.remote(
                sequences=sequences,
                esm2_type=self.esm2_type,
                sequence_length=self.sequence_length,
                temperature=self.temperature,
                decoding_method=self.decoding_method,
                top_k=self.top_k
            )
        else:
            # Use local GPU execution
            print("Using local GPU for ESM2 sampling...")
            mutated_sequences = self._esm2_sample_gpu(
                sequences=sequences,
                esm2_type=self.esm2_type,
                sequence_length=self.sequence_length,
                temperature=self.temperature,
                decoding_method=self.decoding_method,
                top_k=self.top_k
            )

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._generator_output.batch_sequences[i].sequence = sequence

    def _esm2_sample_gpu(
        self,
        sequences: List[str],
        esm2_type: str,
        sequence_length: int,
        temperature: float,
        decoding_method: str,
        top_k: int
    ) -> List[str]:
        """
        Local GPU function for ESM2 sampling.
        
        Args:
            sequences: Protein sequences (empty strings trigger generation from scratch).
            esm2_type: ESM2 model variant to load.
            sequence_length: Target length for generated sequences.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Position scoring method ('entropy', 'max_logit', 'random').
            top_k: Number of positions to mutate per sequence.
            
        Returns:
            List of final protein sequences after mutations/generation.
        """
        # Helper functions
        def batch_forward_pass(protein_seqs: List[str]) -> torch.Tensor:
            """Process protein sequences through ESM2 model."""
            labeled_seqs = [(f"seq_{i}", seq) for i, seq in enumerate(protein_seqs)]
            _, _, tokenized_seqs = batch_converter(labeled_seqs)
            tokenized_seqs = tokenized_seqs.to(device)

            with torch.inference_mode():
                model_output = esm2_model(tokenized_seqs)
            logits = model_output["logits"]
            return logits[:, 1:-1, :]  # Remove start/end special tokens

        def sample_amino_acids(
            sequences: List[str],
            aa_logits: torch.Tensor, 
            target_positions: torch.Tensor,
            valid_token_idx: torch.Tensor,
            temp: float
        ) -> List[str]:
            """Sample amino acids from model logits and mutate sequences."""
            batch_size, num_positions = target_positions.shape
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # [batch_size, 1]

            # Extract logits for target positions: [batch_size, num_positions, vocab_size]
            target_logits = aa_logits[batch_idx, target_positions]

            # Filter to valid amino acid vocabulary only: [batch_size, num_positions, num_valid_tokens]
            filtered_logits = target_logits[:, :, valid_token_idx]

            # Apply temperature scaling and convert to probabilities
            scaled_logits = filtered_logits / max(temp, 1e-8)
            token_probs = torch.softmax(scaled_logits, dim=2)

            # Flatten for multinomial sampling and sample
            flat_probs = token_probs.view(-1, len(valid_token_idx))  # Flatten for multinomial
            sampled_token_idx = torch.multinomial(flat_probs, 1).squeeze(1)
            sampled_token_idx = sampled_token_idx.view(batch_size, num_positions)  # Reshape back

            # Convert vocabulary indices to ESM token indices
            sampled_tokens = valid_token_idx[sampled_token_idx]

            # Apply to sequences (generation or mutation)
            selected_positions_list = target_positions.cpu().tolist()
            mutated_sequences = []
            for orig_seq, pos_list, token_list in zip(sequences, selected_positions_list, sampled_tokens.cpu().tolist()):
                # Convert tokens to amino acids
                new_amino_acids = [alphabet.get_tok(idx) for idx in token_list]

                if orig_seq == "":  # Generation: create sequence from amino acids
                    mutated_sequences.append(''.join(new_amino_acids))
                else:  # Mutation: apply mutations to existing sequence
                    mutated = orig_seq
                    for pos, new_aa in zip(pos_list, new_amino_acids):
                        mutated = mutated[:pos] + new_aa + mutated[pos + 1:]
                    mutated_sequences.append(mutated)
            return mutated_sequences

        def sample_top_k_positions_batch(aa_logits: torch.Tensor, decoding_method: str, k: int) -> torch.Tensor:
            """Select top-k positions to mutate based on model uncertainty."""
            # Compute position uncertainty scores based on decoding method
            if decoding_method == "entropy":
                uncertainty_scores = -torch.sum(torch.softmax(aa_logits, dim=-1) * torch.log_softmax(aa_logits, dim=-1), dim=-1)
            elif decoding_method == "max_logit":
                uncertainty_scores = -torch.max(aa_logits, dim=-1)[0]
            elif decoding_method == "random":
                uncertainty_scores = torch.rand(aa_logits.shape[:-1], device=device)
            else:
                raise ValueError(f"Unknown decoding method: {decoding_method}. Must be one of ['entropy', 'max_logit', 'random']")

            # Convert uncertainty scores to position selection probabilities
            position_probs = torch.softmax(uncertainty_scores, dim=1)  # [batch_size, seq_len]
            selected_positions = torch.multinomial(position_probs, k, replacement=False)
            return selected_positions

        def initialize_random_seqs(
            num_seqs: int,
            seq_length: int,
            valid_token_idx: torch.Tensor,
            temp: float
        ) -> List[str]:
            """Generate random protein sequences by sampling from masked tokens."""
            # Create masked sequences and get model predictions
            masked_seqs = [" ".join(["<mask>"] * seq_length)] * num_seqs
            mask_logits = batch_forward_pass(masked_seqs)

            # Sample all positions (unmask everything)
            all_positions = torch.tensor(
                [list(range(seq_length))] * num_seqs, 
                device=device
            )

            # Use the consolidated sampling function with empty sequences
            empty_sequences = [""] * num_seqs
            return sample_amino_acids(empty_sequences, mask_logits, all_positions, valid_token_idx, temp)

        # Requires GPU to run
        device = "cuda"

        # Load ESM2 model and setup
        esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", esm2_type)
        batch_converter = alphabet.get_batch_converter()
        esm2_model = esm2_model.to(device)
        esm2_model.eval()

        # Create tensor of valid amino acid token indices (exclude ambiguous B, J, X, Z)
        valid_token_idx = torch.tensor([
            alphabet.get_idx(token) for token in alphabet.standard_toks 
            if token not in "BJXZ"
        ], device=device)

        # Check if this is the first call (all input sequences are empty strings)
        if all(seq == "" for seq in sequences):
            return initialize_random_seqs(len(sequences), sequence_length, valid_token_idx, temperature)

        # Mutate existing sequences at selected positions
        seq_logits = batch_forward_pass(sequences)
        target_positions = sample_top_k_positions_batch(seq_logits, decoding_method, top_k)

        return sample_amino_acids(sequences, seq_logits, target_positions, valid_token_idx, temperature)


@final
class ESM3Generator(Generator):
    """
    A protein sequence generator using the ESM-3 open protein language model.

    This generator uses the (open) ESM-3 protein language model to propose sequences and
    mutations based on the model's logits. It supports various decoding strategies
    for selecting positions to mutate and uses temperature-controlled sampling
    for amino acid selection.

    Examples:
        Basic protein generation:
        >>> segment = ConstructSegment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen = ESM3Generator(
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5,
        ...     batch_size=3
        ... )
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(
        self,
        sequence_length: int = 100,
        temperature: float = 1.0,
        decoding_method: str = "entropy",
        top_k: int = 5,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ESM3 generator with model and sampling configuration.

        Args:
            sequence_length: Length of protein sequences to generate.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Strategy for selecting positions to sample:
                - 'entropy': Choose positions with highest prediction entropy
                - 'max_logit': Choose positions with highest maximum logits
                - 'random': Choose positions randomly
            top_k: Number of positions to sample per iteration.
            batch_size: Number of sequences to generate simultaneously.
        """
        super().__init__(batch_size=batch_size)
        if top_k > sequence_length:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed sequence_length ({sequence_length})"
            )

        self.sequence_length = sequence_length
        self.temperature = temperature
        self.decoding_method = decoding_method
        self.top_k = top_k
        self.batch_size = batch_size

    def assign(
        self, assigned_segments: ConstructSegment | Iterable[ConstructSegment]
    ) -> None:
        """
        Assign a ConstructSegment to this generator.

        Creates initial sequences by running ESM3 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: Either a single ConstructSegment or an iterable of ConstructSegment objects.

        Raises:
            ValueError: If more than one segment is provided.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        # Ensure single ConstructSegment assignment
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError(
                "ESM3Generator must be assigned exactly one ConstructSegment"
            )

        # Validate provided sequence length if not empty
        initial_sequence = assigned_segments.batch_sequences[0].sequence
        if initial_sequence != "":
            assert len(initial_sequence) == self.sequence_length, (
                f"Provided sequence length ({len(initial_sequence)}) must match "
                f"configured sequence_length ({self.sequence_length})"
            )

        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM3 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()
        sequences = [
            self._generator_output.batch_sequences[i].sequence
            for i in range(self.batch_size)
        ]

        # Choose execution mode based on configuration
        from .utils import use_cloud_gpu

        if use_cloud_gpu():
            # Use cloud for cloud GPU execution
            print("Using cloud for ESM3 sampling...")
            import cloud

            raise NotImplementedError("esm3_sample_cloud is not implemented yet")

            esm3_sample_cloud = cloud.Function.from_name(
                "proto-language", "esm3_sample_cloud"
            )
            mutated_sequences = esm3_sample_cloud.remote(
                sequences=sequences,
                esm2_type=self.esm2_type,
                sequence_length=self.sequence_length,
                temperature=self.temperature,
                decoding_method=self.decoding_method,
                top_k=self.top_k,
            )
        else:
            # Use local GPU execution
            print("Using local GPU for ESM3 sampling...")
            mutated_sequences = self._esm3_sample_gpu(
                sequences=sequences,
                sequence_length=self.sequence_length,
                temperature=self.temperature,
                decoding_method=self.decoding_method,
                top_k=self.top_k,
            )

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._generator_output.batch_sequences[i].sequence = sequence

    def _esm3_sample_gpu(
        self,
        sequences: List[str],
        sequence_length: int,
        temperature: float,
        decoding_method: str,
        top_k: int,
    ) -> List[str]:
        """
        Local GPU function for ESM3 sampling.

        Args:
            sequences: Protein sequences (empty strings trigger generation from scratch).
            sequence_length: Target length for generated sequences.
            temperature: Sampling temperature for amino acid selection.
            decoding_method: Position scoring method ('entropy', 'max_logit', 'random').
            top_k: Number of positions to mutate per sequence.

        Returns:
            List of final protein sequences after mutations/generation.
        """

        # Helper functions
        def batch_forward_pass(protein_seqs: List[str]) -> torch.Tensor:
            """Process protein sequences through ESM3 model."""

            tokenized_input = esm3_tokenizer.batch_encode_plus(
                protein_seqs,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            tokenized_input = tokenized_input.to(device)

            with torch.inference_mode():
                model_output = esm3_model(
                    sequence_tokens=tokenized_input["input_ids"],
                )

            logits = model_output.sequence_logits
            return logits[:, 1:-1, :]  # Remove start/end special tokens

        def sample_amino_acids(
            sequences: List[str],
            aa_logits: torch.Tensor,
            target_positions: torch.Tensor,
            valid_token_idx: torch.Tensor,
            temp: float,
        ) -> List[str]:
            """Sample amino acids from model logits and mutate sequences."""
            batch_size, num_positions = target_positions.shape
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(
                1
            )  # [batch_size, 1]

            # Extract logits for target positions: [batch_size, num_positions, vocab_size]
            target_logits = aa_logits[batch_idx, target_positions]

            # Filter to valid amino acid vocabulary only: [batch_size, num_positions, num_valid_tokens]
            filtered_logits = target_logits[:, :, valid_token_idx]

            # Apply temperature scaling and convert to probabilities
            scaled_logits = filtered_logits / max(temp, 1e-8)
            token_probs = torch.softmax(scaled_logits, dim=2)

            # Flatten for multinomial sampling and sample
            flat_probs = token_probs.view(
                -1, len(valid_token_idx)
            )  # Flatten for multinomial
            sampled_token_idx = torch.multinomial(flat_probs, 1).squeeze(1)
            sampled_token_idx = sampled_token_idx.view(
                batch_size, num_positions
            )  # Reshape back

            # Convert vocabulary indices to ESM token indices
            sampled_tokens = valid_token_idx[sampled_token_idx]

            # Apply to sequences (generation or mutation)
            selected_positions_list = target_positions.cpu().tolist()
            mutated_sequences = []
            for orig_seq, pos_list, token_list in zip(
                sequences, selected_positions_list, sampled_tokens.cpu().tolist()
            ):
                # Convert tokens to amino acids
                new_amino_acids = [
                    esm3_tokenizer.convert_ids_to_tokens(idx) for idx in token_list
                ]

                if orig_seq == "":  # Generation: create sequence from amino acids
                    mutated_sequences.append("".join(new_amino_acids))
                else:  # Mutation: apply mutations to existing sequence
                    mutated = orig_seq
                    for pos, new_aa in zip(pos_list, new_amino_acids):
                        mutated = mutated[:pos] + new_aa + mutated[pos + 1 :]
                    mutated_sequences.append(mutated)
            return mutated_sequences

        def sample_top_k_positions_batch(
            aa_logits: torch.Tensor, decoding_method: str, k: int
        ) -> torch.Tensor:
            """Select top-k positions to mutate based on model uncertainty."""
            # Compute position uncertainty scores based on decoding method
            if decoding_method == "entropy":
                uncertainty_scores = -torch.sum(
                    torch.softmax(aa_logits, dim=-1)
                    * torch.log_softmax(aa_logits, dim=-1),
                    dim=-1,
                )
            elif decoding_method == "max_logit":
                uncertainty_scores = -torch.max(aa_logits, dim=-1)[0]
            elif decoding_method == "random":
                uncertainty_scores = torch.rand(aa_logits.shape[:-1], device=device)
            else:
                raise ValueError(
                    f"Unknown decoding method: {decoding_method}. Must be one of ['entropy', 'max_logit', 'random']"
                )

            # Convert uncertainty scores to position selection probabilities
            position_probs = torch.softmax(
                uncertainty_scores, dim=1
            )  # [batch_size, seq_len]
            selected_positions = torch.multinomial(position_probs, k, replacement=False)
            return selected_positions

        def initialize_random_seqs(
            num_seqs: int, seq_length: int, valid_token_idx: torch.Tensor, temp: float
        ) -> List[str]:
            """Generate random protein sequences by sampling from masked tokens."""
            # Create masked sequences and get model predictions
            masked_seqs = ["".join(["<mask>"] * seq_length)] * num_seqs
            mask_logits = batch_forward_pass(masked_seqs)

            # Sample all positions (unmask everything)
            all_positions = torch.tensor(
                [list(range(seq_length))] * num_seqs, device=device
            )

            # Use the consolidated sampling function with empty sequences
            empty_sequences = [""] * num_seqs
            return sample_amino_acids(
                empty_sequences, mask_logits, all_positions, valid_token_idx, temp
            )

        # Requires GPU to run
        device = torch.device("cuda")

        # Load ESM3 model and setup
        from esm.models.esm3 import ESM3
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        open_model_name = "esm3-sm-open-v1"
        esm3_model = ESM3.from_pretrained(model_name=open_model_name, device=device)
        esm3_tokenizer = EsmSequenceTokenizer()
        esm3_model.eval()

        # Create tensor of valid amino acid token indices
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        valid_token_idx = torch.tensor(
            [esm3_tokenizer.convert_tokens_to_ids(token) for token in amino_acids],
            device=device,
        )

        # Check if this is the first call (all input sequences are empty strings)
        if all(seq == "" for seq in sequences):
            return initialize_random_seqs(
                len(sequences), sequence_length, valid_token_idx, temperature
            )

        # Mutate existing sequences at selected positions
        seq_logits = batch_forward_pass(sequences)
        target_positions = sample_top_k_positions_batch(
            seq_logits, decoding_method, top_k
        )

        return sample_amino_acids(
            sequences, seq_logits, target_positions, valid_token_idx, temperature
        )


@final
class SlowMutationGenerator(Generator):
    """A generator that introduces mutations slowly for testing and demonstration purposes."""
    
    def __init__(self, batch_size: int = 1, sequence_length: int = 20, sleep_time: float = 2.0):
        super().__init__(batch_size=batch_size)
        self.sequence_length = sequence_length
        self.sleep_time = sleep_time
        
    def assign(self, assigned_segments):
        if not isinstance(assigned_segments, ConstructSegment):
            raise ValueError("SlowMutationGenerator must be assigned exactly one ConstructSegment")
        
        self._generator_output = assigned_segments
        self._generator_output._is_assigned = True
        
        # Initialize with random sequence if empty
        if self._generator_output.batch_sequences[0].sequence == "":
            valid_chars = list(self._generator_output._valid_chars - set(" "))
            initial_sequence = "".join(
                random.choice(valid_chars) for _ in range(self.sequence_length)
            )
            self._generator_output.batch_sequences[0].sequence = initial_sequence
            
        self._generator_output.create_batch(self.batch_size)
        self._is_initialized = True
        
    def sample(self):
        """Sample with a sleep to simulate slow processing."""
        self._validate_generator()
        
        time.sleep(self.sleep_time)
        
        # Simple mutation logic
        for sequence in self._generator_output.batch_sequences:
            if len(sequence.sequence) > 0:
                # Mutate a random position
                mutated_index = random.randint(0, len(sequence.sequence) - 1)
                current_sequence = sequence.sequence
                current_char = current_sequence[mutated_index]
                
                # Get valid mutations
                possible_mutations = [
                    c for c in self._generator_output._valid_chars 
                    if c != current_char and c != " "
                ]
                
                if possible_mutations:
                    mutated_char = random.choice(possible_mutations)
                    new_sequence = (
                        current_sequence[:mutated_index] + 
                        mutated_char + 
                        current_sequence[mutated_index + 1:]
                    )
                    sequence.sequence = new_sequence


@final
class MCMCGenerator(IterativeGenerator):
    """
    Metropolis-Hastings MCMC generator for constraint-driven sequence optimization.

    This generator implements a Metropolis-Hastings sampling algorithm that uses
    multiple sub-generators as proposal distributions and constraints to define
    the energy function. It's designed for iterative sequence refinement where
    proposals are accepted or rejected based on energy improvements.

    The generator supports simulated annealing, multiple constraints with weights,
    and flexible sequence concatenation for complex multi-part designs.

    Examples:
        Basic MCMC optimization:
        >>> constructs = [Construct([segment1, segment2])]
        >>> mcmc = MCMCGenerator(
        ...     constructs=constructs,
        ...     generators=[evo2_gen, mutation_gen],
        ...     constraints=[gc_constraint, homopolymer_constraint],
        ...     constraint_weights=[1.0, 2.0],  # Weight homopolymer constraint more
        ...     num_steps=100,
        ...     temperature=0.5,  # More greedy sampling
        ...     temperature_min=0.001
        ... )
        >>> mcmc.sample()
        >>> final_constructs = mcmc.constructs
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        num_steps: int = 1,
        temperature: float = 1.0,
        temperature_min: float = 0.0001,
        track_step_size: int = 10,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MCMC generator with sub-generators and constraints.

        Args:
            constructs: List of Construct objects to optimize.
            generators: List of Generator objects to generate sequences.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: Optional weights for constraints. If None, all weights are 1.0.
            num_steps: Number of MCMC steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Interval for progress tracking.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.

        Raises:
            ValueError: If any validation checks fail.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        self.num_steps: int = num_steps
        self.temperature: float = temperature
        self.temperature_min: float = temperature_min
        self.track_step_size: int = track_step_size
        self.custom_logging: Optional[
            Callable[[int, Tuple[ConstructSegment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for MCMCGenerator.

        Raises:
            ValueError: If temperature parameters are invalid.
        """
        super()._validate_generator()

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

    def sample(self) -> None:
        """
        Execute Metropolis-Hastings MCMC sampling for sequence optimization.

        Runs the specified number of MCMC steps, where each step:
        1. Selects a random sub-generator
        2. Proposes sequence changes via that generator
        3. Evaluates energy change using constraints
        4. Accepts or rejects based on Metropolis-Hastings criterion
        5. Optionally logs progress and tracks state

        Note:
            - Temperature annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
            - Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize MCMC states
        energies = self.score_energy()
        current_best_energy = np.min(energies)
        current_best_idx = np.argmin(energies)
        self.history.append(copy.deepcopy(self.constructs))

        # Execute MCMC optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            cur_temp = self.temperature * (self.temperature_min / self.temperature) ** (step / self.num_steps)

            # Execute single MCMC step
            current_best_energy, current_best_idx = self._execute_mcmc_step(
                step, cur_temp, current_best_energy, current_best_idx
            )

            # Track progress periodically
            if step % self.track_step_size == 0:
                self.history.append(copy.deepcopy(self.constructs))
        
        # Always store final state, even if not a tracked step
        if self.num_steps % self.track_step_size != 0:
            self.history.append(copy.deepcopy(self.constructs))

    def _execute_mcmc_step(
        self,
        step: int,
        cur_temp: float,
        current_best_energy: float,
        current_best_idx: int,
    ) -> Tuple[float, int]:
        """
        Execute a single MCMC step including proposal, evaluation, and acceptance decision.

        Args:
            step: Current step number.
            cur_temp: Current temperature for this step.
            current_best_energy: Current best energy value.
            current_best_idx: Index of current best sequence.

        Returns:
            Tuple of (updated_best_energy, updated_best_idx).
        """
        # 1. Pick generator and store old sequences for potential revert
        generator = random.choice(self.generators)
        old_generator_outputs = copy.deepcopy(generator.get_generator_outputs())

        # 2. Sample new proposal and evaluate
        generator.sample()
        new_energies = self.score_energy()
        new_best_energy = np.min(new_energies)
        new_best_idx = np.argmin(new_energies)

        # 3. Accept or reject proposal according to Metropolis-Hastings algorithm
        original_best_energy = current_best_energy  # Save original for logging
        current_best_energy, current_best_idx, accept, alpha = (
            self._accept_or_reject_proposal(
                current_best_energy,
                current_best_idx,
                new_best_energy,
                new_best_idx,
                cur_temp,
                generator,
                old_generator_outputs,
            )
        )

        # 4. Log progress
        if self.verbose and step % self.track_step_size == 0:
            self._log_step(
                step,
                original_best_energy,
                new_best_energy,
                alpha,
                accept,
                current_best_idx,
                cur_temp,
            )

        return current_best_energy, current_best_idx

    def _accept_or_reject_proposal(
        self,
        current_best_energy: float,
        current_best_idx: int,
        new_best_energy: float,
        new_best_idx: int,
        cur_temp: float,
        generator: Generator,
        old_generator_outputs: Tuple[ConstructSegment, ...],
    ) -> Tuple[float, int, bool, float]:
        """
        Compute Metropolis-Hastings acceptance probability and execute the decision.

        Args:
            current_best_energy: Energy of current best sequence.
            current_best_idx: Index of current best sequence.
            new_best_energy: Energy of proposed sequence.
            new_best_idx: Index of proposed best sequence.
            cur_temp: Current temperature for acceptance calculation.
            generator: The generator that made the proposal.
            old_generator_outputs: Backup of sequences before proposal.

        Returns:
            Tuple of (final_best_energy, final_best_idx, accept, alpha).
        """
        # Compute acceptance probability
        energy_diff = -(new_best_energy - current_best_energy) / cur_temp
        energy_diff = min(energy_diff, MAX_EXP_ARG)  # Clamp to prevent overflow
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        accept = random.random() < alpha

        # Execute the decision
        if accept:
            # Accept: copy best sequences to all positions
            self._replicate_best_sequence(new_best_idx)
            return new_best_energy, new_best_idx, accept, alpha
        else:
            # Reject: revert the sampled generator's sequences and metadata
            for i, sequence_batch in enumerate(generator.get_generator_outputs()):
                for j, program_seq in enumerate(sequence_batch):
                    program_seq.sequence = (
                        old_generator_outputs[i].batch_sequences[j].sequence
                    )
                    program_seq._metadata = (
                        old_generator_outputs[i].batch_sequences[j]._metadata.copy()
                    )
            return current_best_energy, current_best_idx, accept, alpha

    def _log_step(
        self,
        step: int,
        old_energy: float,
        new_energy: float,
        alpha: float,
        accept: bool,
        best_idx: int,
        cur_temp: float,
    ) -> None:
        """
        Log information about the current MCMC step.

        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            best_idx: Index of best sequence.
            cur_temp: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {cur_temp:.6f}, "
            f"accept: {accept}, "
            f"best_idx: {best_idx}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()


@final
class SequentialGenerator(IterativeGenerator):
    """
    Sequential generator for chaining autoregressive sequence generators.

    Applies multiple generators in sequence where each uses the previous generator's
    output as input prompts. After all generators run, accepts or rejects the
    combined changes based on energy improvement and temperature annealing.

    Requirements:
    - All generators must output exactly one ConstructSegment
    - Generators after the first must accept prompt_seqs parameter in sample()

    Examples:
        Basic sequential chaining:
        >>> constructs = [Construct([segment1, segment2])]
        >>> sequential = SequentialGenerator(
        ...     constructs=constructs,
        ...     generators=[gen1, gen2, gen3],  # Chain: gen1 -> gen2(gen1_out) -> gen3(gen2_out)
        ...     constraints=[constraint1, constraint2],
        ...     constraint_weights=[1.0, 2.0],  # Weight constraint2 more heavily
        ...     num_steps=50,
        ...     temperature=0.8,  # Accept/reject after all generators
        ...     temperature_min=0.001
        ... )
        >>> sequential.sample()
        >>> final_sequences = sequential.constructs

    Notes:
        - Final sequences: initial_prompt + gen1_output + gen2_output + ...
        - Temperature annealing: T(step) = T_max * (T_min / T_max) ^ (step / num_steps)
    """

    def __init__(
        self,
        constructs: List[Construct],
        generators: List[Generator],
        constraints: List[Constraint],
        constraint_weights: Optional[List[float]] = None,
        num_steps: int = 1,
        temperature: float = 1.0,
        temperature_min: float = 0.0001,
        track_step_size: int = 10,
        custom_logging: Optional[Callable[[int, Sequence], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the sequential generator with ordered sub-generators.

        Args:
            constructs: List of Construct objects to be optimized.
            generators: List of Generator objects to be chained sequentially.
            constraints: List of Constraint objects to evaluate sequences.
            constraint_weights: List of weights for each constraint. If None, all weights are 1.0.
            num_steps: Number of optimization steps per sample() call.
            temperature: Maximum temperature for annealing.
            temperature_min: Minimum temperature for annealing.
            track_step_size: Progress tracking interval.
            custom_logging: Custom logging function that takes (step, sequences) arguments.
            verbose: Whether to print progress information.
        """
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        self.num_steps: int = num_steps
        self.temperature: float = temperature
        self.temperature_min: float = temperature_min
        self.track_step_size: int = track_step_size
        self.custom_logging: Optional[
            Callable[[int, Tuple[ConstructSegment, ...]], None]
        ] = custom_logging
        self.verbose: bool = verbose

        self._validate_generator()

    def _validate_generator(self) -> None:
        """
        Validate configuration for SequentialGenerator.

        Raises:
            ValueError: If generators have different batch sizes or temperature parameters are invalid.
        """
        super()._validate_generator()

        # Check that all batch sizes are the same
        batch_sizes = [gen.batch_size for gen in self.generators]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"All generators must have the same batch_size. Found: {batch_sizes}"
            )

        # Validate temperature parameters
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(
                f"temperature_min must be positive, got {self.temperature_min}"
            )
        if self.temperature_min >= self.temperature:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) must be less than temperature ({self.temperature}) for annealing to work properly"
            )

    def sample(self) -> None:
        """
        Execute sequential sampling with chained autoregressive generators.

        Each step: (1) applies all generators sequentially with chaining,
        (2) evaluates energy change, (3) accepts/rejects based on Metropolis-Hastings
        with temperature annealing.

        Snapshots of constructs at tracked timesteps are stored in self.history.
        """
        # Initialize sequential states
        old_energies = self.score_energy()
        current_best_energy = np.min(old_energies)
        self.history.append(copy.deepcopy(self.constructs))

        # Execute sequential optimization steps
        for step in range(1, self.num_steps + 1):
            self.current_step = step
            cur_temp = self.temperature * (self.temperature_min / self.temperature) ** (
                step / self.num_steps
            )

            # Execute single sequential step
            current_best_energy = self._execute_sequential_step(
                step, cur_temp, current_best_energy
            )

            # Track progress periodically
            if step % self.track_step_size == 0:
                self.history.append(copy.deepcopy(self.constructs))
        
        # Always capture final state if it wasn't already captured
        if self.num_steps % self.track_step_size != 0:
            self.history.append(copy.deepcopy(self.constructs))

    def _execute_sequential_step(
        self, step: int, cur_temp: float, current_best_energy: float
    ) -> float:
        """
        Execute a single sequential step including chaining, evaluation, and acceptance decision.

        Args:
            step: Current step number.
            cur_temp: Current temperature for this step.
            current_best_energy: Current best energy value.

        Returns:
            Updated best energy value.
        """
        # 1. Store old sequences for potential revert
        old_sequences_by_gen = self._backup_sequences()

        # 2. Apply all generators sequentially with chaining
        self._sample_sequential_generators()

        # 3. Evaluate new energy
        new_energies = self.score_energy()
        new_best_energy = np.min(new_energies)

        # 4. Accept or reject proposal according to Metropolis-Hastings algorithm
        original_best_energy = current_best_energy  # Save original for logging
        current_best_energy, accept, alpha = self._accept_or_reject_proposal(
            current_best_energy,
            new_best_energy,
            cur_temp,
            old_sequences_by_gen,
            new_energies,
        )

        # 5. Log progress
        if self.verbose and step % self.track_step_size == 0:
            self._log_step(
                step, original_best_energy, new_best_energy, alpha, accept, cur_temp
            )

        return current_best_energy

    def _backup_sequences(self) -> List[List[Any]]:
        """
        Create backup copies of all sequences from all generators.

        Returns:
            List of backed up sequences organized by generator.
        """
        old_sequences_by_gen = []
        for generator in self.generators:
            gen_old_seqs = []
            for sequence_batch in generator.get_generator_outputs():
                for program_seq in sequence_batch:
                    gen_old_seqs.append(copy.deepcopy(program_seq))
            old_sequences_by_gen.append(gen_old_seqs)
        return old_sequences_by_gen

    def _sample_sequential_generators(self) -> None:
        """
        Apply all generators sequentially, chaining outputs between them.

        Each generator uses the accumulated output from previous generators
        as prompts for its own generation.
        """
        first_gen = self.generators[0]

        # Initialize running_prompts based on the first generator type
        if hasattr(first_gen, 'prompt_seqs'):
            # For generators that accept prompts
            running_prompts = first_gen.prompt_seqs.copy()
        else:
            # For generators that don't accept prompts
            outputs = first_gen.get_generator_outputs()
            if outputs:
                batch = outputs[0]
                running_prompts = [seq.sequence for seq in batch.batch_sequences]
            else:
                running_prompts = [""] * first_gen.batch_size

        # Sample from each generator in sequence, chaining outputs
        for i, generator in enumerate(self.generators):
            # For generators that accept prompts
            if self._is_extension_based_generator(generator):
                prompt_seqs = running_prompts if i > 0 else None
                generator.sample(prompt_seqs=prompt_seqs)
            else:
                # For generators that don't accept prompts
                generator.sample()

            # Accumulate this generator's output
            outputs = generator.get_generator_outputs()
            assert (
                len(outputs) == 1
            ), f"Generator {i} must output exactly one ConstructSegment for chaining"
            batch = outputs[0]

            # Update running_prompts with the generator's output
            if hasattr(generator, 'prompt_seqs') or i == 0:
                for batch_idx in range(len(batch)):
                    if i == 0 and getattr(generator, "prepend_prompt", False):
                        # First generator with prepend_prompt: output already includes prompt content,
                        # just add back the prefix tokens that were stripped
                        original_prompt = running_prompts[batch_idx] if hasattr(generator, 'prompt_seqs') else ""
                        generated = batch[batch_idx].sequence
                        valid_chars = batch._valid_chars or set()
                        prefix_tokens = "".join(
                            c for c in original_prompt if c not in valid_chars
                        )
                        running_prompts[batch_idx] = prefix_tokens + generated
                    else:
                        # Normal case: accumulate output to running prompts
                        running_prompts[batch_idx] += batch[batch_idx].sequence
            else:
                # For generators that don't accept prompts
                running_prompts = [seq.sequence for seq in batch.batch_sequences]

    def _accept_or_reject_proposal(
        self,
        current_best_energy: float,
        new_best_energy: float,
        cur_temp: float,
        old_sequences_by_gen: List[List[Any]],
        new_energies: List[float],
    ) -> Tuple[float, bool, float]:
        """
        Compute Metropolis-Hastings acceptance probability and make decision.

        Args:
            current_best_energy: Energy of current best sequence.
            new_best_energy: Energy of proposed sequence.
            cur_temp: Current temperature for acceptance calculation.
            old_sequences_by_gen: Backup of sequences before proposal.
            new_energies: All energy values for the new sequences.

        Returns:
            Tuple of (updated_best_energy, accept, alpha).
        """
        # Compute acceptance probability
        energy_diff = -(new_best_energy - current_best_energy) / cur_temp
        energy_diff = min(energy_diff, MAX_EXP_ARG)  # Clamp to prevent overflow
        alpha = np.exp(energy_diff)
        alpha = min(1.0, alpha)
        accept = random.random() < alpha

        # Execute the decision
        if accept:
            # Accept: copy best sequences to all positions
            new_best_idx = np.argmin(new_energies)
            self._replicate_best_sequence(new_best_idx)
            return new_best_energy, accept, alpha
        else:
            # Revert changes if rejected
            for i, generator in enumerate(self.generators):
                seq_idx = 0
                for sequence_batch in generator.get_generator_outputs():
                    for program_seq in sequence_batch:
                        program_seq.sequence = old_sequences_by_gen[i][seq_idx].sequence
                        program_seq._metadata = old_sequences_by_gen[i][
                            seq_idx
                        ]._metadata.copy()
                        seq_idx += 1
            return current_best_energy, accept, alpha

    def _log_step(
        self,
        step: int,
        old_energy: float,
        new_energy: float,
        alpha: float,
        accept: bool,
        cur_temp: float,
    ) -> None:
        """
        Log information about the current sequential generation step.

        Args:
            step: Current step number.
            old_energy: Energy before proposal.
            new_energy: Energy after proposal.
            alpha: Acceptance probability.
            accept: Whether proposal was accepted.
            cur_temp: Current temperature.
        """
        print(
            f"Iteration {step} | "
            f"old best energy: {old_energy:.4f}, "
            f"new best energy: {new_energy:.4f}, "
            f"alpha: {alpha:.4f}, "
            f"temperature: {cur_temp:.6f}, "
            f"accept: {accept}"
        )
        if self.custom_logging:
            self.custom_logging(step, self.get_generator_outputs())
        sys.stdout.flush()


@final
class BeamSearchGenerator(IterativeGenerator):
    """
    Beam search generator that processes segments sequentially with context accumulation.
    
    This generator implements a sequential beam search where:
    1. Segments are processed one at a time, in order
    2. For each segment, the top K sequences accumulated from previous segments are used as prompts
    3. Generators are applied sequentially within each segment to generate num_candidates
    4. Constraints are evaluated on concatenated sequences after each segment
    5. Top K combinations are selected and used as prompts for the next segment
    
    **Key Features:**
    - Processes segments sequentially (not independently)
    - Accumulates context from previous segments as prompts
    - Applies constraints after each segment to guide optimization
    - Maintains beam search across segment boundaries
    - Freezes optimization of earlier segments as new segments are added
    
    **Important: Generator Batch Size Override**
    - Generator `batch_size` parameters are ignored during beam search
    - The `beam_width` parameter controls how many sequences are maintained (K)
    - The `num_candidates` parameter controls how many candidates are generated per beam (N)
    - Generators are applied to individual sequences, not batches

    Args:
        generators: List of Generator objects for sequence modification
        constraints: List of Constraint objects for evaluation
        constructs: List containing exactly one Construct object to optimize
        constraint_weights: Optional weights for constraints
        beam_width: Number of candidates to maintain per sequence (K). This overrides
                   any `batch_size` parameters of individual generators.
        num_candidates: Number of candidates to generate per beam candidate (N)
        temperature: Temperature for candidate generation (default: 1.0)
        verbose: Whether to print progress information
        
    Raises:
        ValueError: If no constructs are provided or if more than one construct is provided.
    """
    
    def __init__(
        self,
        generators: List[Generator],
        constraints: List[Constraint],
        constructs: List[Construct],
        constraint_weights: Optional[List[float]] = None,
        beam_width: int = 5,
        num_candidates: int = 10,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> None:
        # Validate constructs parameter
        if len(constructs) == 0:
            raise ValueError("At least one construct must be provided")
        if len(constructs) > 1:
            raise ValueError(f"BeamSearchGenerator only supports a single construct, but {len(constructs)} constructs were provided")
        
        super().__init__(
            constructs=constructs,
            generators=generators,
            constraints=constraints,
            constraint_weights=constraint_weights,
        )
        
        self.beam_width = beam_width
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.verbose = verbose
        
        # Cache expensive operations
        self._segment_generators_map: Dict[ConstructSegment, List[Generator]] = {}
        
        # Initialize beam candidates for each segment
        self._initialize_beam_candidates()
    
    def _initialize_beam_candidates(self) -> None:
        """Initialize beam candidates by calling create_batch(K) on each segment."""
        for construct in self.constructs:
            for segment in construct.segments:
                if len(segment.batch_sequences) != self.beam_width:
                    segment.create_batch(self.beam_width)
        
        # Populate caches for expensive operations
        self._populate_caches()
        
        # After adjusting segment batch sizes, recreate constraints with correct batch size
        self._recreate_constraints_with_correct_batch_size()
    
    def _recreate_constraints_with_correct_batch_size(self) -> None:
        """Recreate constraints with the correct batch size after segments have been adjusted."""
        # Get the actual batch size from segments (should be beam_width)
        actual_batch_size = len(self.constructs[0].segments[0].batch_sequences)
        
        # Adjust the batch_size of each constraint.
        for constraint in self.constraints:
            constraint.batch_size = actual_batch_size
    
    def _generate_candidates_for_segment_with_prompts(self, segment: ConstructSegment, prompts: List[str]) -> List[Sequence]:
        """
        Generate candidates for a segment using accumulated prompts from previous segments.
        
        Args:
            segment: The segment to generate candidates for
            prompts: List of accumulated sequences from previous segments to use as prompts
            
        Returns:
            List of generated candidate sequences with metadata
        """
        candidates = []
        
        # Create a temporary segment for generation
        temp_segment = ConstructSegment(sequence_type=segment.sequence_type)
        temp_sequence = Sequence(sequence="", sequence_type=segment.sequence_type, metadata={})
        temp_segment.batch_sequences = [temp_sequence]
        
        # Get generators assigned to this segment
        segment_generators = self._get_segment_generators(segment)
        
        # Get the segment's current sequence to use as the base prompt
        # For now, always use the current sequence in the segment
        # This will be the extended sequence from the previous sample for subsequent segments
        segment_current_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""
        
        # Generate candidates for each prompt (each prompt represents a beam from previous segments)
        for prompt_idx, accumulated_prompt in enumerate(prompts):
            for candidate_idx in range(self.num_candidates):
                full_prompt = accumulated_prompt + segment_current_sequence
                
                # Initialize temp_sequence with the full prompt and metadata
                temp_sequence.sequence = full_prompt
                temp_sequence._metadata = {
                    "accumulated_prompt": accumulated_prompt,
                    "segment_current_sequence": segment_current_sequence,
                    "full_prompt": full_prompt,
                    "prompt_idx": prompt_idx,
                    "candidate_idx": candidate_idx,
                    "generation_steps": []
                }
                
                # Apply each generator assigned to this segment in sequence
                current_prompt = full_prompt  # Start with the full prompt
                final_extended_sequence = segment_current_sequence  # Start with current segment sequence
                
                for generator_idx, generator in enumerate(segment_generators):
                    accepts_prompts = self._is_extension_based_generator(generator)
                    
                    # Update temp_sequence with current prompt for this generator
                    temp_sequence.sequence = current_prompt
                    
                    # Apply the generator to get a new sequence
                    new_sequence = self._apply_generator(
                        generator, temp_segment, temp_sequence, 
                        accepts_prompts
                    )
                    
                    if new_sequence:
                        # Determine if this is an extension-based or mutation-based generator
                        is_extension_based = self._is_extension_based_generator(generator)
                        
                        if is_extension_based:
                            # For extension-based generators, extract the extension and add it
                            if new_sequence.startswith(current_prompt):
                                segment_extension = new_sequence[len(current_prompt):]
                            else:
                                segment_extension = new_sequence
                            final_extended_sequence += segment_extension
                        else:
                            # For mutation-based generators, replace the segment sequence
                            # Extract only the segment part from the full sequence
                            if new_sequence.startswith(accumulated_prompt):
                                segment_part = new_sequence[len(accumulated_prompt):]
                            else:
                                # If the sequence doesn't start with the prompt, use the whole sequence
                                # but this should be the same length as the original segment
                                segment_part = new_sequence
                            
                            # For mutation-based generators, the segment part should be the same length
                            # as the original segment sequence
                            if len(segment_part) == len(segment_current_sequence):
                                final_extended_sequence = segment_part
                            else:
                                # Fallback: use the segment part as is
                                final_extended_sequence = segment_part
                        
                        # Record this generation step
                        self._record_generation_step(
                            temp_sequence._metadata, generator, generator_idx, 
                            current_prompt, new_sequence, temp_segment
                        )
                        
                        # For the next generator, update the prompt to include this generator's output
                        if generator_idx < len(segment_generators) - 1:
                            current_prompt = new_sequence
                
                final_candidate = self._create_final_candidate(
                    final_extended_sequence, segment.sequence_type, temp_sequence._metadata
                )
                candidates.append(final_candidate)
        
        return candidates
    
    def _apply_generator(self, generator: Generator, temp_segment: ConstructSegment, temp_sequence: Sequence, 
                        accepts_prompts: bool) -> Optional[str]:
        """Apply a single generator and return the new sequence."""
        # Temporarily assign the generator to our temp segment
        original_generator_output = generator._generator_output
        generator._generator_output = temp_segment
        
        try:
            # Apply the generator
            if accepts_prompts:
                # Always pass the current sequence as prompt if the generator accepts prompts
                generator.sample(prompt_seqs=[temp_sequence.sequence])
            else:
                generator.sample()
            
            # Get the generated output
            outputs = generator.get_generator_outputs()
            if outputs and outputs[0] and outputs[0][0]:
                return outputs[0][0].sequence
            
            return None
            
        finally:
            # Restore the original generator assignment
            generator._generator_output = original_generator_output
    

    
    def _record_generation_step(self, metadata: Dict[str, Any], generator: Generator, generator_idx: int,
                              current_sequence: str, new_sequence: str, temp_segment: ConstructSegment) -> None:
        """Record metadata about a generation step."""
        step_metadata = {
            "generator_type": type(generator).__name__,
            "step_idx": generator_idx,
            "input_sequence": current_sequence,
            "output_sequence": new_sequence,
            "generator_metadata": temp_segment.batch_sequences[0]._metadata
        }
        metadata["generation_steps"].append(step_metadata)
    
    def _create_final_candidate(self, final_sequence: str, sequence_type: SequenceType, 
                              metadata: Dict[str, Any]) -> Sequence:
        """Create the final candidate sequence with metadata."""
        return Sequence(
            sequence=final_sequence,
            sequence_type=sequence_type,
            metadata=metadata.copy()
        )
    
    def _evaluate_concatenated_combinations(self, combinations: List[Dict[int, Sequence]], construct: Construct, current_segment_idx: int) -> List[Tuple[Dict[int, Sequence], float]]:
        """
        Evaluates concatenated sequences from all segments up to current one.
        
        A "combination" in beam search represents a complete set of sequence choices across
        multiple segments. Each combination is a dictionary where:
        - Keys are segment indices (int)
        - Values are Sequence objects representing the chosen sequence for that segment
        
        For example, a combination might look like:
        {0: Sequence("ATGCTAGCTA"), 1: Sequence("GCTAGCTAGC"), 2: Sequence("TAGCTAGCTA")}
        This represents choosing "ATGCTAGCTA" for segment 0, "GCTAGCTAGC" for segment 1, etc.
        
        The method concatenates these sequences in order and evaluates them against constraints
        to compute an overall energy score (lower is better).
        
        Args:
            combinations: List of combinations to evaluate. Each combination is a Dict[int, Sequence]
                         mapping segment indices to their chosen sequences.
            construct: The construct containing all segments
            current_segment_idx: Index of the current segment being processed
            
        Returns:
            List of tuples (combination, energy_score) for all evaluated combinations,
            where energy_score is the total constraint violation score (lower is better)
        """
        evaluated_combinations = []
        
        for combination in combinations:
            if current_segment_idx == 0:
                # For first segment, evaluate just that segment's candidate
                concatenated_sequence = combination[current_segment_idx].sequence
            else:
                # For subsequent segments, create concatenated sequence for evaluation
                # Only include segments that have been processed so far (up to current_segment_idx)
                concatenated_sequence = ""
                for seg_idx in range(current_segment_idx + 1):
                    if seg_idx in combination:
                        concatenated_sequence += combination[seg_idx].sequence
                    else:
                        concatenated_sequence += ""  # Empty for missing segments
            
            # Evaluate constraints on the concatenated sequence
            total_energy = 0.0
            for constraint in self.constraints:
                # Create a temporary sequence object for evaluation
                temp_sequence = Sequence(
                    sequence=concatenated_sequence,
                    sequence_type=SequenceType.DNA,  # Assume DNA for concatenated evaluation
                    metadata={"concatenated": True}
                )
                
                energy = constraint.scoring_function(temp_sequence, **constraint.scoring_function_config)
                total_energy += energy
            
            # Add evaluation metadata to each candidate
            for seg_idx, candidate in combination.items():
                candidate._metadata["evaluation_energy"] = total_energy
            
            evaluated_combinations.append((combination, total_energy))
        
        if self.verbose:
            print(f"Evaluated {len(evaluated_combinations)} concatenated combinations")
        
        return evaluated_combinations
    

    
    def _select_top_combinations(self, evaluated_combinations: List[Tuple[Dict[int, Sequence], float]]) -> List[Tuple[Dict[int, Sequence], float]]:
        """
        Select the top-K combinations based on energy scores.
        
        This method implements the core beam search selection mechanism. It takes the
        evaluated combinations (each with their computed energy score) and selects
        the top K combinations to maintain in the beam. The beam width (K) determines
        how many promising sequence combinations are kept for the next iteration.
        
        Args:
            evaluated_combinations: List of (combination, energy) tuples where:
                - combination: Dict[int, Sequence] mapping segment indices to sequences
                - energy: float representing the total constraint violation score
                
        Returns:
            Top-K combinations sorted by energy (lower is better), maintaining the beam
            for the next segment's optimization
        """
        # Use heapq.nlargest for efficient top-K selection (lower energy is better, so we negate)
        # Note: heapq.nlargest returns items in descending order, so we negate energy scores
        # to get the lowest energy scores first
        top_combinations = heapq.nlargest(
            self.beam_width, 
            evaluated_combinations, 
            key=lambda x: -x[1]  # Negate energy so lower values come first
        )
        
        if self.verbose:
            best_energy = top_combinations[0][1] if top_combinations else float('inf')
            print(f"Selected top {len(top_combinations)} combinations, best energy: {best_energy:.4f}")
        
        return top_combinations
    
    def _update_segments_with_combinations(self, construct: Construct, top_combinations: List[Tuple[Dict[int, Sequence], float]]) -> None:
        """
        Update all segments with the best combinations from beam search.
        
        This method distributes the top-K combinations across the beam structure.
        Each combination represents a complete set of sequence choices across all
        segments processed so far. The method updates each segment's batch_sequences
        to maintain the beam candidates for the next iteration.
        
        The beam structure ensures that:
        - Each segment maintains exactly beam_width sequences
        - Sequences are organized by their beam index (0 to beam_width-1)
        - Each beam represents a different promising path through the sequence space
        
        Args:
            construct: The construct to update with new beam candidates
            top_combinations: Top-K combinations to distribute, where each combination is:
                - Dict[int, Sequence]: mapping segment indices to their chosen sequences
                - float: the energy score for this combination (lower is better)
        """
        segments = construct.segments
        
        # Initialize beam candidates for each segment
        for segment in segments:
            while len(segment.batch_sequences) < self.beam_width:
                new_sequence = Sequence(sequence="", sequence_type=segment.sequence_type, metadata={})
                segment.batch_sequences.append(new_sequence)
        
        # Distribute combinations across beam width
        for beam_idx, (combination, energy) in enumerate(top_combinations):
            if beam_idx >= self.beam_width:
                break
                
            for seg_idx, segment in enumerate(segments):
                if seg_idx in combination:
                    candidate = combination[seg_idx]
                    existing_seq = segment.batch_sequences[beam_idx]
                    
                    # Update the sequence
                    existing_seq.sequence = candidate.sequence
                    existing_seq._metadata.clear()
                    existing_seq._metadata.update(candidate._metadata)
                    existing_seq._metadata["energy"] = energy
                    existing_seq._metadata["beam_idx"] = beam_idx
        
        # Trim excess sequences if we have more than beam_width
        for segment in segments:
            if len(segment.batch_sequences) > self.beam_width:
                segment.batch_sequences = segment.batch_sequences[:self.beam_width]
    

    

    
    def sample(self) -> List[Construct]:
        """
        Run sequential beam search across all segments with context accumulation.
        
        This method implements the core beam search algorithm by processing segments
        sequentially and maintaining beams of the most promising sequence combinations.
        
        **Beam Search Process:**
        1. For each segment, use accumulated sequences from previous segments as prompts
        2. Generate candidates for the current segment using these prompts
        3. Create combinations by pairing new candidates with sequences from previous segments
        4. Evaluate constraints on concatenated sequences to compute energy scores
        5. Select top-K combinations (lowest energy) to maintain in the beam
        6. Use these combinations as prompts for the next segment
        7. Continue until all segments are processed
        
        **Combination Management:**
        - Combinations are dictionaries mapping segment indices to Sequence objects
        - Each combination represents a complete path through the sequence space
        - The beam maintains K such combinations, where K = beam_width
        - Combinations are evaluated by concatenating sequences and applying constraints
        - Only the best K combinations (lowest energy scores) are kept for the next iteration
        
        Returns:
            List containing the construct with updated beam candidates, where each
            segment maintains exactly beam_width sequences representing the best
            combinations found during the search.
        """
        for construct in self.constructs:
            if self.verbose:
                print(f"Processing {len(construct.segments)} segments with sequential beam search")
            
            # Initialize beam candidates for the first segment
            # Use existing beam candidates from previous sample if available, otherwise start with empty sequences
            if hasattr(self, '_beam_candidates') and self._beam_candidates:
                beam_candidates = self._beam_candidates.copy()
                if self.verbose:
                    print(f"Using existing beam candidates from previous sample")
            else:
                beam_candidates = [""] * self.beam_width
                if self.verbose:
                    print(f"Starting with empty beam candidates")
            
            # Process each segment sequentially
            for segment_idx, segment in enumerate(construct.segments):
                if self.verbose:
                    print(f"\n--- Processing Segment {segment_idx + 1}/{len(construct.segments)} ---")
                    print(f"Using {len(beam_candidates)} beam candidates as prompts")
                
                # Step 1: Generate candidates for current segment using beam prompts
                # For each beam candidate, extract only the sequences from segments that have been processed so far
                current_beam_prompts = []
                for beam_idx, beam_sequence in enumerate(beam_candidates):
                    # Extract only the sequences from segments processed so far in this sample
                    current_pos = 0
                    accumulated_prompt = ""
                    
                    for seg_idx in range(segment_idx):
                        prev_segment = construct.segments[seg_idx]
                        
                        segment_length = self._get_segment_length(prev_segment)
                        
                        # Extract this segment's contribution from the beam sequence
                        if current_pos < len(beam_sequence):
                            segment_contribution = beam_sequence[current_pos:current_pos + segment_length]
                            current_pos += segment_length
                            accumulated_prompt += segment_contribution
                        else:
                            break
                    
                    current_beam_prompts.append(accumulated_prompt)
                
                segment_candidates = self._generate_candidates_for_segment_with_prompts(segment, current_beam_prompts)
                
                if self.verbose:
                    print(f"Generated {len(segment_candidates)} candidates for segment {segment_idx + 1}")
                
                # Step 2: Create combinations with previous segments
                # A combination is a Dict[int, Sequence] mapping segment indices to their chosen sequences
                # For the first segment, just use the candidates directly
                if segment_idx == 0:
                    # Create combinations where each candidate is paired with empty previous segments
                    # Each combination represents a single candidate for segment 0
                    combinations = []
                    for candidate in segment_candidates:
                        combination = {segment_idx: candidate}
                        combinations.append(combination)
                else:
                    # Create combinations with all previous segments
                    # Each combination will contain sequences from segments 0 to segment_idx
                    previous_segments = construct.segments[:segment_idx]
                    
                    # Create combinations by pairing current candidates with previous beam candidates
                    # IMPORTANT: Each beam must maintain continuity - candidates from different beams cannot be mixed
                    # This ensures that combinations represent coherent paths through the sequence space
                    combinations = []
                    
                    # Group candidates by their beam (prompt_idx)
                    candidates_by_beam = {}
                    for candidate in segment_candidates:
                        prompt_idx = candidate._metadata.get("prompt_idx", 0)
                        if prompt_idx not in candidates_by_beam:
                            candidates_by_beam[prompt_idx] = []
                        candidates_by_beam[prompt_idx].append(candidate)
                    
                    # For each beam, create combinations only with candidates from the same beam
                    for prompt_idx, beam_prompt in enumerate(beam_candidates):
                        if prompt_idx in candidates_by_beam:
                            for candidate in candidates_by_beam[prompt_idx]:
                                combination = {segment_idx: candidate}
                                # For previous segments, we need to extract their individual contributions
                                # The beam_prompt contains the full sequence, but we need to split it
                                # into individual segment contributions
                                current_pos = 0
                                for prev_seg_idx, prev_segment in enumerate(previous_segments):
                                    segment_length = self._get_segment_length(prev_segment)
                                    
                                    # Extract this segment's contribution from the beam sequence
                                    if current_pos < len(beam_prompt):
                                        segment_contribution = beam_prompt[current_pos:current_pos + segment_length]
                                        current_pos += segment_length
                                    else:
                                        segment_contribution = ""
                                    
                                    # Create a sequence object for this segment's contribution
                                    prev_sequence = Sequence(
                                        sequence=segment_contribution,
                                        sequence_type=prev_segment.sequence_type,
                                        metadata={"beam": True, "prompt_idx": prompt_idx}
                                    )
                                    combination[prev_seg_idx] = prev_sequence
                                combinations.append(combination)
                
                # Step 3: Evaluate combinations with constraints
                # For all segments, evaluate constraints on the concatenated sequences
                evaluated_combinations = self._evaluate_concatenated_combinations(combinations, construct, segment_idx)
                
                # Step 4: Select top-K combinations
                top_combinations = self._select_top_combinations(evaluated_combinations)
                
                # Unified logging for all segments
                if self.verbose:
                    print(f"Segment {segment_idx + 1} Candidates:")
                    existing_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""
                    print(f"    Existing sequence: '{existing_sequence}'")
                    
                    # Group candidates by beam (prompt_idx)
                    candidates_by_beam = {}
                    for candidate in segment_candidates:
                        prompt_idx = candidate._metadata.get("prompt_idx", 0)
                        if prompt_idx not in candidates_by_beam:
                            candidates_by_beam[prompt_idx] = []
                        candidates_by_beam[prompt_idx].append(candidate)
                    
                    # Show candidates grouped by beam
                    for prompt_idx in sorted(candidates_by_beam.keys()):
                        beam_candidates = candidates_by_beam[prompt_idx]
                        
                        # Get the prompt used for this beam
                        if segment_idx == 0:
                            # For first segment, just show the existing sequence
                            formatted_prompt = existing_sequence
                        else:
                            # For subsequent segments, show the accumulated sequence
                            if prompt_idx < len(current_beam_prompts):
                                beam_prompt = current_beam_prompts[prompt_idx]
                                # Use the same sequence as the generator prompt (no separators)
                                formatted_prompt = beam_prompt + existing_sequence
                            else:
                                formatted_prompt = "unknown"
                        
                        print(f"    Candidates generated from '{formatted_prompt}' (Beam {prompt_idx}):")
                        
                        for candidate_idx, candidate in enumerate(beam_candidates):
                            # Check if this candidate is in the top combinations and get its energy
                            is_selected = False
                            energy = candidate._metadata.get("evaluation_energy", 0.0)  # Get energy from candidate metadata
                            
                            # Check if this candidate is selected (in top combinations)
                            for combination, combo_energy in top_combinations:
                                if segment_idx in combination and combination[segment_idx].sequence == candidate.sequence:
                                    is_selected = True
                                    break
                            
                            status = "✓ SELECTED" if is_selected else "✗ REJECTED"
                            # Show the full segment sequence (not just the extension)
                            final_sequence = candidate._metadata.get("final_sequence", candidate.sequence)
                            print(f"        {status} '{final_sequence}' (Energy: {energy:.4f}, Candidate #{candidate_idx})")
                            
                            # Show generator steps
                            generation_steps = candidate._metadata.get("generation_steps", [])
                            for step_idx, step in enumerate(generation_steps):
                                generator_num = step_idx + 1
                                input_seq = step.get("input_sequence", "")
                                output_seq = step.get("output_sequence", "")
                                
                                # Calculate extension
                                if output_seq.startswith(input_seq):
                                    extension = output_seq[len(input_seq):]
                                else:
                                    extension = output_seq
                                
                                # Show the actual sequences that the generators work with (no separators)
                                print(f"            Generator {generator_num}: prompt='{input_seq}' -> generated '{output_seq}' -> extension '{extension}'")
                    print()
                
                # Step 5: Update beam candidates for next segment
                beam_candidates = []
                for combination, energy in top_combinations:
                    if segment_idx == 0:
                        # For first segment, just use the candidate sequence
                        concatenated_sequence = combination[segment_idx].sequence
                    else:
                        # For subsequent segments, build the full accumulated sequence
                        # by concatenating all segment contributions in order
                        concatenated_sequence = ""
                        for seg_idx in range(segment_idx + 1):
                            if seg_idx in combination:
                                concatenated_sequence += combination[seg_idx].sequence
                    
                    beam_candidates.append(concatenated_sequence)
                
                if self.verbose:
                    print(f"Updated accumulated sequences for next segment")
                    print(f"Best energy: {top_combinations[0][1] if top_combinations else float('inf'):.4f}")
                    print(f"Top {len(beam_candidates)} beam sequences:")
                    for beam_idx, sequence in enumerate(beam_candidates):
                        # Show the full accumulated sequence (no separators for now)
                        print(f"  Beam {beam_idx}: '{sequence}'")
            
            # Final step: Update all segments with their individual extended sequences
            # Each segment should contain its own extended sequence, not the full accumulated sequence
            final_combinations = []
            for acc_idx, beam_sequence in enumerate(beam_candidates):
                combination = {}
                
                # For sequential beam search, each segment gets its own extended sequence
                # We need to extract each segment's contribution from the accumulated sequence
                current_pos = 0
                for seg_idx, segment in enumerate(construct.segments):
                    segment_length = self._get_segment_length(segment)
                    
                    # Extract this segment's contribution from the beam sequence
                    if current_pos < len(beam_sequence):
                        segment_sequence = beam_sequence[current_pos:current_pos + segment_length]
                        current_pos += segment_length
                    else:
                        segment_sequence = ""
                    
                    seq_obj = Sequence(
                        sequence=segment_sequence,
                        sequence_type=segment.sequence_type,
                        metadata={"combination_idx": acc_idx, "segment_idx": seg_idx}
                    )
                    combination[seg_idx] = seq_obj
                
                final_combinations.append((combination, 0.0))  # Energy already evaluated
            
            # Update segments with final combinations
            self._update_segments_with_combinations(construct, final_combinations)
            
            # Store beam candidates for the next sample
            self._beam_candidates = beam_candidates.copy()
        
        # Add final state to history
        self.history.append(copy.deepcopy(self.constructs))
        
        if self.verbose:
            self._log_progress()
        
        return self.constructs
    
    def _log_progress(self) -> None:
        """Log current sequential beam search progress."""
        total_candidates = 0
        total_energy = 0.0
        
        for construct in self.constructs:
            for segment in construct.segments:
                total_candidates += len(segment.batch_sequences)
                for sequence in segment.batch_sequences:
                    energy = sequence._metadata.get("energy", 0.0)
                    total_energy += energy
        
        avg_energy = total_energy / total_candidates if total_candidates > 0 else 0.0
        print(f"BeamSearchGenerator: {total_candidates} total candidates, avg energy: {avg_energy:.4f}")
    
    def _is_extension_based_generator(self, generator) -> bool:
        """
        Determine if a generator is extension-based or mutation-based.
        
        Args:
            generator: The generator to check
            
        Returns:
            True if the generator is extension-based, False if mutation-based
        """
        # Extension-based generators have prepend_prompt attribute
        # Mutation-based generators don't have this attribute
        return hasattr(generator, 'prepend_prompt') and generator.prepend_prompt

    def _populate_caches(self) -> None:
        """Populate caches for expensive operations to avoid repeated calculations."""
        # Cache segment generators mapping
        for segment in self.constructs[0].segments:
            self._segment_generators_map[segment] = [
                gen for gen in self.generators if gen._generator_output == segment
            ]
        
    def _get_segment_generators(self, segment: ConstructSegment) -> List[Generator]:
        """Get cached generators for a segment."""
        return self._segment_generators_map.get(segment, [])
    
    def _get_segment_length(self, segment: ConstructSegment) -> int:
        """Calculate the total length a segment contributes to the accumulated sequence."""
        initial_sequence = segment.batch_sequences[0].sequence if segment.batch_sequences else ""
        initial_length = len(initial_sequence)
        
        segment_generators = self._get_segment_generators(segment)
        
        # Calculate the total contribution from generators
        total_generated_length = 0
        for generator in segment_generators:
            if self._is_extension_based_generator(generator):
                if hasattr(generator, 'sequence_length'):
                    total_generated_length += generator.sequence_length
                elif hasattr(generator, 'n_tokens'):
                    total_generated_length += generator.n_tokens
        
        return initial_length + total_generated_length
