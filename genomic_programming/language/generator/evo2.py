"""
Evo2 Generator

Extracted from generator.py for better code organization.
"""

from typing import Any, List, Optional, Dict, final

from pydantic import Field, field_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from .generator_registry import GeneratorRegistry


class Evo2GeneratorConfig(BaseConfig):
    """Configuration for Evo2Generator."""
    prompt_seqs: List[str] = Field(description="Prompt sequences for generation (1 or batch_size prompts)")
    evo2_type: str = Field(default="evo2_7b", description="Evo2 model variant to use")
    evo2_local_path: Optional[str] = Field(default=None, description="Optional path to local model weights")
    sequence_length: int = Field(default=500, ge=1, description="Number of tokens to generate after prompt")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    top_k: int = Field(default=4, ge=1, description="Top-k sampling parameter")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    batched: bool = Field(default=True, description="Whether to use batched generation")
    cached_generation: bool = Field(default=True, description="Whether to cache model states")
    verbose: int = Field(default=1, ge=0, description="Verbosity level for logging")
    force_prompt_threshold: Optional[int] = Field(default=None, description="Optional threshold for forcing prompt continuation")
    batch_size: int = Field(default=1, ge=1, description="Number of sequences to generate")
    prepend_prompt: bool = Field(default=False, description="Whether to prepend prompt to generated sequences")
    sampling_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional sampling arguments")
    
    @field_validator('prompt_seqs')
    @classmethod
    def validate_prompt_seqs(cls, v):
        if not v:
            raise ValueError("prompt_seqs must not be empty")
        return v


@GeneratorRegistry.register(
    key="evo2",
    config=Evo2GeneratorConfig,
    description="Evo2 genome language model for DNA sequence generation",
    category="language_model",
    requires_gpu=True,
    supports_batch=True
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
        >>> from proto_language.language.generator import Evo2Generator, Evo2GeneratorConfig
        >>> config = Evo2GeneratorConfig(
        ...     prompt_seqs=["+~GA"],
        ...     evo2_type="evo2_7b",
        ...     sequence_length=1000,
        ...     temperature=0.8,
        ...     batch_size=5
        ... )
        >>> gen = Evo2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.DNA)
        >>> gen.assign(segment)
        >>> gen.sample()  # Generates sequences from prompts

        Custom model with local weights:
        >>> config = Evo2GeneratorConfig(
        ...     prompt_seqs=["+~GA", "+~GC"],
        ...     evo2_type="evo2_7b_phage",
        ...     evo2_local_path="/path/to/weights.pt",
        ...     batch_size=2
        ... )
        >>> gen = Evo2Generator(config)
        >>> gen.assign(segment)
        >>> gen.sample()  # Uses local model weights
    """

    def __init__(self, config: Evo2GeneratorConfig) -> None:
        """
        Initialize the Evo2 generator with model configuration and sampling parameters.

        For detailed documentation of Evo2 sampling parameters, refer to:
        https://github.com/arcinstitute/evo2 and https://github.com/Zymrael/vortex

        Args:
            config: Configuration object containing all generator parameters.

        Note:
            Model instances are automatically shared between generators with the same
            evo2_type, evo2_local_path, and sampling_kwargs to save memory and initialization time.
        """
        super().__init__(batch_size=config.batch_size)
        self.config = config

        # Handle batch_size: replicate single prompt or validate multiple prompts
        if len(config.prompt_seqs) == 1:
            self.prompt_seqs = config.prompt_seqs * config.batch_size
        else:
            if len(config.prompt_seqs) != config.batch_size:
                raise ValueError(
                    f"Multiple prompts ({len(config.prompt_seqs)}) must equal batch_size ({config.batch_size})"
                )
            if len(set(len(seq) for seq in config.prompt_seqs)) != 1:
                raise ValueError(
                    f"All prompts must have same length, got: {[len(seq) for seq in config.prompt_seqs]}"
                )
            self.prompt_seqs = config.prompt_seqs

        self.batch_size = config.batch_size
        self.evo2_type = config.evo2_type
        self.evo2_local_path = config.evo2_local_path
        self.n_tokens = config.sequence_length
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.batched = config.batched
        self.cached_generation = config.cached_generation
        self.verbose = config.verbose
        self.force_prompt_threshold = config.force_prompt_threshold
        self.prepend_prompt = config.prepend_prompt
        self.sampling_kwargs = config.sampling_kwargs

    def assign(
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segments is not a single Segment object.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as Evo2 performs autoregressive generation from prompt sequences.
        """
        # Validate that we received a single Segment, not a list or other type
        if not isinstance(assigned_segments, Segment):
            raise ValueError(
                f"Evo2Generator.assign() expects a single Segment object, "
                f"got {type(assigned_segments).__name__}. If you have multiple segments, "
                f"assign them to separate generator instances."
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
        or the default prompt sequences, updating the sequences in the Segment in-place.

        Args:
            prompt_seqs: Optional list of prompt sequences to use instead of self.prompt_seqs.
                        Useful for chaining generators where each uses the output of the previous.

        Raises:
            RuntimeError: If called before assign().
        """
        self._validate_generator()

        # Use provided prompts or fall back to the default prompt
        prompts = prompt_seqs if prompt_seqs is not None else self.prompt_seqs

        # Use the evo2 sampling tool
        from proto_language.tools.models.language_models.evo2 import (
            run_evo2_sample,
            Evo2SampleConfig,
        )

        # Create config for the tool
        sample_config = Evo2SampleConfig(
            prompt_seqs=prompts,
            model_name=self.evo2_type,
            local_path=self.evo2_local_path,
            sequence_length=self.n_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            batched=self.batched,
            cached_generation=self.cached_generation,
            verbose=self.verbose,
            force_prompt_threshold=self.force_prompt_threshold,
            prepend_prompt=self.prepend_prompt,
            sampling_kwargs=self.sampling_kwargs,
        )

        # Run the sampling tool
        result = run_evo2_sample(sample_config)
        generated_sequences = result.sequences

        # Update sequences in the Segment
        for idx, sequence in enumerate(generated_sequences):
            self._generator_output.batch_sequences[idx].sequence = sequence

