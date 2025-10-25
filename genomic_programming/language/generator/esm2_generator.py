"""
ESM2 Generator for protein sequence generation
"""

from typing import final
from pydantic import Field, field_validator

from ..core import Generator, Segment
from proto_language.base_config import BaseConfig
from proto_language.tools.models.language_models.esm2.esm2 import run_esm2_sample, ESM2SampleConfig, LanguageModelInput
from .generator_registry import GeneratorRegistry


class ESM2GeneratorConfig(BaseConfig):
    """Configuration for ESM2Generator."""
    # Required parameters
    sequence_length: int = Field(ge=1, description="Length of protein sequences to generate")

    # Optional parameters (have defaults)
    esm2_type: str = Field(default="esm2_t33_650M_UR50D", description="ESM2 model variant")
    temperature: float = Field(default=1.0, gt=0.0, description="Sampling temperature")
    decoding_method: str = Field(default="entropy", description="Position selection strategy: 'entropy', 'max_logit', or 'random'")
    top_k: int = Field(default=5, ge=1, description="Number of positions to sample per iteration")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        if 'sequence_length' in info.data and v > info.data['sequence_length']:
            raise ValueError(f"top_k ({v}) cannot exceed sequence_length ({info.data['sequence_length']})")
        return v


@GeneratorRegistry.register(
    key="esm2",
    label="ESM2 Protein Language Model",
    config=ESM2GeneratorConfig,
    description="ESM-2 protein language model for protein sequence generation",
    category="language_model",
    requires_gpu=True,
    autoregressive=False,
)
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
        >>> from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig
        >>> config = ESM2GeneratorConfig(
        ...     esm2_type="esm2_t33_650M_UR50D",
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     top_k=5
        ... )
        >>> gen = ESM2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen.assign(segment)  # Creates random initial sequences from mask tokens
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(self, config: ESM2GeneratorConfig) -> None:
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.esm2_type = config.esm2_type
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        self.decoding_method = config.decoding_method
        self.top_k = config.top_k
        self.autoregressive = False

    def assign(
        self, assigned_segment: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        - If starting sequence is provided, validates that the sequence length matches the configured length.
        """
        super().assign(assigned_segment)
        self._assigned_segment = assigned_segment
        self._assigned_segment._is_assigned = True

    def sample(self) -> None:
        """
        Sample new amino acids at selected high-uncertainty positions for all sequences in the batch.

        For each sequence in the batch, uses the current sequence to compute ESM-2 logits,
        selects top-k positions based on the decoding method, and samples new amino acids
        at those positions.

        Raises:
            RuntimeError: If called before assign().
        """
        # Create input and config objects
        sequences = [seq.sequence for seq in self._assigned_segment.candidate_sequences]
        esm2_input = LanguageModelInput(sequences=sequences)
        config = ESM2SampleConfig(
            model_name=self.esm2_type,
            sequence_length=self.sequence_length,
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            top_k=self.top_k,
            keep_on_device=True,  # Keep for repeated calls
            verbose=False
        )
        result = run_esm2_sample(inputs=esm2_input, config=config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._assigned_segment.candidate_sequences[i].sequence = sequence
