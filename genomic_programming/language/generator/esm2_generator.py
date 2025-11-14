"""
ESM2 Generator for protein sequence generation
"""
from __future__ import annotations
from typing import final, Literal
from pydantic import field_validator


from proto_language.language.core import Generator, GeneratorType, Segment
from proto_language.base_config import BaseConfig, ConfigField
from proto_language.tools.language_models.esm2.esm2 import run_esm2_sample, ESM2SampleConfig, LanguageModelInput
from proto_language.tools.language_models.esm2.inference import ESM2_MODEL_CHECKPOINTS
from proto_language.language.generator.generator_registry import GeneratorRegistry


class ESM2GeneratorConfig(BaseConfig):
    """Configuration object for ESM2Generator.

    This class defines configuration parameters for the ESM2 generator, which uses
    a protein language model to generate and refine protein sequences through
    iterative mutation of high-uncertainty positions.

    Attributes:
        sequence_length (int): Target length for generated protein sequences in
            amino acids. All sequences must match this length. Must be at least 1.

        model_checkpoint (str): ESM2 model checkpoint to use. Options:

            - ``"esm2_t6_8M_UR50D"``: 8M parameters, 6 layers (fastest)
            - ``"esm2_t12_35M_UR50D"``: 35M parameters, 12 layers
            - ``"esm2_t30_150M_UR50D"``: 150M parameters, 30 layers
            - ``"esm2_t33_650M_UR50D"``: 650M parameters, 33 layers (default, balanced)
            - ``"esm2_t36_3B_UR50D"``: 3B parameters, 36 layers
            - ``"esm2_t48_15B_UR50D"``: 15B parameters, 48 layers (best quality)

            Default: ``"esm2_t33_650M_UR50D"``.

        temperature (float): Scales randomness of amino acid sampling by adjusting
            probability distribution sharpness:

            - ``< 1.0``: More deterministic, favors high-probability amino acids
            - ``1.0``: Standard sampling from model distribution (default)
            - ``> 1.0``: More diverse, explores lower-probability amino acids

            Must be greater than 0. Default: 1.0.

        decoding_method (str): Strategy for selecting which positions to mutate:

            - ``"entropy"``: Select positions with highest prediction uncertainty (default)
            - ``"max_logit"``: Select positions with lowest confidence predictions
            - ``"random"``: Randomly select positions to mutate

            ``"entropy"`` typically produces the most natural-looking proteins.
            Default: ``"entropy"``.

        num_mutations (int): Number of positions to mutate per sampling iteration.
            Higher values explore more of sequence space but may reduce biological
            plausibility. Must be at least 1 and cannot exceed ``sequence_length``.
            Default: 1.

    Note:
        For bidirectional models like ESM2, the ``sequence_length`` parameter should
        ideally be determined from input sequences rather than configured manually
        (planned for future versions). TODO
    """
    # Required parameters
    # TODO: For bidirectional model sampling, this should probably not be configurable/ should be based on input sequences
    sequence_length: int = ConfigField(
        ge=1, title="Sequence Length", description="Target length for generated sequences"
    )

    model_checkpoint: ESM2_MODEL_CHECKPOINTS = ConfigField(
        default="esm2_t33_650M_UR50D",
        title="Model Checkpoint",
        description="ESM2 model checkpoint to use",
    )

    # Advanced parameters
    temperature: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="Temperature",
        description="Scales the randomness of sampling by adjusting probability distribution sharpness.",  # Lower values (<1) make outputs more deterministic; higher values (>1) produce more varied and creative generations.
        advanced=True,
    )
    decoding_method: Literal["entropy", "max_logit", "random"] = ConfigField(
        default="entropy",
        title="Decoding Method",
        description="Position selection strategy for sampling: entropy, max_logit, or random",
        advanced=True,
    )
    num_mutations: int = ConfigField(
        default=1,
        ge=1,
        title="Num Mutations",
        description="Number of positions to mutate per sampling iteration",
        advanced=True,
    )

    @field_validator('num_mutations')
    @classmethod
    def validate_num_mutations(cls, v, info):
        if 'sequence_length' in info.data and v > info.data['sequence_length']:
            raise ValueError(f"num_mutations ({v}) cannot exceed sequence_length ({info.data['sequence_length']})")
        return v


@GeneratorRegistry.register(
    key="esm2",
    label="ESM2 Protein Language Model",
    config=ESM2GeneratorConfig,
    description="ESM-2 protein language model for protein sequence generation",
    type=GeneratorType.MUTATION,
    requires_gpu=True,
)
@final
class ESM2Generator(Generator):
    """Protein sequence generator using ESM2 language model.

    This generator uses the ESM2 protein language model to generate and refine
    protein sequences through iterative mutation. It identifies high-uncertainty
    positions based on model confidence and samples biologically plausible amino
    acids at those positions.

    The generator type is ``GeneratorType.MUTATION``, indicating it refines sequences
    through targeted mutations rather than generating from scratch.

    Attributes:
        model_checkpoint (str): ESM2 model checkpoint name.
        sequence_length (int): Length of sequences to generate/mutate.
        temperature (float): Sampling temperature for diversity control.
        decoding_method (str): Position selection strategy (entropy/max_logit/random).
        num_mutations (int): Number of positions to mutate per iteration.
        type (GeneratorType): Set to ``GeneratorType.MUTATION``.

    Example:
        >>> from proto_language.language.generator import ESM2Generator, ESM2GeneratorConfig
        >>> from proto_language.language.core import Segment, SequenceType
        >>> config = ESM2GeneratorConfig(
        ...     sequence_length=100,
        ...     temperature=1.0,
        ...     decoding_method="entropy",
        ...     num_mutations=5
        ... )
        >>> gen = ESM2Generator(config)
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
        >>> gen.assign(segment)
        >>> gen.sample()  # Refines 5 highest-entropy positions
    """

    def __init__(self, config: ESM2GeneratorConfig) -> None:
        """
        Initialize the ESM-2 generator with model and sampling configuration.

        Args:
            config: Configuration object containing all generator parameters.
        """
        super().__init__()
        self.model_checkpoint = config.model_checkpoint
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        self.decoding_method = config.decoding_method
        self.num_mutations = config.num_mutations
        self.type = GeneratorType.MUTATION

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
            model_checkpoint=self.model_checkpoint,
            sequence_length=self.sequence_length,   
            temperature=self.temperature,
            decoding_method=self.decoding_method,
            num_mutations=self.num_mutations,
            keep_on_gpu=True,  # Keep for repeated calls
            verbose=False
        )
        result = run_esm2_sample(inputs=esm2_input, config=config)
        mutated_sequences = result.sequences

        # Update sequences in the batch
        for i, sequence in enumerate(mutated_sequences):
            self._assigned_segment.candidate_sequences[i].sequence = sequence
