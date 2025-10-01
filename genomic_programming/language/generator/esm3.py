"""
Esm3 Generator

Extracted from generator.py for better code organization.
"""

from typing import List, final
import torch

from ..base import Generator, Segment


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
        >>> segment = Segment(sequence="", sequence_type=SequenceType.PROTEIN)
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
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Creates initial sequences by running ESM3 on sequences of mask tokens
        and sampling amino acids from the resulting probability distributions.
        If the segment already contains sequences, they will be used as starting points.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Raises:
            ValueError: If assigned_segments is not a single Segment object.
            AssertionError: If provided sequence length doesn't match configured length.
        """
        # Validate that we received a single Segment, not a list or other type
        if not isinstance(assigned_segments, Segment):
            raise ValueError(
                f"ESM3Generator.assign() expects a single Segment object, "
                f"got {type(assigned_segments).__name__}. If you have multiple segments, "
                f"assign them to separate generator instances."
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
        from ...utils import use_cloud_gpu

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

