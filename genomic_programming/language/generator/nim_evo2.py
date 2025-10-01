"""
Nim Evo2 Generator

Extracted from generator.py for better code organization.
"""

from typing import Optional, List, final
import os
import json

import requests

from ..base import Generator, Segment


@final
class NimEvo2Generator(Generator):
    """
    A sequence generator that uses the Nvidia NIM Evo2 API for DNA sequence generation.

    Users must provide a NVIDIA API key for authentication or set it in the NV_API_KEY environment variable.

    Examples:
        >>> segment = Segment(sequence="", sequence_type=SequenceType.DNA)
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
            if len(prompt_seqs) != batch_size:
                raise ValueError(
                    f"Multiple prompts ({len(prompt_seqs)}) must equal batch_size ({batch_size})"
                )
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
        self, assigned_segments: Segment
    ) -> None:
        """
        Assign a Segment to this generator.

        Args:
            assigned_segments: A single Segment to be assigned to this generator.

        Warning:
            Any existing sequences in the assigned segment will be overwritten when sample()
            is called, as NIM Evo2 performs autoregressive generation from prompt sequences.
        """
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
        or the default prompt sequences, updating the sequences in the Segment in-place.

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

        # Update sequences in the Segment
        for idx, sequence in enumerate(generated_sequences):
            if self.prepend_prompt:
                sequence = prompts[idx] + sequence
            self._generator_output.batch_sequences[idx].sequence = sequence

