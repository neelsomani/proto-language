"""
Promoter strength constraint using Barrick Lab Promoter Calculator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from promoter_calculator.wrapper import promoter_calculator

from ...base import Sequence


def promoter_strength_constraint(
    sequences: Union["Sequence", List["Sequence"]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[float, List[float]]:
    """
    Run Barrick Lab Promoter Calculator and return a [0,1] penalty score.

    Also caches the full promoter_calculator output in each Sequence's metadata
    under the "promoter_strength" key.

    Penalty scheme:
        For tx_rate penalty:
        - Tx_rate < 1500: weak     (penalty = 1.0)
        - 1500–5000: moderate (linear from 1.0 --> 0.5)
        - > 5000: strong   (linear from 0.5 --> 0.0, capped at 10,000)
        For dG penalty:
        - dG > -1.5: weak (penalty = 1.0)
        - dG between -1.5 and -3.0: moderate (linear scale from 1 to 0.5)
        - dG < -3.0: strong (linear from 0.5 --> 0.0, capped at -5.0)

    Args:
        sequences: Sequence or list[Sequence] (DNA only).
        config: optional params for promoter_calculator:
            - add_context (bool, default False, adds additional nucleotides to end of sequence to ensure
            sequence meets promoter calcualtor length minimums)
            - context_length (int, default 10, amount of additional nucleotides to add)
            - threads (int, default 1)
            - verbosity (int, default 0)
            - circular (bool, default False, circularizes sequence if needed)
            - batch_size (int, default None, process all at once)
            - scoring_type (string, default = dG)

    Returns:
        float or list[float]: penalty scores.
    """
    if config is None:
        config = {}

    is_single = isinstance(sequences, Sequence)
    if is_single:
        sequences = [sequences]

    # Extract config parameters
    add_context = bool(config.get("add_context", False))
    context_length = int(config.get("context_length", 10))
    threads = int(config.get("threads", 8))
    verbosity = int(config.get("verbosity", 0))
    circular = bool(config.get("circular", False))
    batch_size = config.get("batch_size", None)  # If None, process all at once
    scoring_type = config.get("scoring_type", "dG")

    # Clean all sequences
    processed_sequences = []
    for seq_obj in sequences:
        s = seq_obj.sequence.upper().replace(" ", "").replace("\n", "")
        if add_context:
            s = ("A" * context_length) + s + ("A" * context_length)
        processed_sequences.append(s)

    penalties: List[float] = []

    # Process in batches if batch_size is specified, otherwise all at once
    if batch_size and batch_size < len(processed_sequences):
        # Process in batches
        all_results = []
        for i in range(0, len(processed_sequences), batch_size):
            batch = processed_sequences[i : i + batch_size]
            batch_results = []
            for seq in batch:
                res = (
                    promoter_calculator(
                        seq, threads=threads, verbosity=verbosity, circular=circular
                    )
                    or []
                )
                batch_results.append(res)
            all_results.extend(batch_results)
    else:
        try:
            # Attempt batch processing
            all_results = promoter_calculator(
                processed_sequences,
                threads=threads,
                verbosity=verbosity,
                circular=circular,
            )
            if not isinstance(all_results[0], list):
                raise NotImplementedError("Batch processing format not recognized")
        except (TypeError, AttributeError, NotImplementedError):
            all_results = []
            for seq in processed_sequences:
                res = (
                    promoter_calculator(
                        seq, threads=threads, verbosity=verbosity, circular=circular
                    )
                    or []
                )
                all_results.append(res)

    # Process results for each sequence
    for seq_obj, res in zip(sequences, all_results):
        # Keep only + strand
        res = [r for r in res if getattr(r, "strand", "+") == "+"]

        if not res:
            penalty = 1.0
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "reason": "no_promoter_found",
                "raw_output": [],
            }
            penalties.append(penalty)
            continue

        if scoring_type == "tx_rate":
            # Extract tx_rate
            tx_rate = max(float(r.Tx_rate) for r in res if hasattr(r, "Tx_rate"))

            # Penalty mapping
            if tx_rate < 1500.0:
                penalty = 1.0
            elif tx_rate <= 5000.0:
                penalty = 1.0 - 0.5 * ((tx_rate - 1000.0) / (5000.0 - 1000.0))
            else:
                penalty = 0.5 - min((tx_rate - 5000.0) / 5000.0 * 0.5, 0.5)
                penalty = max(0.0, penalty)

            # Store metadata
            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "tx_rate": tx_rate,
                "raw_output": [r.__dict__ for r in res],
            }
            penalties.append(penalty)
        else:
            dG = min(float(r.dG_total) for r in res if hasattr(r, "dG_total"))
            if dG >= 0:
                penalty = 1.0
            elif dG > -1.5:
                penalty = 1.0
            elif dG >= -3.0:
                penalty = 1.0 - 0.5 * ((dG + 1.5) / -1.5)
            else:
                normalized = (dG - (-3.0)) / (-5.0 - (-3.0))
                penalty = 0.5 * (1 - normalized**2)
                penalty = max(0.0, min(0.5, penalty))

            seq_obj._metadata["promoter_strength"] = {
                "penalty": penalty,
                "dG_rate": dG,
                "raw_output": [r.__dict__ for r in res],
            }
    return penalties[0] if is_single else penalties
