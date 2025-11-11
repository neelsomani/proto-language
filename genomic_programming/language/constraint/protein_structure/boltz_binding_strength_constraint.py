"""
Boltz binding strength constraint for protein-protein and protein-ligand interactions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any

from pydantic import Field

from proto_language.language.core import Sequence, SequenceType
from proto_language.base_config import BaseConfig
from proto_language.language.constraint.constraint_registry import (
    ConstraintRegistry,
)
from proto_language.tools.structure_prediction.boltz import (
    run_boltz,
    BoltzInput,
    BoltzConfig,
)
from proto_language.tools.structure_prediction.schemas import (
    StructurePredictionComplex,
)
from proto_language.language.core.sequence import SequenceType
from numpy import clip


# Default target values and tolerances for binding strength metrics
DEFAULT_DESIRED_HIGHER = {
    "iptm": 0.90,
    "ligand_iptm": 0.80,
    "protein_iptm": 0.85,
    "complex_iplddt": 0.85,
    "complex_plddt": 0.80,
    "ptm": 0.70,
    "confidence_score": 0.85,
}

DEFAULT_DESIRED_LOWER = {
    "complex_ipde": 2.0,  # Angstroms
    "complex_pde": 2.0,  # Angstroms
}

DEFAULT_TOL_HIGHER = {
    "iptm": 0.05,
    "ligand_iptm": 0.10,
    "protein_iptm": 0.07,
    "complex_iplddt": 0.10,
    "complex_plddt": 0.15,
    "ptm": 0.15,
    "confidence_score": 0.10,
}

DEFAULT_TOL_LOWER = {
    "complex_ipde": 2.0,  # Angstroms
    "complex_pde": 3.0,  # Angstroms
}


class BoltzBindingStrengthConfig(BaseConfig):
    """Configuration for Boltz binding strength constraint."""

    # Constraint-specific parameters
    desired_higher: Dict[str, float] = Field(
        default_factory=lambda: dict(),
        description="Target values for 'higher is better' metrics. Provide partial dict to override specific metrics while keeping defaults for others.",
    )
    desired_lower: Dict[str, float] = Field(
        default_factory=lambda: dict(),
        description="Target values for 'lower is better' metrics (in Angstroms). Provide partial dict to override specific metrics.",
    )
    tol_higher: Dict[str, float] = Field(
        default_factory=lambda: dict(),
        description="Tolerances for 'higher is better' metrics (distance below target = penalty 1.0). Provide partial dict to override.",
    )
    tol_lower: Dict[str, float] = Field(
        default_factory=lambda: dict(),
        description="Tolerances for 'lower is better' metrics (distance above target = penalty 1.0, in Angstroms). Provide partial dict to override.",
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for combining penalties. If None, defaults based on complex type (monomer/ligand/protein-protein)",
    )
    include_confidence_score: bool = Field(
        default=True,
        description="Whether to include confidence_score in penalty calculation (adds weight 0.10)",
    )
    on_error: str = Field(
        default="penalize",
        description="How to handle prediction errors: 'penalize' (return 1.0) or 'raise' (raise exception)",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Number of complexes to fold at once (None = process all together)",
    )
    return_component: str = Field(
        default="total_penalty",
        description="Component to return: 'total_penalty' (weighted combination) or specific metric name like 'iptm', 'ligand_iptm', 'complex_iplddt', etc.",
    )

    # Nested Boltz2 configuration
    boltz_config: Optional[BoltzConfig] = Field(
        default=None,
        description="Optional Boltz2 configuration (use_msa_server, msa_server_url, recycling_steps, sampling_steps, diffusion_samples, num_workers, devices, verbose). If None, uses defaults.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Merges user overrides with defaults after validation. If no user overrides
        are provided, uses defaults.
        """
        super().model_post_init(__context)
        self.desired_higher = {**DEFAULT_DESIRED_HIGHER, **self.desired_higher}
        self.desired_lower = {**DEFAULT_DESIRED_LOWER, **self.desired_lower}
        self.tol_higher = {**DEFAULT_TOL_HIGHER, **self.tol_higher}
        self.tol_lower = {**DEFAULT_TOL_LOWER, **self.tol_lower}


@ConstraintRegistry.register(
    key="boltz-binding-strength",
    label="Boltz Binding Strength",
    config=BoltzBindingStrengthConfig,
    description="Evaluate protein-protein/protein-ligand binding using Boltz2 structure prediction",
    batched=True,
    concatenate=False,  # Boltz handles multi-chain complexes
    gpu_required=True,
)
def boltz_binding_strength_constraint(
    complex_sequences: List[List[Sequence]], config: BoltzBindingStrengthConfig
) -> Union[float, List[float]]:
    """
    Runs Boltz2 to predict structure(s)/complex(es) and computes a binding-strength
    penalty in [0,1], where:
        0 = close to ideal (desired binding/structure)
        1 = poor (>= tolerance away from targets)

    Args:
      complex_sequences: List[List[Sequence]]:
        List of lists of sequences where each inner list is a complex containing
            all the sequences that should be predicted within a single complex.

      config: BoltzBindingStrengthConfig:
        Configuration containing penalty calculation parameters and Boltz2 prediction parameters.
        See BoltzBindingStrengthConfig for full parameter descriptions

    Returns:
      float or list[float]: penalty score(s).
    """

    # Prepare inputs for Boltz2
    inputs = BoltzInput(
        complexes=[
            StructurePredictionComplex(
                chains=[s.sequence for s in seq_list],
                entity_types=[s.sequence_type for s in seq_list],
            )  # entity types are auto-inferred
            for seq_list in complex_sequences
        ]
    )

    # Run Boltz2
    outputs = run_boltz(inputs=inputs, config=config.boltz_config or BoltzConfig())

    # Scoring each complex
    penalties = []
    for seq_obj_tuple, comp, structure in zip(complex_sequences, inputs, outputs):

        # Determine complex type
        n_chains = comp.num_chains()
        has_ligand = SequenceType.LIGAND.value in comp.get_entity_type_set()

        # Default weights by case
        if config.weights is not None:
            weights = dict(config.weights)
        else:
            # Weights for monomer
            if n_chains == 1:
                weights = {"ptm": 0.35, "complex_plddt": 0.45, "complex_pde": 0.20}

            # Weights for complex that contains a ligand
            elif has_ligand:
                weights = {
                    "ligand_iptm": 0.50,
                    "complex_iplddt": 0.25,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }
            # Weights for multi-chain complex (no ligand)
            else:
                weights = {
                    "iptm": 0.45,
                    "complex_iplddt": 0.30,
                    "complex_ipde": 0.15,
                    "complex_plddt": 0.10,
                }

        # Add confidence score weight if requested
        if config.include_confidence_score:
            weights.setdefault("confidence_score", 0.10)

        # Initialize penalties dictionary
        penalties_dict = {}

        # Case-specific penalties
        if n_chains == 1:
            # Monomer penalties
            penalties_dict["ptm_penalty"] = get_penalty_for_metric(
                metric_name="ptm", metric_value=structure.ptm, config=config
            )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )
            if structure.complex_pde is not None:
                penalties_dict["complex_pde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_pde",
                    metric_value=structure.complex_pde,
                    config=config,
                )

        elif has_ligand:
            penalties_dict["ligand_iptm_penalty"] = get_penalty_for_metric(
                metric_name="ligand_iptm",
                metric_value=structure.ligand_iptm,
                config=config,
            )
            penalties_dict["complex_iplddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_iplddt",
                metric_value=structure.complex_iplddt,
                config=config,
            )
            if structure.complex_ipde is not None:
                penalties_dict["complex_ipde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_ipde",
                    metric_value=structure.complex_ipde,
                    config=config,
                )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )

        else:
            prot_iptm = structure.protein_iptm
            iptm = structure.iptm
            chosen = "protein_iptm" if (prot_iptm and prot_iptm > 0) else "iptm"
            if chosen == "iptm":
                penalties_dict["iptm_penalty"] = get_penalty_for_metric(
                    metric_name="iptm",
                    metric_value=iptm,
                    config=config,
                )
            else:
                penalties_dict["protein_iptm_penalty"] = get_penalty_for_metric(
                    metric_name="protein_iptm",
                    metric_value=prot_iptm,
                    config=config,
                )
            penalties_dict["complex_iplddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_iplddt",
                metric_value=structure.complex_iplddt,
                config=config,
            )
            if structure.complex_ipde is not None:
                penalties_dict["complex_ipde_penalty"] = get_penalty_for_metric(
                    metric_name="complex_ipde",
                    metric_value=structure.complex_ipde,
                    config=config,
                )
            penalties_dict["complex_plddt_penalty"] = get_penalty_for_metric(
                metric_name="complex_plddt",
                metric_value=structure.complex_plddt,
                config=config,
            )

        if "confidence_score" in weights:
            penalties_dict["confidence_score_penalty"] = get_penalty_for_metric(
                metric_name="confidence_score",
                metric_value=structure.confidence_score,
                config=config,
            )

        # If user requests a specific component
        if config.return_component != "total_penalty":
            key = config.return_component.strip()
            if not key.endswith("_penalty"):
                key = f"{key}_penalty"
            if key not in penalties_dict:
                raise ValueError(
                    f"Requested component '{config.return_component}' not available."
                )
            penalty = clip(float(penalties_dict[key]), 0.0, 1.0)
        else:
            # Weighted sum
            used_weights = {
                k: weights[k.replace("_penalty", "")]
                for k in penalties_dict
                if k.replace("_penalty", "") in weights
            }
            wsum = sum(used_weights.values()) or 1.0
            penalty = clip(
                sum((w / wsum) * penalties_dict[k] for k, w in used_weights.items()),
                0.0,
                1.0,
            )

        # Store metadata for all Sequences in complex
        for seq_obj in seq_obj_tuple:
            seq_obj._metadata.setdefault("boltz_binding", []).append(
                {
                    "penalty": penalty,
                    "metrics": structure.metrics,
                    "penalties": penalties_dict,
                }
            )

        penalties.append(penalty)

    return penalties


def get_penalty_for_metric(
    metric_name: str, metric_value: float, config: BoltzBindingStrengthConfig
) -> float:
    """
    Retrieves the penalty for the given metric's value based on the default target
    and tolerance values.

    Args:
        metric_name: The name of the metric to retrieve the penalty for.
        metric_value: The value of the metric to retrieve the penalty for.

    Returns:
        The penalty for the given metric's value.
    """

    higher_is_better = None
    target = None
    tolerance = None
    if metric_name in config.desired_higher and metric_name in config.desired_lower:
        raise ValueError(
            f"Metric {metric_name} has both desired_higher and tol_higher values. Please provide only one."
        )
    elif metric_name in config.desired_higher:
        higher_is_better = True
        target = config.desired_higher[metric_name]
        tolerance = config.tol_higher[metric_name]
    elif metric_name in config.desired_lower:
        higher_is_better = False
        target = config.desired_lower[metric_name]
        tolerance = config.tol_lower[metric_name]
    else:
        raise ValueError(
            f"Metric {metric_name} not found in config.desired_higher or config.desired_lower"
        )

    deviation = (target - metric_value) if higher_is_better else (metric_value - target)
    normalized = deviation / max(tolerance, 1e-9)
    return max(0.0, min(1.0, normalized))
