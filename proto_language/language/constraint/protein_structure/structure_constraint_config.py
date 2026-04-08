"""Base configuration classes for structure-based constraints.

This module provides standardized configuration classes for constraints that
use structure prediction tools (ESMFold, AlphaFold3, Boltz2, Chai1).
"""

from typing import Any, Literal

from proto_tools import (
    AlphaFold3Config,
    Boltz2Config,
    Chai1Config,
    ESMFoldConfig,
)
from pydantic import model_validator

from proto_language.base_config import BaseConfig, ConfigField


class StructureBasedConstraintConfig(BaseConfig):
    """Base configuration for constraints using structure prediction tools.

    This base class standardizes how structure prediction tools and their
    configurations are specified across all structure-based constraints.
    Each tool has its own dedicated config field gated by ``depends_on``,
    so the client renders only the config form for the selected tool.

    Subclasses can optionally restrict which tools are supported by overriding
    the structure_tool field with a narrower Literal type.

    Attributes:
        structure_tool (Literal['esmfold', 'alphafold3', 'boltz2', 'chai1']): Tool to use for structure prediction. Supported options:
            - "esmfold": ESMFold (Meta AI)
            - "alphafold3": AlphaFold 3 (Google DeepMind)
            - "boltz2": Boltz2 (MIT)
            - "chai1": Chai-1 (Chai Discovery)
            Default is "esmfold".

        esmfold_config (ESMFoldConfig): Configuration for ESMFold structure prediction.
            Only visible when ``structure_tool == "esmfold"``.

        alphafold3_config (AlphaFold3Config): Configuration for AlphaFold3 structure prediction.
            Only visible when ``structure_tool == "alphafold3"``.

        boltz2_config (Boltz2Config): Configuration for Boltz2 structure prediction.
            Only visible when ``structure_tool == "boltz2"``.

        chai1_config (Chai1Config): Configuration for Chai1 structure prediction.
            Only visible when ``structure_tool == "chai1"``.

    Example:
        >>> config = MyConstraintConfig(structure_tool="esmfold", esmfold_config=ESMFoldConfig(device="cuda"))
        >>>
        >>> config = MyConstraintConfig(structure_tool="alphafold3", alphafold3_config={"seeds": [0, 1]})
    """

    structure_tool: Literal["esmfold", "alphafold3", "boltz2", "chai1"] = ConfigField(
        title="Structure Prediction Tool",
        default="esmfold",
        description="Tool to use for structure prediction.",
    )

    esmfold_config: ESMFoldConfig = ConfigField(
        default_factory=ESMFoldConfig,
        title="ESMFold Configuration",
        description="Configuration for ESMFold structure prediction.",
        advanced=True,
        depends_on={"field": "structure_tool", "value": "esmfold"},
    )
    alphafold3_config: AlphaFold3Config = ConfigField(
        default_factory=AlphaFold3Config,
        title="AlphaFold3 Configuration",
        description="Configuration for AlphaFold3 structure prediction.",
        advanced=True,
        depends_on={"field": "structure_tool", "value": "alphafold3"},
    )
    boltz2_config: Boltz2Config = ConfigField(
        default_factory=Boltz2Config,
        title="Boltz2 Configuration",
        description="Configuration for Boltz2 structure prediction.",
        advanced=True,
        depends_on={"field": "structure_tool", "value": "boltz2"},
    )
    chai1_config: Chai1Config = ConfigField(
        default_factory=Chai1Config,
        title="Chai1 Configuration",
        description="Configuration for Chai1 structure prediction.",
        advanced=True,
        depends_on={"field": "structure_tool", "value": "chai1"},
    )

    @property
    def tool_config(self) -> ESMFoldConfig | AlphaFold3Config | Boltz2Config | Chai1Config:
        """Return the active tool configuration based on structure_tool."""
        configs = {
            "esmfold": self.esmfold_config,
            "alphafold3": self.alphafold3_config,
            "boltz2": self.boltz2_config,
            "chai1": self.chai1_config,
        }
        return configs[self.structure_tool]

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_tool_config(cls, values: Any) -> Any:
        """Reject legacy tool_config kwarg with a helpful migration message."""
        if isinstance(values, dict) and "tool_config" in values:
            raise ValueError(
                "tool_config is no longer accepted. Use the per-tool config field instead: "
                "esmfold_config, alphafold3_config, boltz2_config, or chai1_config."
            )
        return values
