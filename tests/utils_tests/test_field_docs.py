"""Tests for per-field docstring extraction and ``x-proto-doc`` schema injection.

Every constraint/generator/optimizer config field must resolve to the full
docstring text from its class's Google-style ``Attributes:`` section. That text
is the source for the schema's ``x-proto-doc`` annotation and the CLI's per-field
help, so the coverage check guards both surfaces going forward.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from proto_language.cli import main
from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.generator.generator_registry import GeneratorRegistry
from proto_language.optimizer.optimizer_registry import OptimizerRegistry
from proto_language.utils.field_docs import field_docs_from_docstrings, inject_field_docs


def _config_models() -> list[tuple[str, type[BaseModel]]]:
    """Return ``(key, config_model)`` for every registered component."""
    specs = [
        *ConstraintRegistry.list_all(),
        *GeneratorRegistry.list_all(),
        *OptimizerRegistry.list_all(),
    ]
    return [(spec.key, spec.config_model) for spec in specs]


_CONFIG_MODELS = _config_models()
_IDS = [f"{key}:{model.__name__}" for key, model in _CONFIG_MODELS]


def test_components_discovered() -> None:
    """Registries are populated, so the parametrized checks aren't vacuous."""
    assert len(_CONFIG_MODELS) > 50


@pytest.mark.parametrize("key, model", _CONFIG_MODELS, ids=_IDS)
def test_every_config_field_has_docstring(key: str, model: type[BaseModel]) -> None:
    """Every config field resolves to non-empty docstring text."""
    docs = field_docs_from_docstrings(model)
    missing = [name for name in model.model_fields if not docs.get(name)]
    assert not missing, f"{key} ({model.__name__}) missing field docstrings for: {missing}"


@pytest.mark.parametrize("key, model", _CONFIG_MODELS, ids=_IDS)
def test_schema_carries_x_proto_doc(key: str, model: type[BaseModel]) -> None:
    """``inject_field_docs`` annotates every documented top-level property."""
    docs = field_docs_from_docstrings(model)
    schema = inject_field_docs(model.model_json_schema(), model)
    for name, prop in schema.get("properties", {}).items():
        if docs.get(name):
            assert prop.get("x-proto-doc") == docs[name], f"{key}: '{name}' missing x-proto-doc"


def test_inherited_optimizer_seed_doc_resolves() -> None:
    """Inherited ``seed`` (documented only on ``BaseOptimizerConfig``) is picked up."""
    _, mcmc = next((k, m) for k, m in _CONFIG_MODELS if k == "mcmc")
    docs = field_docs_from_docstrings(mcmc)
    assert "seed" in docs and "reproducible" in docs["seed"].lower()


def test_subclass_overrides_inherited_doc() -> None:
    """A subclass that re-documents an inherited field wins over the base text."""

    class Base(BaseModel):
        """Base.

        Attributes:
            shared (int): Base description.
        """

        shared: int = 0

    class Child(Base):
        """Child.

        Attributes:
            shared (int): Child description.
        """

        shared: int = 1

    assert field_docs_from_docstrings(Child)["shared"] == "Child description."


def test_nested_models_annotated_under_defs() -> None:
    """Nested model defs also receive ``x-proto-doc`` on their fields."""
    nested_found = False
    for _key, model in _CONFIG_MODELS:
        schema = inject_field_docs(model.model_json_schema(), model)
        defs = schema.get("$defs", {})
        for def_schema in defs.values():
            for prop in def_schema.get("properties", {}).values():
                if isinstance(prop, dict) and prop.get("x-proto-doc"):
                    nested_found = True
    assert nested_found, "expected at least one nested $defs property with x-proto-doc"


def test_cli_config_renders_docstring(capsys: pytest.CaptureFixture[str]) -> None:
    """``proto-language constraint config`` prints the rich docstring text."""
    rc = main(["constraint", "config", "gc-content"])
    out = capsys.readouterr().out
    assert rc == 0
    # The rich Attributes text is longer/different than the terse Field description.
    assert "min_gc" in out
    assert "penalized" in out  # appears in the Attributes docstring, not the terse one
