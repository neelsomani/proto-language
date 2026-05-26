"""Tests for ``proto_language.utils.docs_api``."""

from __future__ import annotations

import pytest

from proto_language.constraint.constraint_registry import ConstraintRegistry
from proto_language.generator.generator_registry import GeneratorRegistry
from proto_language.utils.docs_api import (
    CompatibilityReport,
    ComponentDoc,
    ConfigModelDoc,
    ConstraintSpecMetadata,
    CoreTypeDoc,
    GeneratorSpecMetadata,
    OptimizerSpecMetadata,
    get_compatibility,
    get_constraint_doc,
    get_core_type_doc,
    get_generator_doc,
    get_optimizer_doc,
    resolve_key,
)

# ----------------------------------------------------------------------------
# resolve_key
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier",
    ["gc-content", "gc_content", "gc_content_constraint"],
)
def test_resolve_key_three_identifier_forms(identifier: str) -> None:
    """Registry key, snake_case form, and function ``__name__`` all collapse to the same key."""
    assert resolve_key("constraint", identifier) == "gc-content"


def test_resolve_key_unknown_raises() -> None:
    """Unknown identifiers raise ValueError listing the available keys."""
    with pytest.raises(ValueError, match="Unknown constraint"):
        resolve_key("constraint", "definitely-not-a-real-constraint")


# ----------------------------------------------------------------------------
# Constraint docs
# ----------------------------------------------------------------------------


def test_get_constraint_doc_shape() -> None:
    """A real constraint produces a well-formed ComponentDoc with typed metadata."""
    doc = get_constraint_doc("gc-content")
    assert isinstance(doc, ComponentDoc)
    assert doc.kind == "constraint"
    assert doc.key == "gc-content"
    assert doc.label
    assert doc.description
    assert doc.docstring, "constraint function docstring should be populated"
    assert isinstance(doc.config, ConfigModelDoc)
    assert doc.config.fields, "GC content config has at least min_gc/max_gc"
    assert isinstance(doc.spec_metadata, ConstraintSpecMetadata)
    assert doc.spec_metadata.mode in {"discrete", "gradient", "dual"}
    assert "dna" in doc.spec_metadata.supported_sequence_types


# ----------------------------------------------------------------------------
# Generator docs
# ----------------------------------------------------------------------------


def test_get_generator_doc_shape() -> None:
    """Generator docs include typed category + input_type spec metadata."""
    keys = [s.key for s in GeneratorRegistry.list_all()]
    assert keys, "no generators registered"
    doc = get_generator_doc(keys[0])
    assert doc.kind == "generator"
    assert isinstance(doc.spec_metadata, GeneratorSpecMetadata)
    assert doc.spec_metadata.category in {"autoregressive", "mutation", "inverse_folding", "gradient"}
    assert doc.spec_metadata.input_type in {"prompt", "starting_sequence", "structure", "logits"}


# ----------------------------------------------------------------------------
# Optimizer docs
# ----------------------------------------------------------------------------


def test_get_optimizer_doc_shape() -> None:
    """Optimizer docs expose typed compatible_generators + targets_single_segment."""
    doc = get_optimizer_doc("mcmc")
    assert doc.kind == "optimizer"
    assert doc.docstring
    assert isinstance(doc.spec_metadata, OptimizerSpecMetadata)
    assert doc.spec_metadata.required_constraint_mode in {None, "discrete", "gradient"}


# ----------------------------------------------------------------------------
# Compatibility graph
# ----------------------------------------------------------------------------


def test_compatibility_optimizer_filters_by_required_constraint_mode() -> None:
    """For mcmc (required_constraint_mode='discrete'), every returned constraint supports discrete."""
    report = get_compatibility("optimizer", "mcmc")
    assert isinstance(report, CompatibilityReport)
    spec = get_constraint_doc  # to verify per-constraint metadata
    for c_key in report.compatible_constraints:
        c_doc = spec(c_key)
        assert isinstance(c_doc.spec_metadata, ConstraintSpecMetadata)
        assert c_doc.spec_metadata.mode in {"discrete", "dual"}


def test_compatibility_constraint_reverses_optimizer_mode_check() -> None:
    """A discrete-only constraint pairs with every optimizer not requiring gradient mode."""
    report = get_compatibility("constraint", "gc-content")
    assert report.compatible_optimizers, "gc-content should pair with at least one optimizer"
    for o_key in report.compatible_optimizers:
        o_doc = get_optimizer_doc(o_key)
        assert isinstance(o_doc.spec_metadata, OptimizerSpecMetadata)
        assert o_doc.spec_metadata.required_constraint_mode in {None, "discrete"}


def test_compatibility_generator_uses_optimizer_allow_list() -> None:
    """A generator's compatible optimizers are exactly those whose compatible_generators include it."""
    keys = [s.key for s in GeneratorRegistry.list_all()]
    assert keys
    g_key = keys[0]
    report = get_compatibility("generator", g_key)
    for o_key in report.compatible_optimizers:
        o_doc = get_optimizer_doc(o_key)
        assert isinstance(o_doc.spec_metadata, OptimizerSpecMetadata)
        compat = o_doc.spec_metadata.compatible_generators
        assert compat is None or g_key in compat


# ----------------------------------------------------------------------------
# Registry wrappers
# ----------------------------------------------------------------------------


def test_registry_get_docs_matches_module_function() -> None:
    """ConstraintRegistry.get_docs returns the same payload as get_constraint_doc."""
    via_module = get_constraint_doc("gc-content")
    via_registry = ConstraintRegistry.get_docs("gc-content")
    assert via_module.model_dump() == via_registry.model_dump()


# ----------------------------------------------------------------------------
# Core types
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Sequence", "Program"])
def test_get_core_type_doc(name: str) -> None:
    """Each core type yields a docstring + parameter list."""
    doc = get_core_type_doc(name)
    assert isinstance(doc, CoreTypeDoc)
    assert doc.name == name
    assert doc.docstring, f"{name} missing class docstring"
    assert doc.params, f"{name}.__init__ produced no params"


def test_get_core_type_doc_unknown_raises() -> None:
    """Unknown core types raise ValueError listing the valid names."""
    with pytest.raises(ValueError, match="Unknown core type"):
        get_core_type_doc("NotARealType")
