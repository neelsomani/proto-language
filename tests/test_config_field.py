"""
Unit tests for ConfigField, focusing on the depends_on parameter and its
serialization into model_json_schema().
"""
from __future__ import annotations

from typing import Optional

import pytest

from proto_language.base_config import BaseConfig, ConfigField

# ---------------------------------------------------------------------------
# Test: depends_on with a single value
# ---------------------------------------------------------------------------


def test_depends_on_single_value():
    """depends_on with a single string value produces correct x-depends-on."""

    class M(BaseConfig):
        """Test model."""
        mode: str = ConfigField(
            default="basic",
            title="Mode",
            description="Operating mode.",
        )
        detail: str = ConfigField(
            default="off",
            title="Detail",
            description="Detail level.",
            depends_on={"field": "mode", "value": "advanced"},
        )

    schema = M.model_json_schema()
    detail_props = schema["properties"]["detail"]
    assert detail_props["x-depends-on"] == {
        "field": "mode",
        "value": "advanced",
    }


# ---------------------------------------------------------------------------
# Test: depends_on with a list value
# ---------------------------------------------------------------------------


def test_depends_on_list_value():
    """depends_on with a list of values produces correct x-depends-on."""

    class M(BaseConfig):
        """Test model."""
        tool: str = ConfigField(
            default="a",
            title="Tool",
            description="Tool selector.",
        )
        param: int = ConfigField(
            default=1,
            title="Param",
            description="A parameter.",
            depends_on={"field": "tool", "value": ["a", "b"]},
        )

    schema = M.model_json_schema()
    param_props = schema["properties"]["param"]
    assert param_props["x-depends-on"] == {
        "field": "tool",
        "value": ["a", "b"],
    }


# ---------------------------------------------------------------------------
# Test: depends_on with not_null
# ---------------------------------------------------------------------------


def test_depends_on_not_null():
    """depends_on with not_null=True produces correct x-depends-on."""

    class M(BaseConfig):
        """Test model."""
        target: Optional[str] = ConfigField(
            default=None,
            title="Target",
            description="Optional target.",
        )
        extra: str = ConfigField(
            default="x",
            title="Extra",
            description="Extra option.",
            depends_on={"field": "target", "not_null": True},
        )

    schema = M.model_json_schema()
    extra_props = schema["properties"]["extra"]
    assert extra_props["x-depends-on"] == {
        "field": "target",
        "not_null": True,
    }


# ---------------------------------------------------------------------------
# Test: depends_on truthy check (field only, no value/not_null)
# ---------------------------------------------------------------------------


def test_depends_on_truthy_check():
    """depends_on with only 'field' key produces a truthy-check entry."""

    class M(BaseConfig):
        """Test model."""
        enabled: bool = ConfigField(
            default=False,
            title="Enabled",
            description="Toggle feature.",
        )
        sub: str = ConfigField(
            default="",
            title="Sub",
            description="Sub option.",
            depends_on={"field": "enabled"},
        )

    schema = M.model_json_schema()
    sub_props = schema["properties"]["sub"]
    xdep = sub_props["x-depends-on"]
    assert xdep == {"field": "enabled"}
    assert "value" not in xdep
    assert "not_null" not in xdep


# ---------------------------------------------------------------------------
# Test: depends_on=None (default) omits x-depends-on entirely
# ---------------------------------------------------------------------------


def test_depends_on_none_omits_key():
    """When depends_on is None (default), x-depends-on must not appear."""

    class M(BaseConfig):
        """Test model."""
        plain: int = ConfigField(
            default=0,
            title="Plain",
            description="A plain field.",
        )

    schema = M.model_json_schema()
    plain_props = schema["properties"]["plain"]
    assert "x-depends-on" not in plain_props


# ---------------------------------------------------------------------------
# Test: depends_on missing 'field' key raises ValueError
# ---------------------------------------------------------------------------


def test_depends_on_missing_field_key_raises():
    """depends_on without a 'field' key must raise ValueError."""
    with pytest.raises(ValueError, match="must include a 'field' key"):

        class M(BaseConfig):
            """Test model."""
            bad: str = ConfigField(
                default="",
                title="Bad",
                description="Should fail.",
                depends_on={"value": "bar"},
            )
