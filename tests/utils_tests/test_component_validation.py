"""Tests for proto_language.utils.component_validation."""

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from proto_language.utils import (
    LintResult,
    TestResult,
    lint_component_file,
    test_constraint,
    test_generator,
    test_optimizer,
)

GOOD_CONSTRAINT_SOURCE = textwrap.dedent(
    """
    from proto_language.base_config import BaseConfig, ConfigField
    from proto_language.language.constraint.constraint_registry import constraint
    from proto_language.language.core import ConstraintOutput, Sequence


    class ToyConfig(BaseConfig):
        threshold: float = ConfigField(default=0.5, description="Threshold.", title="Threshold")


    @constraint(
        key="toy-constraint",
        label="Toy",
        config=ToyConfig,
        description="Toy.",
        supported_sequence_types=["protein"],
        tools_called=[],
        category="testing",
    )
    def toy_constraint(
        input_sequences: list[tuple[Sequence, ...]], config: ToyConfig
    ) -> list[ConstraintOutput]:
        return [ConstraintOutput(score=config.threshold) for _ in input_sequences]
    """
).lstrip()


GOOD_GENERATOR_SOURCE = textwrap.dedent(
    """
    from proto_language import BaseConfig, Generator, generator


    class ToyGeneratorConfig(BaseConfig):
        pass


    @generator(
        key="toy-generator",
        label="Toy",
        config=ToyGeneratorConfig,
        description="Toy.",
        category="mutation",
        supported_sequence_types=["dna"],
    )
    class ToyGenerator(Generator):
        def __init__(self, config: ToyGeneratorConfig):
            super().__init__()
            self.config = config

        def sample(self) -> None:
            pass
    """
).lstrip()


GOOD_OPTIMIZER_SOURCE = textwrap.dedent(
    """
    from proto_language.base_config import BaseOptimizerConfig
    from proto_language.language.core import Optimizer
    from proto_language.language.optimizer.optimizer_registry import optimizer


    class ToyOptimizerConfig(BaseOptimizerConfig):
        pass


    @optimizer(
        key="toy-optimizer",
        label="Toy",
        config=ToyOptimizerConfig,
        description="Toy.",
    )
    class ToyOptimizer(Optimizer):
        def run(self) -> None:
            pass
    """
).lstrip()


def _lint(tmp_path: Path, source: str, filename: str = "component.py") -> LintResult:
    path = tmp_path / filename
    path.write_text(source)
    return lint_component_file(path)


def _has_error(result: LintResult, substring: str) -> bool:
    return any(substring.lower() in error.lower() for error in result.errors)


@pytest.mark.parametrize(
    ("source", "component_type", "registry_key"),
    [
        (GOOD_CONSTRAINT_SOURCE, "constraint", "toy-constraint"),
        (GOOD_GENERATOR_SOURCE, "generator", "toy-generator"),
        (GOOD_OPTIMIZER_SOURCE, "optimizer", "toy-optimizer"),
    ],
)
def test_lint_accepts_well_formed_components(
    tmp_path: Path, source: str, component_type: str, registry_key: str
) -> None:
    result = _lint(tmp_path, source)

    assert result.success
    assert result.component_type == component_type
    assert result.registry_key == registry_key


def test_lint_accepts_public_proto_language_imports(tmp_path: Path) -> None:
    source = GOOD_CONSTRAINT_SOURCE.replace(
        "from proto_language.base_config import BaseConfig, ConfigField\n"
        "from proto_language.language.constraint.constraint_registry import constraint\n"
        "from proto_language.language.core import ConstraintOutput, Sequence\n",
        "from proto_language import BaseConfig, ConstraintOutput, Sequence, constraint\n",
    ).replace(
        'threshold: float = ConfigField(default=0.5, description="Threshold.", title="Threshold")',
        "threshold: float = 0.5",
    )

    assert _lint(tmp_path, source).success


@pytest.mark.parametrize(
    ("transform", "error_substring"),
    [
        (lambda src: src.replace("from proto_language.base_config import BaseConfig, ConfigField\n", ""), "BaseConfig"),
        (lambda src: src.replace("class ToyConfig(BaseConfig):", "class ToyConfig(SomeOtherBase):"), "config class"),
        (
            lambda src: src.replace(
                "from proto_language.base_config import BaseConfig, ConfigField",
                "from proto_language.base_config import BaseConfig, BaseOptimizerConfig, ConfigField",
            ).replace("class ToyConfig(BaseConfig):", "class ToyConfig(BaseOptimizerConfig):"),
            "should inherit from BaseConfig",
        ),
        (lambda src: src.replace('key="toy-constraint"', 'key="Toy_Constraint"'), "kebab-case"),
        (lambda src: src.replace("config=ToyConfig", "config=MissingConfig"), "unknown class 'MissingConfig'"),
        (lambda src: src.replace('supported_sequence_types=["protein"],\n', ""), "supported_sequence_types"),
        (
            lambda src: src.replace(
                "def toy_constraint(\n"
                "    input_sequences: list[tuple[Sequence, ...]], config: ToyConfig\n"
                ") -> list[ConstraintOutput]:",
                "def toy_constraint() -> list[ConstraintOutput]:",
            ),
            "at least 2 parameters",
        ),
    ],
)
def test_lint_rejects_invalid_constraint_contract(
    tmp_path: Path, transform: Callable[[str], str], error_substring: str
) -> None:
    result = _lint(tmp_path, transform(GOOD_CONSTRAINT_SOURCE))

    assert not result.success
    assert _has_error(result, error_substring)


def test_lint_rejects_multiple_components(tmp_path: Path) -> None:
    second_constraint = textwrap.dedent(
        """

        @constraint(
            key="second-constraint",
            label="Second",
            config=ToyConfig,
            description="Second.",
            supported_sequence_types=["protein"],
        )
        def second_constraint(
            input_sequences: list[tuple[Sequence, ...]], config: ToyConfig
        ) -> list[ConstraintOutput]:
            return []
        """
    )

    result = _lint(tmp_path, GOOD_CONSTRAINT_SOURCE + second_constraint)

    assert not result.success
    assert _has_error(result, "exactly one @constraint")


@pytest.mark.parametrize(
    ("source", "error_substring"),
    [
        ("def broken(:\n    pass\n", "Line "),
        ("def plain():\n    return 1\n", "@constraint"),
        (GOOD_GENERATOR_SOURCE.replace("    def sample(self) -> None:\n        pass\n", ""), "sample"),
    ],
)
def test_lint_rejects_unusable_files(tmp_path: Path, source: str, error_substring: str) -> None:
    result = _lint(tmp_path, source)

    assert not result.success
    assert _has_error(result, error_substring)


def test_lint_missing_path_returns_error(tmp_path: Path) -> None:
    result = lint_component_file(tmp_path / "does_not_exist.py")

    assert not result.success
    assert _has_error(result, "File not found")


@pytest.mark.parametrize(
    ("seq", "expected", "should_pass"),
    [
        ("ATGC", [0.0], True),
        ("AAAA", [0.0], False),
        ("ATGC", None, True),
    ],
)
def test_constraint_helper_gc_content(seq: str, expected: list[float] | None, should_pass: bool) -> None:
    result = test_constraint(
        "gc-content",
        [seq],
        config={"min_gc": 40.0, "max_gc": 60.0},
        expected_scores=expected,
        tolerance=0.05,
        sequence_type="dna",
    )

    assert isinstance(result, TestResult)
    assert result.passed is should_pass
    assert len(result.actual) == 1


def test_generator_helper_checks_length_and_alphabet() -> None:
    result = test_generator(
        "random-nucleotide",
        segment_length=12,
        n_samples=3,
        sequence_type="dna",
        expected_alphabet="ACGT",
    )

    assert result.passed
    assert len(result.actual) == 3
    assert all(len(seq) == 12 and set(seq) <= set("ACGT") for seq in result.actual)


@pytest.mark.parametrize(
    ("config", "should_pass"),
    [({"num_steps": 10}, True), ({}, False)],
)
def test_optimizer_helper_validates_config(config: dict, should_pass: bool) -> None:
    assert test_optimizer("mcmc", config=config).passed is should_pass


_LOAD_CONSTRAINT_SOURCE = textwrap.dedent(
    """
    from proto_language.base_config import BaseConfig, ConfigField
    from proto_language.language.constraint.constraint_registry import constraint
    from proto_language.language.core import ConstraintOutput, Sequence


    class LoadProbeConfig(BaseConfig):
        offset: float = ConfigField(default=0.25, title="Offset", description="Offset.")


    @constraint(
        key="load-probe-constraint",
        label="Load Probe",
        config=LoadProbeConfig,
        description="Probe used by test_constraint(load=...).",
        supported_sequence_types=["protein"],
        tools_called=[],
        category="testing",
    )
    def load_probe_constraint(
        input_sequences: list[tuple[Sequence, ...]], config: LoadProbeConfig
    ) -> list[ConstraintOutput]:
        return [ConstraintOutput(score=config.offset) for _ in input_sequences]
    """
).lstrip()


def test_constraint_load_kwarg_registers_workspace_file(tmp_path: Path) -> None:
    """test_constraint(load=...) execs the file so its decorator registers."""
    from proto_language.language.constraint import ConstraintRegistry

    key = "load-probe-constraint"
    path = tmp_path / "load_probe.py"
    path.write_text(_LOAD_CONSTRAINT_SOURCE)

    # Sanity: not registered yet.
    with pytest.raises((KeyError, ValueError)):
        ConstraintRegistry.get(key)

    try:
        result = test_constraint(
            key,
            sequences=["MAKL"],
            expected_scores=[0.25],
            tolerance=1e-6,
            load=path,
        )
        assert result.passed
        assert result.actual == [0.25]
    finally:
        ConstraintRegistry._registry.pop(key, None)
