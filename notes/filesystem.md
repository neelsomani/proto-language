# Filesystem

This guide is about where files live and where runtime artifacts are written. For component behavior and contributor rules, use `CLAUDE.md`, the source code, and the relevant `.claude/skills/` workflow.

## Package Layout

```
proto_language/
├── core/                    Sequence, Segment, Construct, Constraint, Generator,
│                            Optimizer, Program, export helpers, validation
│   ├── sequence.py          Typed sequence plus optional logits, structure, metadata
│   ├── segment.py           One design region with proposal/result sequences
│   ├── construct.py         Ordered segments that form a biological construct
│   ├── constraint.py        Constraint wrapper, outputs, gradients, metadata writes
│   ├── generator.py         Generator ABC, assignment, tied-segment behavior
│   ├── optimizer.py         Optimizer ABC, scoring, history, export surface
│   └── program.py           Multi-stage orchestration and program-level export
├── constraint/              Registered scoring/filter functions by domain
│   ├── constraint_registry.py
│   ├── protein_quality/
│   ├── protein_structure/
│   ├── rna_secondary_structure/
│   ├── rna_splicing/
│   ├── sequence_alignment/
│   ├── sequence_annotation/
│   ├── sequence_composition/
│   └── sequence_scoring/
├── generator/               Registered proposal generators and registry
├── optimizer/               Search strategies and compiled-constraint providers
│   └── constraint_compiler/
└── utils/                   BaseConfig/BaseRegistry, scoring constants, IO,
                             logging, serialization, gradients, scheduling,
                             sequence matrices, ORF helpers
```

Important conventions:

- Add pluggable components under the appropriate `constraint/`, `generator/`, or `optimizer/` module and export them through the local `__init__.py` chain.
- Registries live beside their component families: `constraint_registry.py`, `generator_registry.py`, and `optimizer_registry.py`.
- `utils/io.py` owns result flattening and export writers. `core/program.py` and `core/optimizer.py` expose the public export methods.

## Tests

`tests/` is not a perfect mirror of `proto_language/`; it has several lanes:

```
tests/
├── conftest.py                         pytest flags, markers, fixtures, logging
├── language_tests/                     core/component behavior tests
│   ├── constraint_tests/
│   ├── generator_tests/
│   ├── optimizer_tests/
│   └── test_*.py
├── utils_tests/                        utility module tests
├── tests_cpu/                          CPU integration/regression workflows
├── test_codebase_consistency.py        repo-wide source consistency checks
└── README.md                           short marker reference
```

`tests/conftest.py` is the source of truth for custom pytest flags, automatic CPU marking, `skip_ci` and `only_chimera` behavior, and test logging. The `toy_json` fixture loads `examples/jsons/toy.json`, so the JSON program path is exercised by the suite.

See `notes/testing.md` for the long-form testing guide.

## Examples

`examples/` contains runnable programs and data. Current top-level conventions:

```
examples/
├── bin/         Standalone utility/analysis scripts; run directly
├── bindcraft/   Binder-design example programs and assets
├── germinal/    Antibody/VHH generation pipeline content
├── data/        Immutable reference assets used by examples
├── jsons/       Declarative Program.from_json() inputs
└── scripts/     Larger Python workloads and generated program collections
```

Use `examples/scripts/` and `examples/jsons/` for idiomatic program shape. Domain-specific subtrees such as `germinal/` and `bindcraft/` carry their own assets and assumptions.

## Logs

Gitignored runtime logs:

| Path | Producer |
|---|---|
| `logs/pytest_*.log` | `setup_test_logging` in `tests/conftest.py` |
| `logs/proto_language_*.log` | `setup_logging()` default filename |
| `logs/<custom>.log` | `setup_logging(log_filename=...)` |
| `tests/logs/` | Reserved test log location |

`setup_logging()` defaults to `logs/` under the nearest project root containing `pyproject.toml`. During pytest, timestamped file logging is disabled unless a fixture or caller supplies `log_filename`; the test fixture does supply `pytest_*.log`.

## Exports

`Program.export()` and `Optimizer.export()` write an export directory. If `path` is `None`, the directory is created under the current working directory using the shared proto-tools export-name convention: `{project}__{YYYY-MM-DD_HHMMSS}`.

Layout:

```
<export-dir>/
├── sequences.<fmt>
├── constraints.<fmt>
├── constructs.<fmt>
├── optimization.<fmt>
├── sequences.fasta
└── assets/
    ├── res{i}_con{c}_seg{s}_structure.{pdb|cif}
    ├── res{i}_con{c}_seg{s}_logits.npy
    └── *.csv        nested row-shaped metadata sidecars
```

Supported table formats are `csv`, `tsv`, `json`, and `xlsx`. For `xlsx`, the four tables are sheets in `<export-dir>/results.xlsx`; `sequences.fasta` and `assets/` are still written separately. Empty tables are materialized as empty files for non-XLSX formats.

Useful public helpers:

- `Program.to_dataframe(...)` for one flattened table in memory.
- `Program.to_fasta(...)` for FASTA-only output.
- `proto_language.utils.io.write_results_folder(...)` for the folder writer used by program and optimizer exports.

## Persistent Storage

`proto-language` has no storage-specific environment variables of its own. Model weights, tool environments, micromamba, and package caches are owned by the `proto-tools` submodule and inherited by language generators/constraints that call tools.

| Variable | Owned by | What it controls |
|---|---|---|
| `PROTO_HOME` | proto-tools | Top-level root, default `~/.proto/`; contains model cache, tool envs, package caches, micromamba |
| `PROTO_MODEL_CACHE` | proto-tools | Override only model-weight storage; safe for shared team caches |
| `PROTO_{TOOL_NAME}_WEIGHTS_DIR` | proto-tools | Per-tool weight override |
| `UV_CACHE_DIR` / `PIP_CACHE_DIR` | proto-tools | Optional package-cache overrides; default under `PROTO_HOME` |
| `HF_TOKEN` | proto-tools / HuggingFace | Auth for gated model downloads |

Full reference: `proto-tools/notes/storage.md`. Set these in the shell or job environment; `proto-language` picks them up through proto-tools calls.
