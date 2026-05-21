# Filesystem

This guide describes where things live in the proto-language repo: source code, tests, examples, runtime outputs, and how persistent data is split between proto-language and the `proto-tools` submodule.

For *what* the components do, see [`CLAUDE.md`](../CLAUDE.md) and the `.claude/skills/` reference. This doc is about *where files go*.

## Package layout: `proto_language/`

```
proto_language/
├── constraint/              Constraint functions, grouped into subpackages by domain
│   ├── constraint_registry.py        @constraint decorator + ConstraintRegistry
│   ├── sequence_composition/         Composition rules (GC content, kmers, length, homopolymers)
│   ├── sequence_alignment/           Alignment-based scoring
│   ├── sequence_annotation/          Chromatin / regulatory / CRISPR / promoter scorers
│   ├── sequence_scoring/             Language-model perplexity (forward + gradient variants)
│   ├── protein_structure/            Structure-quality metrics from folded predictions
│   ├── protein_quality/              Complexity, repetitiveness, diversity, domain hits
│   ├── rna_secondary_structure/      Base-pair / motif / feature similarity
│   └── rna_splicing/                 Splice-site usage, intron boundaries
├── core/                    Data-model + base ABCs (Sequence → Segment → Construct → Program)
│   ├── sequence.py                   Typed sequence + optional logits, structure, metadata bags
│   ├── segment.py                    Groups proposal/result sequences for one design region
│   ├── construct.py                  Ordered list of Segments
│   ├── program.py                    Composes optimizer stages, runs the design loop
│   ├── constraint.py                 Constraint ABC + ConstraintOutput types
│   ├── generator.py                  Generator ABC + GeneratorInputType
│   └── optimizer.py                  Optimizer ABC
├── generator/               Generator implementations (one module per generator + registry)
│   ├── generator_registry.py         @generator decorator + GeneratorRegistry
│   └── <one .py per generator>       Mutation-based, masked-LM, causal-LM, inverse-folding, …
├── optimizer/               Optimizer implementations (one module per optimizer + registry)
│   ├── optimizer_registry.py         @optimizer decorator + OptimizerRegistry
│   ├── <one .py per optimizer>       Monte Carlo, beam search, rejection sampling, cycling, gradient
│   └── constraint_compiler/          Compiled-constraint grouping for batched tool calls
└── utils/                   Shared infrastructure: base ABCs (BaseConfig, BaseRegistry, ConfigField),
                              the export chain, logging, serialization, ML-optimizer wrappers
```

## Tests: `tests/`

Mirrors the package layout under `tests/language_tests/`. `conftest.py` patches mock generators so optimizer tests don't load real model weights; `dummy_data/` carries small fixed inputs (PDBs, sequences). See [`tests/README.md`](../tests/README.md) for the canonical marker reference.

## Examples: `examples/`

Reference content shipped with the repo. Conventions:

```
examples/
├── bin/         Standalone analysis / utility scripts. Run directly, not imported.
├── bindcraft/   Binder-design example programs.
├── germinal/    Antibody-generation pipeline + presets and PDBs.
├── data/        Immutable reference datasets (HMMs, genomic context tracks, peak
│                files, training proteins, etc.) used by example programs.
├── jsons/       Declarative JSON program definitions, consumed via
│                Program.from_json(). Reproducible seed points for end-to-end runs.
└── scripts/     Larger multi-file workloads: regulatory design, intron design,
                 symmetric proteins, inverse-folding ensembles, gradient-based
                 protein hallucination, multi-stage design pipelines.
```

The `toy_json` fixture in `tests/conftest.py` loads `examples/jsons/toy.json`, so the JSON examples are exercised by the test suite.

## Logs and run outputs

| Path | Producer | Tracked? |
|---|---|---|
| `logs/pytest_*.log` | `setup_test_logging` in `tests/conftest.py` | gitignored |
| `logs/<program>_*.log` | `setup_logging()` called from a `Program.run()` | gitignored |
| `tests/logs/` | pytest reserved location | gitignored |
| User-chosen path via `Program.export(path=...)` | DSL itself | user-managed |

`Program.export(path=..., format="csv"|"xlsx")` writes a folder containing four tables (`sequences`, `constraints`, `constructs`, `optimization`), plus `sequences.fasta` and an `assets/` sidecar directory. The default is no path — the user always chooses where exports land.

## Persistent storage (model weights, tool envs)

proto-language has **no proto-language-specific environment variables for storage**. All persistent data is owned by the `proto-tools` submodule:

| Variable | Owned by | What it controls |
|---|---|---|
| `PROTO_HOME` (default `~/.proto/`) | proto-tools | Top-level cache root: model weights, tool envs, micromamba |
| `PROTO_MODEL_CACHE` | proto-tools | Override just the model-weight directory (useful for shared team caches) |
| `PROTO_{TOOL}_WEIGHTS_DIR` | proto-tools | Per-tool weight override |
| `HF_TOKEN` | proto-tools (gated models) | HuggingFace auth for gated models |

Full reference: [`proto-tools/notes/storage.md`](../proto-tools/notes/storage.md). Set these in your shell once; proto-language picks them up automatically because every generator that loads weights routes through proto-tools.
