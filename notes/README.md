# Notes

Team-shared development docs for proto-language. These files capture setup guides, architecture decisions, platform quirks, and CI procedures; knowledge that **every developer** needs.

For personal discoveries (debugging patterns, tool quirks found during a session), use Claude's auto-memory instead of adding to these files. Only add to notes/ when the knowledge benefits the whole team.

## Index

- `batching.md`: Batching architecture across generator, language, tool, and GPU boundaries.
- `claude-code.md`: Skills, CI integration, and common workflows for the Claude Code layer.
- `dev.md`: Setup, submodule sync, CI checks, docs generation.
- `error-handling.md`: Raise-vs-soft-fail rules inside Constraint / Generator / Optimizer; `format_pydantic_error()`.
- `seeding.md`: Program/optimizer/generator/constraint seed hierarchy.
- `testing.md`: Markers, placement, templates per component type, conftest fixtures, mock scoring functions.
