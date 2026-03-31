# Running Tests

The test suite has two tiers:

| Tier | Command | What runs | Marker |
|------|---------|-----------|--------|
| **Unit** | `pytest` | Fast, fully mocked, no I/O | *(none needed)* |
| **Integration** | `pytest --integration` | Requires external tools (MAFFT, etc.) | `@pytest.mark.integration` |

- `--all` includes integration and slow tests

## Test Markers

When writing tests, use these markers:
```python
import pytest

# Mark a test as requiring GPU
@pytest.mark.uses_gpu
def test_gpu_function():
    ...

# Mark a test as slow (will be skipped unless --all or --slow specified)
@pytest.mark.slow
def test_long_running_operation():
    ...

# Mark a test as requiring external tools (MAFFT, BLAST, etc.)
@pytest.mark.integration
def test_with_mafft():
    ...

# CPU tests don't need explicit marking (auto-marked)
def test_cpu_function():
    ...
```

- By default, all tests are marked as CPU-only unless explicitly marked with `@pytest.mark.uses_gpu`.
- By default, all tests are marked as fast unless explicitly marked with `@pytest.mark.slow`.
