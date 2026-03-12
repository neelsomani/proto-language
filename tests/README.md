# Running Tests

The test suite has three tiers:

| Tier | Command | What runs | Marker |
|------|---------|-----------|--------|
| **Unit** | `pytest` | Fast, fully mocked, no I/O | *(none needed)* |
| **Integration** | `pytest --integration` | Requires external tools (MAFFT, etc.) | `@pytest.mark.integration` |
| **E2E** | `pytest --e2e` | Starts real a cache + API server, makes HTTP requests | `@pytest.mark.e2e` |

- `--all` includes integration but NOT e2e (e2e requires live services)
- `--e2e` must be explicitly specified

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

# Mark a test as end-to-end (requires a cache + API server)
@pytest.mark.e2e
def test_api_health():
    ...

# CPU tests don't need explicit marking (auto-marked)
def test_cpu_function():
    ...
```

- By default, all tests are marked as CPU-only unless explicitly marked with `@pytest.mark.uses_gpu`.
- By default, all tests are marked as fast unless explicitly marked with `@pytest.mark.slow`.

## External Dependencies

The test suite automatically mocks external dependencies (a cache, databases) so unit tests run without setting up services. E2E tests (`tests/e2e_tests/`) override these mocks and start real services via session-scoped fixtures.
