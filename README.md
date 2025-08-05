# Proto Language

A framework for designing biological sequences (DNA, RNA, proteins) with constraint-based optimization.

## Installation

The package is now structured with `pyproject.toml`. Install with:

```bash
conda create --name proto-language python=3.11 -y
conda activate proto-language
pip install -e .
```

[MMseqs2](https://github.com/soedinglab/MMseqs2?tab=readme-ov-file#installation) is also needed:

```bash
# On Mac:
brew install mmseqs2
```

## Running the API

### Local Development

```bash
python api/start_dev.py
```

### Docker

```bash
docker-compose up
```

API will be available at http://localhost:8000

## Tests

```bash
pytest -sv
```

For CPU-specific tests:

```bash
pytest -sv tests/tests_cpu/
```

## Running the Toy Example

```bash
jupyter notebook
```

Open and run `notebooks/toy_example.ipynb`
