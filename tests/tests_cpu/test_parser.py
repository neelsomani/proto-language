import json
import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from api.parser import GPLParser


@pytest.fixture(scope="session")
def toy_json():
    with open(os.path.join(os.path.dirname(__file__), "dummy_data/toy.json")) as f:
        return json.load(f)


def test_gpl_parser_runs(toy_json):
    parser = GPLParser(toy_json)
    program = parser.parse()
    sequence_history = program.run()
    assert isinstance(sequence_history, list)
    assert len(sequence_history) > 0
