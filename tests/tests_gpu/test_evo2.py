import sys
sys.path.append('.')
from language.generator import Evo2Generator


def test_evo2_sampling():
    prompts = ['ATCG', 'AAAA']
    evo2_generator = Evo2Generator(prompt_seqs=prompts, n_tokens=100)

    evo2_outputs = evo2_generator.register()
    assert len(evo2_outputs) == len(prompts)

    evo2_generator.sample()

    for evo2_output in evo2_outputs:
        assert evo2_output.sequence is not None
