from typing import Tuple

import sys
sys.path.append('.')
import language

from language.base import ProgramConstraint, ProgramSequence
from language.constraint import (
    esmfold_plddt_constraint,
    esmfold_ptm_constraint,
    protein_globularity_constraint,
    protein_symmetry_ring_constraint,
)
from language.generator import ProgramMCMCGenerator, UniformMutationGenerator
from language.program import Program


MONOMER_LENGTH = 30
N_SYMMETRIC_UNITS = 4
N_STEPS = 10_000


################
## Generators ##
################

uniform_generator = UniformMutationGenerator(
    sequence_length=MONOMER_LENGTH,
    sequence_type='protein',
)

#############
## Outputs ##
#############

esm2_outputs = uniform_generator.register()

#################
## Constraints ##
#################

esmfold_plddt = ProgramConstraint(
    inputs=list(esm2_outputs),
    scoring_function=esmfold_plddt_constraint,
    scoring_function_config={'n_replications': N_SYMMETRIC_UNITS},
)

esmfold_ptm = ProgramConstraint(
    inputs=list(esm2_outputs),
    scoring_function=esmfold_ptm_constraint,
    scoring_function_config={'n_replications': N_SYMMETRIC_UNITS},
)

symmetry = ProgramConstraint(
    inputs=list(esm2_outputs),
    scoring_function=protein_symmetry_ring_constraint,
    scoring_function_config={
        'n_replications': N_SYMMETRIC_UNITS,
        'all_to_all_protomer_symmetry': True,
    },
)

globularity = ProgramConstraint(
    inputs=list(esm2_outputs),
    scoring_function=protein_symmetry_ring_constraint,
    scoring_function_config={'n_replications': N_SYMMETRIC_UNITS},
)

#############
## Program ##
#############

def custom_logging(step: int, outputs: Tuple[ProgramSequence]) -> None:
    output = outputs[0]
    print(
        f"Iteration {step} | "
        f"sequence: {output._metadata['esmfolded_sequence']}, "
        f"pLDDT: {output._metadata['avg_plddt']}, "
        f"pTM: {output._metadata['ptm']}"
    )

program = Program(
    ebm_class=ProgramMCMCGenerator,
    constraints=[
        esmfold_plddt,
        esmfold_ptm,
        symmetry,
        globularity,
    ],
    constraint_weights=[20., 20., 1., 1.,],
    generators=[uniform_generator],
    num_steps=N_STEPS,
    track_step_size=1,
    temperature=2.,
    custom_logging=custom_logging,
)

sequence_history = program.run()

with open('design.pdb', 'w') as f:
    f.write(sequence_history[-1][0]._metadata['pdb_output'] + '\n')
