import sys
sys.path.append('.')

import language
from language.tools.structure_prediction import esmfold_protein_sequence


if __name__ == '__main__':
    esmfold_output = esmfold_protein_sequence(sys.argv[1])

    with open('design.pdb', 'w') as f:
        f.write(esmfold_output['pdb_output'] + '\n')
