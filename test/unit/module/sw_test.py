import pytest
from rdkit import Chem

from nemd import stillinger
from nemd import sw


class TestFunc:

    STRUCT = stillinger.Struct()
    STRUCT.addMol(Chem.MolFromSmiles('[Si]'))

    @pytest.mark.parametrize('elements,struct,expected',
                             [(('Si', ), None, 'Si.sw'),
                              (None, STRUCT, 'Si.sw'), (('Ar', ), None, None)])
    def testGetFile(self, elements, struct, expected):
        file = sw.get_file(elements, struct=struct)
        assert file is None if expected is None else file.endswith('.sw')
