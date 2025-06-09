import pytest

from nemd import cru
from nemd import np


class TestMoiety:

    @pytest.fixture
    def moiety(self, mol):
        return cru.Moiety(mol)

    @pytest.mark.parametrize('smiles,role_id,expected',
                             [('O', 1, 0), ('C*', 1, 0), ('C*', 0, 1),
                              ('*C*', 0, 2), ('C*C', 0, cru.MoietyError),
                              ('C[*:1]', 1, 1)])
    def testGetCapping(self, moiety, role_id, expected, raises):
        with raises:
            assert expected == len(moiety.getCapping(role_id))

    @pytest.mark.parametrize('smiles,expected', [('O', 0), ('C*', 1),
                                                 ('*C*', 2), ('C*C', 1)])
    def testStars(self, moiety, expected):
        assert expected == len(moiety.stars)

    @pytest.mark.parametrize('smiles,expected',
                             [('O', 'regular'), ('C*', 'terminator'),
                              ('*C*', 'monomer'), ('*C[*:1]', 'monomer'),
                              ('[*:1]C[*:1]', 'initiator'),
                              ('[*:1]C(*)[*:1]', cru.MoietyError)])
    def testRole(self, moiety, expected, raises):
        with raises:
            assert expected == moiety.role


class TestMol:

    ROLE = ('regular', 'initiator', 'terminator', 'monomer')

    @pytest.fixture
    def mol(self, smiles):
        return cru.Mol.MolFromSmiles(smiles, united=False)

    @pytest.mark.parametrize('smiles,expected', [('O', [1, 0, 0, 0]),
                                                 ('C[*:1]', [0, 1, 0, 0]),
                                                 ('C*', [0, 0, 1, 0]),
                                                 ('*C*', [0, 0, 0, 1]),
                                                 ('C.*C*', [1, 0, 0, 1]),
                                                 ('O*.C*', [0, 0, 2, 0])])
    def testSetMoieties(self, mol, expected):
        mol.setMoieties()
        assert expected == [len(mol.moieties[x]) for x in self.ROLE]

    @pytest.mark.parametrize('smiles,allow_reg,expected',
                             [('O', True, None), ('O', False, cru.MoietyError),
                              ('*C*', False, None),
                              ('O.*C*', False, cru.MoietyError)])
    def testCheckReg(self, mol, allow_reg, expected, raises):
        mol.allow_reg = allow_reg
        mol.setMoieties()
        with raises:
            mol.checkReg()

    @pytest.mark.parametrize('smiles,expected',
                             [('*C*', 0), ('C*', 0),
                              ('C[*:1]', 1), ('C*.C*', 1),
                              ('C[*:1].C[*:1]', cru.MoietyError)])
    def testSetInitiator(self, mol, expected, raises):
        mol.setMoieties()
        with raises:
            mol.setInitiator()
            assert expected == len(mol.moieties['initiator'])

    @pytest.mark.parametrize('smiles,expected',
                             [('*C*', 0), ('C*', 1), ('C[*:1]', 0),
                              ('C*.C*', 1), ('C*.C*.C*', cru.MoietyError)])
    def testSetTerminator(self, mol, expected, raises):
        mol.setMoieties()
        mol.setInitiator()
        with raises:
            mol.setTerminator()
            assert expected == len(mol.moieties['terminator'])

    @pytest.mark.parametrize('smiles,expected',
                             [('*C*', 1), ('*C*.*O*', 2), ('*C[*:1]', 1),
                              ('[*:1]C[*:1]', 0),
                              ('C*', 0)])
    def testSetMonomer(self, mol, expected):
        mol.setMoieties()
        mol.setMonomer()
        assert expected == len(mol.moieties['monomer'])
        for monomer in mol.moieties['monomer']:
            nums = [len(monomer.getCapping(x)) for x in [0, 1]]
            np.testing.assert_equal(nums, [1, 1])

    @pytest.mark.parametrize('smiles,expected', [('O', 'O'),
                                                 ('C[*:1]', 'C[*:1]'),
                                                 ('C*', '*C'),
                                                 ('C.*C*', '*C*.C')])
    def testGetSmiles(self, mol, expected):
        mol.setMoieties()
        assert expected == mol.getSmiles()
