import pytest

from nemd import cru


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
