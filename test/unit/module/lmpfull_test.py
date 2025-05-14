import pytest

from nemd import lmpfull
from nemd import np
from nemd import oplsua

PARSER = oplsua.Parser.get()


class TestMass:

    @pytest.fixture
    def atoms(self, indices, ff=PARSER):
        return ff.atoms.loc[indices]

    @pytest.mark.parametrize('indices,expected', [([77, 78, 78], (3, 2))])
    def testFromAtoms(self, atoms, expected):
        assert expected == lmpfull.Mass.fromAtoms(atoms).shape


class TestId:

    @pytest.fixture
    def ids(self, mol):
        return lmpfull.Mol(mol).ids

    @pytest.mark.parametrize('smiles,expected', [('O', (3, 3))])
    def testFromAtoms(self, ids, expected):
        assert expected == ids.shape

    @pytest.mark.parametrize('smiles,gids,expected',
                             [('O', [4, 5, 6], (3, 3))])
    def testToNumpy(self, ids, gids, expected):
        assert expected == ids.to_numpy(np.array(gids)).shape


class TestBond:

    @pytest.fixture
    def bonds(self, mol):
        return lmpfull.Mol(mol).bonds

    @pytest.mark.parametrize('smiles,expected', [('O', (2, 3))])
    def testFromAtoms(self, bonds, expected):
        assert expected == bonds.shape

    @pytest.mark.parametrize('smiles,expected', [('O', [(0, 1), (0, 2)])])
    def testGetPair(self, bonds, expected):
        assert expected == bonds.getPairs()


class TestAngle:

    @pytest.fixture
    def angles(self, mol):
        return lmpfull.Mol(mol).angles

    @pytest.mark.parametrize('smiles,expected', [('O', (1, 4)),
                                                 ('CC(C)C', (2, 4))])
    def testFromAtoms(self, angles, expected):
        assert expected == angles.shape

    @pytest.mark.parametrize(
        'angle,improper,expected',
        [([[297, 1, 0, 2]], None, (1, 4)),
         ([[304, 0, 1, 2], [304, 0, 1, 3], [304, 2, 1, 3]], [[30, 0, 2, 1, 3]],
          (2, 4))])
    def testDropLowest(self, angle, improper, expected):
        angles = lmpfull.Angle(angle)
        imprps = lmpfull.Improper(improper) if improper else lmpfull.Improper()
        angles.dropLowest(imprps.getAngles(), PARSER.angles.ene.to_dict())
        assert expected == angles.shape

    @pytest.mark.parametrize('smiles,expected', [('O', (4, 1)),
                                                 ('CC(C)C', (4, 3))])
    def testRow(self, angles, expected):
        assert expected == angles.row.shape
