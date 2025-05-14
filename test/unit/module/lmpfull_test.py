import pytest
from rdkit import Chem

from nemd import lmpfull
from nemd import np
from nemd import oplsua
from nemd import parserutils

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


class TestImproper:

    @pytest.fixture
    def impropers(self, mol):
        return lmpfull.Mol(mol).impropers

    @pytest.mark.parametrize('smiles,expected', [('O', (0, 5)),
                                                 ('CC(C)C', (1, 5))])
    def testFromAtoms(self, impropers, expected):
        assert expected == impropers.shape

    @pytest.mark.parametrize('smiles,expected', [('O', 0), ('CC(C)C', 3)])
    def testGetPairs(self, impropers, expected):
        assert expected == len(impropers.getPairs())

    @pytest.mark.parametrize('smiles,expected', [('O', 0), ('CC(C)C', 1)])
    def testGetAngles(self, impropers, expected):
        assert expected == len(impropers.getAngles())


@pytest.mark.parametrize('smiles', [('CCC(C)C')])
class TestConformer:

    @pytest.fixture
    def conf(self, mol):
        mol = lmpfull.Mol(mol)
        mol.EmbedMolecule(randomSeed=1)
        return mol.GetConformer()

    def testIds(self, conf):
        assert (5, 3) == conf.ids.shape

    def testBonds(self, conf):
        assert (4, 3) == conf.bonds.shape

    def testAngles(self, conf):
        assert (3, 4) == conf.angles.shape

    def testDihedrals(self, conf):
        assert (2, 5) == conf.dihedrals.shape

    def testImpropers(self, conf):
        assert (1, 5) == conf.impropers.shape

    @pytest.mark.parametrize('aids,val', [((2, 3), 2.12)])
    def testSetBondLength(self, conf, aids, val):
        conf.setBondLength(aids, val)
        assert val == Chem.rdMolTransforms.GetBondLength(conf, *aids)

    @pytest.mark.parametrize('aids,val', [((0, 1, 2), 121)])
    def testSetAngleDeg(self, conf, aids, val):
        conf.setAngleDeg(aids, val)
        assert val == Chem.rdMolTransforms.GetAngleDeg(conf, *aids)

    @pytest.mark.parametrize('aids,val', [((0, 1, 2, 4), 171)])
    def testGetDihedralDeg(self, conf, aids, val):
        conf.setDihedralDeg(aids, val)
        measured = Chem.rdMolTransforms.GetDihedralDeg(conf, *aids)
        np.testing.assert_almost_equal(measured, val)

    @pytest.mark.parametrize('args,expected', [([], None), (['CC'], 1.526),
                                               (['CC', '1.6'], 1.6),
                                               (['CCC'], 112.4),
                                               (['CCC', '120'], 120),
                                               (['CCCC'], 141.3842856),
                                               (['CCCC', '30'], 30)])
    def testMeasure(self, conf, smiles, args, expected):
        args = [smiles, '-substruct'] + args if args else [smiles]
        options = parserutils.MolBase().parse_args(args)
        strt = lmpfull.Struct.fromMols([conf.GetOwningMol()], options=options)
        measured = next(strt.conf).measure()
        if expected is None:
            assert measured is None
        else:
            np.testing.assert_almost_equal(measured, expected)
