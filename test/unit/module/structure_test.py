import pytest

from nemd import np
from nemd import structure


class TestConformer:

    @pytest.fixture
    def conf(self, mol):
        return structure.Conformer(mol=mol)

    @pytest.mark.parametrize('smiles,expected', [('O', True), (None, False)])
    def testHasOwningMol(self, conf, expected):
        assert expected == conf.HasOwningMol()

    @pytest.mark.parametrize('smiles', ['O', None])
    def testGetOwningMol(self, conf, mol):
        assert mol == conf.GetOwningMol()

    @pytest.mark.parametrize('smiles', ['O', None])
    @pytest.mark.parametrize('xyz', [np.zeros((3, 3))])
    def testSetPositions(self, conf, xyz):
        conf.setPositions(xyz)
        np.testing.assert_equal(xyz, conf.GetPositions())

@pytest.mark.parametrize('smiles', ['O'])
class TestMol:

    @pytest.mark.parametrize('polym,vecs',
                             [(None, None), (False, None),
                              (True, (5.43, 5.43, 5.43, 90.0, 90.0, 90.0))])
    def testSetUp(self, mol, polym, vecs):
        np.testing.assert_equal(mol.gids, [0, 1, 2])
        assert (False, False) == (mol.polym, mol.vecs)
        mol = structure.Mol(mol, polym=polym, vecs=vecs)
        expected = tuple(False if x is None else x for x in [polym, vecs])
        assert expected == (mol.polym, mol.vecs)
        mol = structure.Mol(mol)
        assert expected == (mol.polym, mol.vecs)

    def testSetConformers(self, mol):
        assert 0 == len(mol.confs)
        mol.EmbedMolecule()
        assert 1 == len(mol.confs)
        mol.setConfs(mol.confs)
        assert 2 == len(mol.confs)
        np.testing.assert_equal(mol.confs[-1].gids, [3, 4, 5])
        assert 1 == mol.confs[-1].gid

