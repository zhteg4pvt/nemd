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


class TestMol:

    def