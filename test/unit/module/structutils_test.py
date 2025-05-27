from unittest import mock

import numpy as np
import pytest
import scipy
from rdkit import Chem

from nemd import structutils


@pytest.mark.parametrize('smiles', ['O'])
class TestConf:

    @pytest.fixture
    def conf(self, smiles):
        mol = structutils.GriddedMol.MolFromSmiles(smiles)
        mol.EmbedMolecule()
        return mol.GetConformer()

    @pytest.mark.parametrize('aids,ignoreHs,expected',
                             [(None, False, [0, 0, 0]),
                              ([0, 1], False, [-0.40656618, 0.09144816, 0.]),
                              (None, True, [-0.00081616, 0.36637843, 0.])])
    def testCentroid(self, conf, aids, ignoreHs, expected):
        centroid = conf.centroid(aids=aids, ignoreHs=ignoreHs)
        np.testing.assert_almost_equal(centroid, expected)

    @pytest.mark.parametrize('axis,angle,expected', [('x', 180, [1, -1, -1]),
                                                     ('z', 180, [-1, -1, 1])])
    def testRotate(self, conf, axis, angle, expected):
        oxyz = conf.GetPositions()
        rotation = scipy.spatial.transform.Rotation.from_euler(axis, [angle],
                                                               degrees=True)
        conf.rotate(rotation)
        xyz = conf.GetPositions()
        for idx, factor in enumerate(expected):
            np.testing.assert_almost_equal(oxyz[:, idx], factor * xyz[:, idx])

    @pytest.mark.parametrize('vec', [(1, 2, 3)])
    def testTranslate(self, conf, vec):
        oxyz = conf.GetPositions()
        conf.translate(vec)
        xyz = conf.GetPositions()
        np.testing.assert_almost_equal(oxyz, xyz - vec)


@pytest.mark.parametrize('smiles,seed', [('O', 0)])
class TestPackedConf:

    @pytest.fixture
    def conf(self, smiles, seed, random_seed):
        mol = structutils.PackedMol.MolFromSmiles(smiles)
        mol.EmbedMultipleConfs(2, randomSeed=seed)
        struct = structutils.PackedStruct.fromMols([mol])
        struct.setBox()
        struct.setFrame()
        struct.dist.set([0, 1, 2])
        return list(struct.conf)[1]

    def testSetConformer(self, conf):
        conf.setConformer()
        assert conf.mol.struct.dist.gids.all()
        np.testing.assert_almost_equal(conf.GetPositions().mean(), 3.0663129)

    @pytest.mark.parametrize('gids', [([2])])
    def testCheckClash(self, conf, gids):
        assert conf.checkClash([3, 4, 5]) is None
        conf.setConformer()
        assert conf.checkClash([3, 4, 5]) is True

    @pytest.mark.parametrize('aids', [([0, 1])])
    def testRotateRandomly(self, conf, aids):
        oxyz = conf.GetPositions()
        obond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        conf.rotateRandomly()
        assert not (oxyz == conf.GetPositions()).all()
        bond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        np.testing.assert_almost_equal(bond, obond)

    def testRest(self, conf):
        conf.setConformer()
        oxyz = conf.GetPositions()
        conf.reset()
        assert not (oxyz == conf.GetPositions()).all()


@pytest.mark.parametrize('smiles,seed', [('CCCCC(CC)CC', 0)])
class TestGrownConf:

    @pytest.fixture
    def conf(self, smiles, seed, random_seed):
        mol = structutils.GrownMol.MolFromSmiles(smiles)
        mol.EmbedMultipleConfs(2, randomSeed=seed)
        struct = structutils.GrownStruct.fromMols([mol])
        struct.setBox()
        struct.setFrame()
        struct.dist.set([0, 1, 2])
        return list(struct.conf)[1]

    def testSetUp(self, conf):
        np.testing.assert_almost_equal(conf.init, [9, 10, 11])
        assert 5 == len(list(conf.frag.next()))
        assert 1 == len(conf.frags)

    def testGrow(self, conf):
        while conf.frags:
            conf.grow()
        assert 1 == conf.failed
        assert conf.mol.struct.dist.gids[conf.gids].all()

    def testSetConformer(self, conf):
        conf.setConformer()
        assert 6 == conf.mol.struct.dist.gids.on.size

    def testCentroid(self, conf):
        np.testing.assert_almost_equal(conf.centroid(),
                                       [2.64317873, -0.42000759, -0.21083656])

    @pytest.mark.parametrize('dihe,expected', [((0, 1, 2, 3), 6),
                                               ((1, 2, 3, 4), 5)])
    def testGetSwingAtoms(self, conf, dihe, expected):
        assert expected == len(conf.getSwingAtoms(dihe))

    def testReset(self, conf):
        while conf.frags:
            conf.grow()
        conf.reset()
        assert 0 == conf.failed
        assert len(conf.frags)
        assert 36 == len(conf.frag.vals)


@pytest.mark.parametrize('smiles,seed', [('O', 0)])
class TestGriddedMol:

    @pytest.fixture
    def mol(self, smiles, seed):
        mol = structutils.GriddedMol.MolFromSmiles(smiles)
        mol.EmbedMultipleConfs(2, randomSeed=seed)
        return mol

    @pytest.mark.parametrize('size,expected', [([10, 10, 10], 4)])
    def testRun(self, mol, size, expected):
        mol.run(np.array(size))
        assert expected == np.prod(mol.num) == len(mol.vecs)

    @pytest.mark.parametrize('size,num,expected', [([10, 10, 10], 20, (2, 19)),
                                                   ([6, 5, 5], 20, (2, 18))])
    def testSetConformers(self, mol, size, num, expected):
        mol.run(np.array(size))
        unused = mol.setConformers(np.random.rand(num, 3))
        assert expected == (len(mol.vectors), len(unused))

    def testSize(self, mol):
        np.testing.assert_almost_equal(mol.size, [5.62544856, 4.54986054, 4.])

    @pytest.mark.parametrize('size,expected', [([10, 10, 10], 1),
                                               ([6, 5, 5], 2)])
    def testBoxNum(self, mol, size, expected):
        mol.run(np.array(size))
        assert expected == mol.box_num


class TestPackedMol:

    @pytest.mark.parametrize('smiles,seed,expected', [('O', 0, (3,3))])
    def testSetUp(self, smiles, seed, expected):
        mol = structutils.PackedMol.MolFromSmiles(smiles)
        mol.EmbedMultipleConfs(2, randomSeed=seed)
        for conf in mol.confs:
            assert conf.oxyz is None
        for conf in structutils.PackedMol(mol).confs:
            assert expected == conf.oxyz.shape