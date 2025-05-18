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


@pytest.mark.parametrize('smiles', ['O'])
class TestPackedConf:

    @pytest.fixture
    def conf(self, smiles):
        mol = structutils.PackedMol.MolFromSmiles(smiles)
        mol.EmbedMolecule()
        struct = structutils.PackedStruct.fromMols([mol])
        struct.setBox()
        return next(struct.conf)

    def testSetConformer(self, conf):
        conf.setConformer()
        assert conf.mol.struct.dist.gids.all()

    @pytest.mark.parametrize('seed,aids', [(1234, [0, 1])])
    def testRotateRandomly(self, conf, seed, aids):
        oxyz = conf.GetPositions()
        obond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        conf.rotateRandomly(seed=seed)
        assert not (oxyz == conf.GetPositions()).all()
        bond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        np.testing.assert_almost_equal(bond, obond)

    def testRest(self, conf):
        conf.setConformer()
        oxyz = conf.GetPositions()
        conf.reset()
        assert not (oxyz == conf.GetPositions()).all()
