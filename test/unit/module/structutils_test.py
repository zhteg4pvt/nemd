import numpy as np
import pytest

from nemd import structutils


class TestGriddedConf:

    @pytest.fixture
    def conf(self, smiles):
        mol = structutils.GriddedMol.MolFromSmiles(smiles)
        mol.EmbedMolecule()
        return mol.GetConformer()

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('aids,ignoreHs,expected',
                             [(None, False, [0, 0, 0]),
                              ([0, 1], False, [-0.40656618, 0.09144816, 0.]),
                              (None, True, [-0.00081616, 0.36637843, 0.])])
    def testCentroid(self, conf, aids, ignoreHs, expected):
        centroid = conf.centroid(aids=aids, ignoreHs=ignoreHs)
        np.testing.assert_almost_equal(centroid, expected)
