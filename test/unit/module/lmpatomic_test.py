import os

import numpy as np
import pytest

from nemd import lmpatomic
from nemd import numpyutils
from nemd import table


class TestBase:

    @pytest.fixture
    def base(self):
        return lmpatomic.Base()

    def testWriteCount(self, base, tmp_dir):
        with open('file', 'w') as fh:
            base.writeCount(fh)
        assert os.path.exists('file')

    @pytest.mark.parametrize('lines,expected', [(['1 2'], [0, 2])])
    def testFromLines(self, base, lines, expected):
        base = base.fromLines(lines)
        assert expected == [*base.index, *base.values]


class TestMass:

    ATOM = table.TABLE.reset_index()

    @pytest.fixture
    def masses(self, indices):
        return lmpatomic.Mass.fromAtoms(self.ATOM.iloc[indices])

    @pytest.mark.parametrize('indices', [([1, 2]), ([4, 2, 112])])
    def testFromAtoms(self, masses, indices):
        assert len(indices) == masses.shape[0]

    @pytest.mark.parametrize('indices,expected',
                             [([1, 2], ['H', 'He']),
                              ([4, 2, 112], ['Be', 'He', 'Cn'])])
    def testElement(self, masses, expected):
        np.testing.assert_equal(masses.element, expected)

    @pytest.mark.parametrize('indices', [([1, 2]), ([4, 2, 112])])
    def testWrite(self, masses, tmp_dir):
        with open('file', 'w') as fh:
            masses.write(fh)
        assert os.path.isfile('file')

    @pytest.mark.parametrize('lines',
                             [(['1 1.0080 # H #\n', '2 4.0030 # He #\n'])])
    def testFromLines(self, lines):
        assert len(lines) == lmpatomic.Mass.fromLines(lines).shape[0]


class TestId:

    @pytest.fixture
    def ids(self, tmol):
        return lmpatomic.Id.fromAtoms(tmol.GetAtoms())

    @pytest.mark.parametrize('smiles,expected',
                             [('O', [[0, 77], [1, 78], [2, 78]])])
    def testFromAtoms(self, ids, expected):
        np.testing.assert_equal(ids.values, expected)

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('gids,expected', [([0, 1, 2], [0, 1, 2]),
                                               ([11, 12, 13], [11, 12, 13])])
    def testToNumpy(self, ids, gids, expected):
        gids = ids.to_numpy(np.array(gids))[:, 0]
        np.testing.assert_equal(gids, expected)

    @pytest.mark.parametrize(
        'arrays,on,expected',
        [(([[1, 2]], [[3, 4]]), [1, 2, 4], [[1, 1], [3, 2]])])
    def testConcatenate(self, arrays, on, expected):
        type_map = numpyutils.IntArray(on=on)
        ids = lmpatomic.Id.concatenate(arrays, type_map=type_map)
        np.testing.assert_equal(ids.values, expected)
