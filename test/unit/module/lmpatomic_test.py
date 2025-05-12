import os.path
import types

import numpy as np
import pytest

from nemd import lmpatomic
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

    TABLE = table.TABLE.reset_index()

    @pytest.fixture
    def masses(self, indices):
        return lmpatomic.Mass.fromAtoms(self.TABLE.iloc[indices])

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
    def testFromLines(self, lines, tmp_dir):
        assert len(lines) == lmpatomic.Mass.fromLines(lines).shape[0]
