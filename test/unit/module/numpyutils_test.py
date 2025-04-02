import numpy as np
import pytest

from nemd import numpyutils
from nemd import pytestutils


class TestFunc:

    @pytest.fixture
    def array(self, shape, on):
        return numpyutils.IntArray(shape=shape, on=on)

    @pytest.mark.parametrize('shape,on,expected', [(None, [2, 5], 6),
                                                   (10, None, 10)])
    def testNew(self, on, shape, array, expected):
        assert expected == array.shape[0]
        assert (on is not None) == array.any()

    @pytest.mark.parametrize('shape,on,expected',
                             [(None, [1, 2, 5], [1, 2, 5]), (2, None, [])])
    def testOn(self, on, array, expected):
        np.testing.assert_array_equal(array.on, expected)

    @pytestutils.Raises
    @pytest.mark.parametrize('shape,on', [(None, [1, 2, 5])])
    @pytest.mark.parametrize("to_index,expected", [([2, 5], [1, 2]),
                                                   ([2, 3], KeyError)])
    def testIndex(self, array, to_index, expected):
        np.testing.assert_array_equal(array.index(to_index), expected)
