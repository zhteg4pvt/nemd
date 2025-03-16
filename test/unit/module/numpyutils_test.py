import numpy as np
import pytest

from nemd import numpyutils
from nemd import pytestutils


class TestFunc:

    @pytest.fixture
    def array(self, values, mval):
        return numpyutils.IntArray(values=values, mval=mval)

    @pytest.mark.parametrize('values,mval,expected', [([2, 5], 0, 6),
                                                      (None, 10, 11)])
    def testNew(self, values, mval, array, expected):
        assert expected == array.shape[0]

    @pytest.mark.parametrize('values,mval,expected',
                             [([1, 2, 5], 0, [1, 2, 5]), (None, 2, [])])
    def testValues(self, values, array, expected):
        np.testing.assert_array_equal(array.values, expected)

    @pytestutils.Raises
    @pytest.mark.parametrize('values,mval', [([1, 2, 5], 0)])
    @pytest.mark.parametrize("to_index,expected", [([2, 5], [1, 2]),
                                                   ([2, 3], KeyError)])
    def testIndex(self, array, to_index, expected):
        np.testing.assert_array_equal(array.index(to_index), expected)
