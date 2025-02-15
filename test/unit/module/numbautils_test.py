import numpy as np
import pytest

from nemd import numpyutils


class TestFunc:

    @pytest.fixture
    def array(self):
        return numpyutils.IntArray(10)

    def testValues(self, array):
        assert not array.values.any()
        array[[1, 4, 7]] = True
        np.testing.assert_array_equal(array.values, [1, 4, 7])

    @pytest.mark.parametrize("values,expected,raise_type,is_raise",
                             [([2, 5], [1, 2], None, False),
                              ([2, 3], None, KeyError, True)])
    def testIndex(self, values, expected, array, check_raise):
        array[[1, 2, 5]] = True
        with check_raise():
            np.testing.assert_array_equal(array.index(values), expected)