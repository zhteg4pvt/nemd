import numpy as np
import pytest

from nemd import constants


class TestFunction:

    @pytest.mark.parametrize("from_unit,to_unit,expected",
                             [('fs', 'ps', 1E-3), ('ps', 'fs', 1E3),
                              ('metal', 'ps', 1)])
    def testTimeFac(self, from_unit, to_unit, expected):
        value = constants.time_fac(from_unit=from_unit, to_unit=to_unit)
        np.testing.assert_almost_equal(expected, value)
