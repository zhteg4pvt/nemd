import numpy as np
import pytest

from nemd import geometry


class TestPoint:

    @pytest.fixture()
    def pnt(self, xyzs):
        return geometry.Point(xyzs)

    @pytest.mark.parametrize("xyzs,expected",
                             [([[0, 0, 0], [1, 1, 1]], 1.7321),
                              ([[1, 2, 3], [1, 1, 1]], 2.2361)])
    def test_distance(self, pnt, expected):
        np.testing.assert_almost_equal(pnt.distance(), expected, 4)

    @pytest.mark.parametrize("xyzs,expected",
                             [([[1, 0, 0], [1, 0, 1]], 45),
                              ([[3, 2, 3], [1, 1, 1]], 10.0250)])
    def test_angle_vs(self, xyzs, expected):
        angle = geometry.Point.angle_vs(np.array(xyzs))
        np.testing.assert_almost_equal(angle, expected, 4)

    @pytest.mark.parametrize("xyzs,expected",
                             [([[1, 0, 0], [0, 0, 0], [0, 1, 0]], 90),
                              ([[4, 3, 4], [1, 1, 1], [2, 2, 2]], 10.0250)])
    def test_angle(self, pnt, expected):
        np.testing.assert_almost_equal(pnt.angle(), expected, 4)

    @pytest.mark.parametrize(
        "xyzs,expected",
        [([[4, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 3]], 90),
         ([[1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0]], 180)])
    def test_dihedral(self, pnt, expected):
        np.testing.assert_almost_equal(pnt.dihedral(), expected, 4)
