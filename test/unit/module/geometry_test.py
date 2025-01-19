import numpy as np
import pytest

from nemd import geometry


class TestGeometry:

    @pytest.mark.parametrize("xyzs,expected",
                             [([[0, 0, 0], [1, 1, 1]], 1.7321),
                              ([[1, 2, 3], [1, 1, 1]], 2.2361)])
    def test_distance(self, xyzs, expected):
        dist = geometry.distance(np.array(xyzs))
        np.testing.assert_almost_equal(dist, expected, 4)

    @pytest.mark.parametrize("xyzs,expected",
                             [([[1, 0, 0], [1, 0, 1]], 45),
                              ([[3, 2, 3], [1, 1, 1]], 10.0250)])
    def test_angle_vs(self, xyzs, expected):
        angle = geometry.angle_vs(np.array(xyzs))
        np.testing.assert_almost_equal(angle, expected, 4)

    @pytest.mark.parametrize("xyzs,expected",
                             [([[1, 0, 0], [0, 0, 0], [0, 1, 0]], 90),
                              ([[4, 3, 4], [1, 1, 1], [2, 2, 2]], 10.0250)])
    def test_angle(self, xyzs, expected):
        angle = geometry.angle(np.array(xyzs))
        np.testing.assert_almost_equal(angle, expected, 4)

    @pytest.mark.parametrize(
        "xyzs,expected",
        [([[4, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 3]], 90),
         ([[1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0]], 180)])
    def test_dihedral(self, xyzs, expected):
        angle = geometry.dihedral(np.array(xyzs))
        np.testing.assert_almost_equal(angle, expected, 4)
