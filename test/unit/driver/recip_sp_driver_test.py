from unittest import mock

import numpy as np
import pandas as pd
import pytest
import recip_sp_driver as driver

from nemd import plotutils


class TestDriverParser:

    @pytest.mark.parametrize("miller_indices,valid", [(['1', '2'], True),
                                                      (['1'], False),
                                                      (['1', '1', '1'], False),
                                                      (['0', '0'], False)])
    def testParseArgs(self, miller_indices, valid):
        parser = driver.DriverParser()
        argv = [driver.FLAG_MILLER_INDICES] + miller_indices
        with mock.patch.object(parser, 'error'):
            parser.parse_args(argv)
            assert not valid == parser.error.called


class TestReciprocal:

    @pytest.fixture
    def recip(self, miller):
        with plotutils.get_pyplot() as plt:
            ax = plt.figure().add_subplot(111)
            vecs = [[1.5, 0.8660254], [1.5, -0.8660254]]
            vecs = pd.DataFrame(vecs, index=['x', 'y'], columns=['a1', 'a2'])
            return driver.Reciprocal(ax, vecs=vecs, miller=miller)

    @pytest.mark.parametrize(('miller', 'vecs'),
                             [([0, 1], [[0.0, 0.866025], [0.0, -0.866025]]),
                              ([1, 0], [[1.5, 0.0], [1.5, -0.0]]),
                              ([2, 11], [[3.0, 9.526279], [3.0, -9.526279]])])
    def testSetMiller(self, recip, vecs):
        recip.setMiller()
        np.testing.assert_almost_equal(vecs, recip.m_vecs, decimal=6)

    @pytest.mark.parametrize(('miller', 'vec'),
                             [([0, 1], [0.866025, -0.866025]),
                              ([1, 0], [1.5, 1.5]),
                              ([2, 11], [12.526279, -6.526279])])
    def testSetMillerError(self, recip, vec):
        recip.setMiller()
        recip.setVec()
        np.testing.assert_almost_equal(vec, recip.vec, decimal=6)

    @pytest.mark.parametrize(
        ('miller', 'lim', 'shape'),
        [([0, 1], [-10.5, 10.5, -10.5, 10.5], [173, 173]),
         ([1, 0], [-10.5, 10.5, -10.5, 10.5], [173, 173]),
         ([2, 11], [-18.0, 18.0, -18.0, 18.0], [469, 469])])
    def testSetGridsAndLim(self, recip, lim, shape):
        recip.setGridsAndLim()
        np.testing.assert_almost_equal(lim, recip.xlim + recip.ylim)
        np.testing.assert_almost_equal(shape, [len(x) for x in recip.grids])

    @pytest.mark.parametrize(('miller', 'lim'),
                             [([0, 1], [-10.5, 10.5, -10.5, 10.5]),
                              ([1, 0], [-10.5, 10.5, -10.5, 10.5]),
                              ([2, 11], [-18.0, 18.0, -18.0, 18.0])])
    def testPlotGrids(self, recip, lim):
        recip.setGridsAndLim()
        recip.plotGrids()
        assert 1 == len(recip.ax.collections)

    @pytest.mark.parametrize(('miller'), [([0, 1])])
    def testQuiver(self, recip):
        recip.quiver(recip.vecs.a1)
        assert 1 == len(recip.quivers)

    @pytest.mark.parametrize(('miller'), [([0, 1])])
    def testAnnotate(self, recip):
        recip.setMiller()
        recip.annotate(recip.m_vecs.a2)
        with plotutils.get_pyplot() as plt:
            assert isinstance(recip.ax.get_children()[0], plt.Annotation)

    @pytest.mark.parametrize(('miller'), [([0, 1])])
    def testSetLegend(self, recip):
        recip.quiver(recip.vecs.a1)
        recip.legend()
        assert recip.ax.get_legend()


class TestReal:

    @pytest.fixture
    def real(self, miller):
        with plotutils.get_pyplot() as plt:
            ax = plt.figure().add_subplot(111)
            vecs = np.array([[1.5, 0.8660254], [1.5, -0.8660254]]).T
            vecs = pd.DataFrame(vecs, index=['x', 'y'], columns=['a1', 'a2'])
            return driver.Real(ax, vecs=vecs, miller=miller)

    @pytest.mark.parametrize(('miller', 'vec'),
                             [([0, 1], [0.75, -1.2990381]), ([1, 1], [1.5, 0]),
                              ([1, 2], [1.5, 0.8660254]),
                              ([0.5, 2], [0.5769231, 0.599556]),
                              ([-1, 1], [0, -0.8660254])])
    def testSetVec(self, real, vec):
        real.setMiller()
        real.setVec()
        np.testing.assert_almost_equal(vec, real.vec)

    @pytest.mark.parametrize(('miller', 'factor', 'vec'),
                             [([0, 1], 0, [0, 0]), ([1, 1], 1, [1.5, 0]),
                              ([1, 2], 2, [3, 1.73205081]),
                              ([0.5, 2], 0.5, [0.28846154, 0.29977802])])
    def testGetNormal(self, real, factor, vec):
        real.setMiller()
        norm = real.getNormal(factor=factor)
        np.testing.assert_almost_equal(vec, norm)

    @pytest.mark.parametrize(
        ('miller', 'factor', 'points'),
        [([0, 1], 0, [[0., 0.], [1.5, 0.8660254]]),
         ([1, 1], 1, [[1.5, 0.8660254], [1.5, -0.8660254]]),
         ([1, 2], 2, [[3., 1.7320508], [6., -3.4641016]]),
         ([0.5, 2], 0.5, [[0.375, 0.21650635], [1.5, -0.8660254]])])
    def testGetPlane(self, real, factor, points):
        real.setMiller()
        plane = real.getPlane(factor=factor)
        np.testing.assert_almost_equal(points, plane)

    @pytest.mark.parametrize(('miller', 'factor'), [([0, 1], 0), ([1, 1], 1),
                                                    ([1, 2], 2),
                                                    ([0.5, 2], 0.5)])
    def testPlotPlane(self, real, factor):
        real.setMiller()
        real.setGridsAndLim()
        real.plotPlane(factor=factor)
        assert 1 == len(real.ax.lines) or 1 == len(real.ax.collections)
