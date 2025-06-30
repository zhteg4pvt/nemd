import argparse
import os
from unittest import mock

import pytest
import recip_sp_driver as driver

from nemd import np
from nemd import plotutils


@pytest.mark.parametrize('args', [(['-miller_indices', '2', '4'])])
class TestRecipSp:

    @pytest.fixture
    def recip_sp(self, args, logger, tmp_dir):
        options = driver.Parser().parse_args(args)
        return driver.RecipSp(options, logger=logger)

    def testSetReal(self, recip_sp):
        recip_sp.setReal()
        np.testing.assert_almost_equal([[1.5, 0.8660254], [1.5, -0.8660254]],
                                       recip_sp.real.lat.T)

    def testSetRecip(self, recip_sp):
        recip_sp.setReal()
        recip_sp.setRecip()
        np.testing.assert_almost_equal(
            [[2.0943951, 3.62759873], [2.0943951, -3.62759873]],
            recip_sp.recip.lat.T)

    def testProduct(self, recip_sp):
        recip_sp.setReal()
        recip_sp.setRecip()
        recip_sp.product()
        recip_sp.logger.log.assert_called_with(
            'The real and reciprocal vectors are parallel to each other with '
            '2π being the dot product.')

    def testPlot(self, recip_sp):
        recip_sp.setReal()
        recip_sp.setRecip()
        recip_sp.plot()
        assert os.path.isfile(recip_sp.outfile)


class TestParser:

    @pytest.fixture
    def parser(self):
        parser = driver.Parser()
        parser.error = mock.Mock(side_effect=argparse.ArgumentTypeError)
        return parser

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller_indices', '1', '2'], (1, 2)),
         (['-miller_indices', '0', '0'], argparse.ArgumentTypeError)])
    def testMillerAction(self, parser, args, expected, raises):
        with raises:
            assert expected == parser.parse_args(args).miller_indices


@pytest.mark.parametrize('lat', [[[1.5, 1.5], [0.8660254, -0.8660254]]])
class TestRecip:

    @pytest.fixture
    def recip(self, lat, args, logger):
        options = driver.Parser().parse_args(args)
        with plotutils.pyplot() as plt:
            return driver.Recip(lat,
                                ax=plt.figure().add_subplot(111),
                                options=options,
                                logger=logger)

    @pytest.mark.parametrize('args,expected',
                             [(['-miller_indices', '1', '1'],
                               [[1.5, 0.8660254], [1.5, -0.8660254], [3, 0]])])
    def testSetUp(self, recip, expected):
        to_compare = [recip.scaled.a1, recip.scaled.a2, recip.vec]
        np.testing.assert_almost_equal(expected, to_compare)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller_indices', '1', '1'
           ], 'The reciprocal space vector [3. 0.] has a norm of 3')])
    def testLogNorm(self, recip, expected):
        recip.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected',
                             [(['-miller_indices', '1', '1'], 9)])
    def testPlot(self, recip, expected):
        recip.plot(recip.ax)
        assert expected == len(recip.ax._children)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller_indices', '1', '1'
           ], [1, (-9.0, 9.0), (-5.1961524, 5.1961524), 'Reciprocal Space'])])
    def testGrid(self, recip, expected):
        recip.grid()
        assert expected == [
            len(recip.ax.collections),
            recip.ax.get_xlim(),
            recip.ax.get_ylim(),
            recip.ax.get_title()
        ]

    @pytest.mark.parametrize('args,expected',
                             [(['-miller_indices', '1', '1'], 85)])
    def testGrids(self, recip, expected):
        assert expected == recip.grids.shape[0]

    @pytest.mark.parametrize('args', [['-miller_indices', '1', '1']])
    @pytest.mark.parametrize('pnt,expected', [([-9, -5.1961524], True),
                                              ([9, 5.1961524], True),
                                              ([9, 5.1961525], True),
                                              ([9, 15.2], False),
                                              ([-9.001, 5.1961525], False)])
    def testCrop(self, recip, pnt, expected):
        assert expected == recip.crop(np.array([pnt])).any()


#     @pytest.mark.parametrize(('miller', 'vec'),
#                              [([0, 1], [0.866025, -0.866025]),
#                               ([1, 0], [1.5, 1.5]),
#                               ([2, 11], [12.526279, -6.526279])])
#     def testSetMillerError(self, recip, vec):
#         recip.setMiller()
#         recip.setVec()
#         np.testing.assert_almost_equal(vec, recip.vec, decimal=6)
#
#     @pytest.mark.parametrize(
#         ('miller', 'lim', 'shape'),
#         [([0, 1], [-10.5, 10.5, -10.5, 10.5], [173, 173]),
#          ([1, 0], [-10.5, 10.5, -10.5, 10.5], [173, 173]),
#          ([2, 11], [-18.0, 18.0, -18.0, 18.0], [469, 469])])
#     def testSetGridsAndLim(self, recip, lim, shape):
#         recip.setGridsAndLim()
#         np.testing.assert_almost_equal(lim, recip.xlim + recip.ylim)
#         np.testing.assert_almost_equal(shape, [len(x) for x in recip.grids])
#
#     @pytest.mark.parametrize(('miller', 'lim'),
#                              [([0, 1], [-10.5, 10.5, -10.5, 10.5]),
#                               ([1, 0], [-10.5, 10.5, -10.5, 10.5]),
#                               ([2, 11], [-18.0, 18.0, -18.0, 18.0])])
#     def testPlotGrids(self, recip, lim):
#         recip.setGridsAndLim()
#         recip.plotGrids()
#         assert 1 == len(recip.ax.collections)
#
#     @pytest.mark.parametrize(('miller'), [([0, 1])])
#     def testQuiver(self, recip):
#         recip.quiver(recip.vecs.a1)
#         assert 1 == len(recip.quivers)
#
#     @pytest.mark.parametrize(('miller'), [([0, 1])])
#     def testAnnotate(self, recip):
#         recip.setMiller()
#         recip.annotate(recip.m_vecs.a2)
#         with plotutils.pyplot() as plt:
#             assert isinstance(recip.ax.get_children()[0], plt.Annotation)
#
#     @pytest.mark.parametrize(('miller'), [([0, 1])])
#     def testSetLegend(self, recip):
#         recip.quiver(recip.vecs.a1)
#         recip.legend()
#         assert recip.ax.get_legend()
#
#
# class TestReal:
#
#     @pytest.fixture
#     def real(self, miller):
#         with plotutils.pyplot() as plt:
#             ax = plt.figure().add_subplot(111)
#             vecs = np.array([[1.5, 0.8660254], [1.5, -0.8660254]]).T
#             vecs = pd.DataFrame(vecs, index=['x', 'y'], columns=['a1', 'a2'])
#             return driver.Real(ax, vecs=vecs, miller=miller)
#
#     @pytest.mark.parametrize(('miller', 'vec'),
#                              [([0, 1], [0.75, -1.2990381]), ([1, 1], [1.5, 0]),
#                               ([1, 2], [1.5, 0.8660254]),
#                               ([0.5, 2], [0.5769231, 0.599556]),
#                               ([-1, 1], [0, -0.8660254])])
#     def testSetVec(self, real, vec):
#         real.setMiller()
#         real.setVec()
#         np.testing.assert_almost_equal(vec, real.vec)
#
#     @pytest.mark.parametrize(('miller', 'factor', 'vec'),
#                              [([0, 1], 0, [0, 0]), ([1, 1], 1, [1.5, 0]),
#                               ([1, 2], 2, [3, 1.73205081]),
#                               ([0.5, 2], 0.5, [0.28846154, 0.29977802])])
#     def testGetNormal(self, real, factor, vec):
#         real.setMiller()
#         norm = real.getNormal(factor=factor)
#         np.testing.assert_almost_equal(vec, norm)
#
#     @pytest.mark.parametrize(
#         ('miller', 'factor', 'points'),
#         [([0, 1], 0, [[0., 0.], [1.5, 0.8660254]]),
#          ([1, 1], 1, [[1.5, 0.8660254], [1.5, -0.8660254]]),
#          ([1, 2], 2, [[3., 1.7320508], [6., -3.4641016]]),
#          ([0.5, 2], 0.5, [[0.375, 0.21650635], [1.5, -0.8660254]])])
#     def testGetPlane(self, real, factor, points):
#         real.setMiller()
#         plane = real.getPlane(factor=factor)
#         np.testing.assert_almost_equal(points, plane)
#
#     @pytest.mark.parametrize(('miller', 'factor'), [([0, 1], 0), ([1, 1], 1),
#                                                     ([1, 2], 2),
#                                                     ([0.5, 2], 0.5)])
#     def testPlotPlane(self, real, factor):
#         real.setMiller()
#         real.setGridsAndLim()
#         real.plotPlane(factor=factor)
#         assert 1 == len(real.ax.lines) or 1 == len(real.ax.collections)
