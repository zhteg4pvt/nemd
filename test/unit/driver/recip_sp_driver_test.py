import argparse
import os
from unittest import mock

import pytest
import recip_sp_driver as driver

from nemd import np
from nemd import pd
from nemd import plotutils


@pytest.mark.parametrize('args', [(['-miller', '2', '4'])])
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
        'args,expected', [(['-miller', '1', '2'], (1, 2)),
                          (['-miller', '0', '0'], argparse.ArgumentTypeError)])
    def testMillerAction(self, parser, args, expected, raises):
        with raises:
            assert expected == parser.parse_args(args).miller


@pytest.mark.parametrize('lat',
                         [[[2.0943951, 2.0943951], [3.62759873, -3.62759873]]])
class TestRecip:

    @pytest.fixture
    def recip(self, lat, args, logger):
        options = driver.Parser().parse_args(args)
        with plotutils.pyplot() as plt:
            return driver.Recip(lat,
                                ax=plt.figure().add_subplot(111),
                                options=options,
                                logger=logger)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller', '1', '1'], [[2.0943951, 3.62759873],
                                  [2.0943951, -3.62759873], [4.1887902, 0]])])
    def testSetUp(self, recip, expected):
        to_compare = [recip.scaled.a1, recip.scaled.a2, recip.vec]
        np.testing.assert_almost_equal(expected, to_compare)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller', '1', '1'
           ], 'The norm of reciprocal vector [4.1887902 0.       ] is 4.189')])
    def testLogNorm(self, recip, expected):
        recip.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected', [([], 9)])
    def testPlot(self, recip, expected):
        recip.plot(recip.ax)
        assert expected == len(recip.ax._children)

    @pytest.mark.parametrize('args,expected', [([], [
        1, (-12.5663706, 12.5663706),
        (-21.76559238, 21.76559238), 'Reciprocal Space'
    ])])
    def testGrid(self, recip, expected):
        recip.grid()
        assert expected[0] == len(recip.ax.collections)
        lim = [recip.ax.get_xlim(), recip.ax.get_ylim()]
        np.testing.assert_almost_equal(lim, expected[1:-1])
        assert expected[-1] == recip.ax.get_title()

    @pytest.mark.parametrize('args,expected', [([], 85)])
    def testGrids(self, recip, expected):
        assert expected == recip.grids.shape[0]

    @pytest.mark.parametrize('args', [[]])
    @pytest.mark.parametrize('pnt,expected',
                             [([-12.5663706, -21.76559238], True),
                              ([12.5663706, 21.76559238], True),
                              ([12.5663706, 21.7655924], True),
                              ([12.5663706, 21.8], False),
                              ([-12.567, -21.76559238], False)])
    def testCrop(self, recip, pnt, expected):
        assert expected == recip.crop(np.array([pnt])).any()

    @pytest.mark.parametrize(
        'args,expected',
        [([], [[-12.5663706, 12.5663706], [-21.76559238, 21.76559238]])])
    def testLim(self, recip, expected):
        np.testing.assert_almost_equal(expected, recip.lim)

    @pytest.mark.parametrize('args,expected', [([], (13, 13, 2))])
    def testMeshed(self, recip, expected):
        assert expected == recip.meshed.shape

    @pytest.mark.parametrize(
        'args,vec,expected',
        [([], pd.Series([1, 2], name='name'), [1, 2, '$\\vec name^*$'])])
    def testQuiver(self, recip, vec, expected):
        recip.quiver(vec)
        annotation = list(recip.qvs.values())[0]
        assert expected == [*annotation.xy, annotation.get_text()]

    @pytest.mark.parametrize('args,vec,expected',
                             [([], pd.Series([1, 2], name='name'), 1)])
    def testArrow(self, recip, vec, expected):
        recip.arrow(vec)
        assert expected == len(recip.ax._children)

    @pytest.mark.parametrize(
        'args,vec,expected',
        [([], pd.Series([1, 2], name='name'), ['$\\vec name^*$ (1, 2)'])])
    def testLegend(self, recip, vec, expected):
        recip.quiver(vec)
        recip.legend()
        legend = recip.ax.get_legend()
        assert expected == [x.get_text() for x in legend.texts]


@pytest.mark.parametrize('lat', [[[1.5, 1.5], [0.8660254, -0.8660254]]])
class TestReal:

    @pytest.fixture
    def real(self, lat, args, logger):
        options = driver.Parser().parse_args(args)
        with plotutils.pyplot() as plt:
            return driver.Real(lat,
                               ax=plt.figure().add_subplot(111),
                               options=options,
                               logger=logger)

    @pytest.mark.parametrize(
        'args,expected', [(['-miller', '0', '1'], [0.75, -1.2990381]),
                          (['-miller', '1', '2'], [0.75, -0.4330127]),
                          (['-miller', '0.5', '2'], [0.57692307, -0.59955605]),
                          (['-miller', '1', '1'], [1.5, 0.])])
    def testSetUp(self, real, expected):
        np.testing.assert_almost_equal(real.vec, expected)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-miller', '0', '1'], [[1.5, -0.8660254], [1.5, 0.8660254]]),
         (['-miller', '1', '0'], [[1.5, 0.8660254], [1.5, -0.8660254]]),
         (['-miller', '1', '1'], [[1.5, 0.8660254], [0., -1.7320508]]),
         (['-miller', '-1', '1'], [[-1.5, -0.8660254], [3., 0.]]),
         (['-miller', '1', '2'], [[1.5, 0.8660254], [-0.75, -1.2990381]]),
         (['-miller', '0.5', '2'], [[3., 1.7320508], [-2.25, -2.1650635]])])
    def testGetPlane(self, real, expected):
        point, vec = real.getPlane()
        np.testing.assert_almost_equal([point, vec], expected)
        point2, vec2 = real.getPlane(factor=2)
        np.testing.assert_almost_equal(point2, point * 2)
        np.testing.assert_almost_equal(vec2, vec)

    @pytest.mark.parametrize('args', [(['-miller', '0', '1']),
                                      (['-miller', '1', '-1']),
                                      (['-miller', '0.5', '2']),
                                      (['-miller', '1', '1'])])
    def testPlot(self, real):
        real.plot(real.ax)
        assert 3 == len(real.ax.lines)

    @pytest.mark.parametrize(
        'args,vec,expected',
        [([], pd.Series([1, 2], name='name'), '$\\vec name$')])
    def testQuiver(self, real, vec, expected):
        real.quiver(vec)
        assert expected == list(real.qvs.values())[0].get_text()

    @pytest.mark.parametrize(
        'args,idx,expected',
        [(['-miller', '0', '1'], 0, [[-9., -5.1961524], [9., 5.1961524]]),
         (['-miller', '1', '-1'], -1, [[-9., -0.8660254], [9., -0.8660254]]),
         (['-miller', '0.5', '2'], 1, [[-4.2, -5.1961524], [6.6, 5.1961524]]),
         (['-miller', '1', '1'], 2, [[3., -5.1961524], [3., 5.1961524]])])
    def testPlane(self, real, idx, expected):
        real.plane(idx)
        data = np.array([y for x in real.ax.lines for y in zip(*x.get_data())])
        np.testing.assert_almost_equal(data, expected)
