import os

import dispersion_driver as driver
import numpy as np
import pytest

from nemd import envutils

ARGS = ['-name', 'Si', '-JOBNAME', 'dispersion']
BANDS = envutils.test_data('0044', 'phonons', 'dispersion.bands')


@pytest.mark.skipif(BANDS is None, reason="test data not found")
@pytest.mark.parametrize('args,infile', [(ARGS, BANDS)])
class TestPlotter:

    @pytest.fixture
    def plot(self, args, infile):
        options = driver.Parser().parse_args(args)
        return driver.Plotter(infile, options=options)

    @pytest.mark.parametrize('expected',
                             [[(153, 6), 'G', 'X|R', 'G', 'L', 'cm^-1']])
    def testRead(self, plot, expected):
        plot.read()
        assert expected == [plot.data.shape, *plot.points.index, plot.unit]

    @pytest.mark.parametrize('expected',
                             [[(153, 6), 'G', 'X|R', 'G', 'L', 'cm^-1']])
    def testPlot(self, plot, expected, tmp_dir):
        plot.read()
        plot.plot()
        assert os.path.exists(plot.outfile)


class TestDispersion:

    @pytest.fixture
    def disp(self, args, logger, tmp_dir):
        options = driver.Parser().parse_args(args)
        return driver.Dispersion(options, logger=logger)

    @pytest.mark.parametrize("args,expected", [(ARGS, 'dispersion.png')])
    def testRun(self, disp, expected):
        disp.run()
        assert os.path.exists(expected)

    @pytest.mark.parametrize("args,expected",
                             [(ARGS, (160.16, 8)),
                              ([*ARGS, '-scale_factor', '1.1'], (213.18, 8)),
                              ([*ARGS, '-dim', '2', '1', '1'], (160.16, 16))])
    def testBuild(self, disp, expected):
        disp.build()
        to_compare = [disp.crystal.volume, disp.struct.atom_total]
        np.testing.assert_almost_equal(to_compare, expected, decimal=2)

    @pytest.mark.slow(3)
    @pytest.mark.parametrize("args", [ARGS])
    def testWrite(self, disp):
        disp.build()
        disp.write()
        assert os.path.exists(disp.struct.outfile)

    @pytest.mark.parametrize("args,expected", [(ARGS, 'dispersion.png')])
    def testPlot(self, disp, expected):
        disp.build()
        disp.write()
        disp.plot()
        assert os.path.exists(expected)


class TestParser:

    @pytest.fixture
    def parser(self):
        return driver.Parser()

    @pytest.mark.parametrize(
        'expected', [dict(no_minimize=True, temp=0, force_field=['SW'])])
    def testAdd(self, parser, expected):
        assert expected == parser._defaults
