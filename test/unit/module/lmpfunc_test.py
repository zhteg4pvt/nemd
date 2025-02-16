import os.path

import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfunc

BASE_DIR = envutils.test_file('itest', '5524d62a356ac00d781a9cb1e5a6f03b')


class TestPress:

    DATA = os.path.join(BASE_DIR, 'defm_000', 'press_vol.data')

    @pytest.fixture
    def press(self):
        return lmpfunc.Press(self.DATA)

    def testSetData(self, press):
        press.setData()
        assert (300, 2) == press.data.shape

    def testSetAve(self, press):
        press.setData()
        press.setAve()
        assert 0.99842596 == press.ave

    def testGetColumn(self, press):
        press.setData()
        assert 'c_thermo_press' == press.getColumn('press').name

    def testGetLabel(self, press):
        assert 'Thermo Press' == press.getLabel('c_thermo_press')


class TestLength:

    DATA = os.path.join(BASE_DIR, 'xyzl.data')

    @pytest.fixture
    def length(self, last_pct, ending):
        return lmpfunc.Length(self.DATA, last_pct=last_pct, ending=ending)

    @pytest.mark.parametrize('last_pct,ending', [(0.2, 'xl'), (0.1, 'yl'),
                                                 (0.4, 'zl')])
    def testSetData(self, length):
        length.setData()
        assert (99, 3) == length.data.shape

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 159.1362), (0.1, 'yl', 159.1479),
                              (0.4, 'zl', 159.174875)])
    def testSetAve(self, expected, length):
        length.setData()
        length.setAve()
        np.testing.assert_almost_equal(length.ave, expected)

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 'xyzl_xl.png'),
                              (0.1, 'yl', 'xyzl_yl.png'),
                              (0.4, 'zl', 'xyzl_zl.png')])
    def testPlot(self, expected, length, tmp_dir):
        length.setData()
        length.setAve()
        length.plot()
        os.path.isfile(expected)
