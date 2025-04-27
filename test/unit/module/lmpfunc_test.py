import os.path

import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfunc
from nemd import plotutils

BASE_DIR = envutils.test_data('5524d62a356ac00d781a9cb1e5a6f03b')
PRESS_DATA = os.path.join(BASE_DIR, 'defm_000', 'press_vol.data')
XYZL_DATA = os.path.join(BASE_DIR, 'xyzl.data')


class TestBase:

    @pytest.fixture
    def base(self):
        return lmpfunc.Base(PRESS_DATA)

    def testSetData(self, base):
        base.setData()
        assert (300, 2) == base.data.shape

    def testSetAve(self, base):
        base.setData()
        base.setAve()
        assert (2, ) == base.ave.shape

    def testGetColumn(self, base):
        base.setData()
        assert 'c_thermo_press' == base.getColumn('press').name

    def testGetLabel(self, base):
        assert 'Thermo Press' == base.getLabel('c_thermo_press')


class TestLength:

    @pytest.fixture
    def length(self, last_pct, ending):
        return lmpfunc.Length(XYZL_DATA, last_pct=last_pct, ending=ending)

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


class TestPress:

    @pytest.fixture
    def press(self):
        return lmpfunc.Press(PRESS_DATA)

    def testSetAve(self, press):
        press.setData()
        press.setAve()
        np.testing.assert_almost_equal(press.ave, 0.99842596)


class TestModulus:

    @pytest.fixture
    def modulus(self):
        return lmpfunc.Modulus(PRESS_DATA, 100)

    def testSetData(self, modulus):
        modulus.setData()
        assert (300, 2) == modulus.data.shape

    def testSetAve(self, modulus):
        modulus.setData()
        modulus.setAve()
        assert (100, 6) == modulus.ave.shape

    @pytest.mark.parametrize('lower_bound,expected', [(10, 10),
                                                      (0, 1.1052269716604515)])
    def testSetModulus(self, lower_bound, expected, modulus):
        modulus.setData()
        modulus.setAve()
        modulus.setModulus(lower_bound=lower_bound)
        np.testing.assert_almost_equal(modulus.modulus, expected)

    def testPlot(self, modulus, tmp_dir):
        modulus.setData()
        modulus.setAve()
        modulus.plot()
        os.path.isfile('press_vol_modulus.png')

    @pytest.mark.parametrize('label,expected',
                             [('c_thermo_press', 'xyzl_xl.png'),
                              ('v_vol', 'xyzl_yl.png')])
    def testSubplot(self, label, expected, modulus):
        modulus.setData()
        modulus.setAve()
        with plotutils.pyplot(inav=False) as plt:
            ax = plt.subplot()
            modulus.subplot(ax, label)
            assert 2 == len(ax.lines)
            assert 1 == len(ax.collections)


class TestVol:

    @pytest.fixture
    def vol(self, press):
        return lmpfunc.Vol(press, PRESS_DATA)

    @pytest.mark.parametrize('press', [(1)])
    def testSetAve(self, vol):
        vol.setData()
        vol.setAve()
        assert 100 == len(vol.vol)
        assert 100 == len(vol.ave)

    @pytest.mark.parametrize('press,expected', [(1, 1),
                                                (100, 0.9774937675048474),
                                                (-100, 1.03691022129205)])
    def testSetAve(self, vol, expected):
        vol.setData()
        vol.setAve()
        vol.setFactor()
        np.testing.assert_almost_equal(vol.factor, expected)

    @pytest.mark.parametrize('press', [(1)])
    def testPlot(self, vol, tmp_dir):
        vol.setData()
        vol.setAve()
        vol.plot()
        os.path.isfile('press_vol_scale.png')


class TestFunc:

    def testGetL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getL(XYZL_DATA), 159.1362)

    def testGetXL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getXL(XYZL_DATA), 159.1362)

    def testGetYL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getYL(XYZL_DATA), 159.1362)

    def testGetZL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getZL(XYZL_DATA), 159.1362)

    def testGetPress(self):
        np.testing.assert_almost_equal(lmpfunc.getPress(PRESS_DATA), 0.998426)

    def testGetModulus(self, tmp_dir):
        modulus = lmpfunc.getModulus(PRESS_DATA, 100)
        np.testing.assert_almost_equal(modulus, 10)

    @pytest.mark.parametrize('press,expected', [(1, 1),
                                                (100, 0.9774937675048474),
                                                (-100, 1.03691022129205)])
    def testGetVolFactor(self, press, expected, tmp_dir):
        factor = lmpfunc.getVolFactor(press, PRESS_DATA)
        np.testing.assert_almost_equal(factor, expected)

    @pytest.mark.parametrize('press,expected', [(1, 1),
                                                (100, 0.9924409268894283),
                                                (-100, 1.0121550629238298)])
    def testGetBdryFactor(self, press, expected, tmp_dir):
        factor = lmpfunc.getBdryFactor(press, PRESS_DATA)
        np.testing.assert_almost_equal(factor, expected)
