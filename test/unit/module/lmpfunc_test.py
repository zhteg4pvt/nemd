import os.path

import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfunc
from nemd import plotutils

BASE_DIR = envutils.test_data('water')
PRESS_DATA = os.path.join(BASE_DIR, 'defm_00', 'press_vol.data')
XYZL_DATA = os.path.join(BASE_DIR, 'xyzl.data')


class TestBase:

    @pytest.fixture
    def base(self):
        return lmpfunc.Base(PRESS_DATA)

    def testRead(self, base):
        base.read()
        assert (299, 2) == base.data.shape

    def testSetAve(self, base):
        base.read()
        base.setAve()
        assert (2, ) == base.ave.shape

    def testGetColumn(self, base):
        base.read()
        assert 'c_thermo_press' == base.getColumn('press').name

    def testGetLabel(self, base):
        assert 'Thermo Press' == base.getLabel('c_thermo_press')


class TestLength:

    @pytest.fixture
    def length(self, last_pct, ending):
        return lmpfunc.Length(XYZL_DATA, last_pct=last_pct, ending=ending)

    @pytest.mark.parametrize('last_pct,ending', [(0.2, 'xl'), (0.1, 'yl'),
                                                 (0.4, 'zl')])
    def testRead(self, length):
        length.read()
        assert (999, 3) == length.data.shape

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 4.81020585), (0.1, 'yl', 4.8101973),
                              (0.4, 'zl', 4.8101627)])
    def testSetAve(self, expected, length):
        length.read()
        length.setAve()
        np.testing.assert_almost_equal(length.ave, expected)

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 'xyzl_xl.png'),
                              (0.1, 'yl', 'xyzl_yl.png'),
                              (0.4, 'zl', 'xyzl_zl.png')])
    def testPlot(self, expected, length, tmp_dir):
        length.read()
        length.setAve()
        length.plot()
        os.path.isfile(expected)


class TestPress:

    @pytest.fixture
    def press(self):
        return lmpfunc.Press(PRESS_DATA)

    def testSetAve(self, press):
        press.read()
        press.setAve()
        np.testing.assert_almost_equal(press.ave, -2937.6461839464882)


class TestModulus:

    @pytest.fixture
    def modulus(self):
        return lmpfunc.Modulus(PRESS_DATA, 100)

    def testRead(self, modulus):
        modulus.read()
        assert (299, 2) == modulus.data.shape

    def testSetAve(self, modulus):
        modulus.read()
        modulus.setAve()
        assert (100, 6) == modulus.ave.shape

    @pytest.mark.parametrize('lower_bound,expected', [(10, 67468.3215132747),
                                                      (1E5, 1E5)])
    def testSetModulus(self, lower_bound, expected, modulus):
        modulus.read()
        modulus.setAve()
        modulus.setModulus(lower_bound=lower_bound)
        np.testing.assert_almost_equal(modulus.modulus, expected)

    def testPlot(self, modulus, tmp_dir):
        modulus.read()
        modulus.setAve()
        modulus.plot()
        os.path.isfile('press_vol_modulus.png')

    @pytest.mark.parametrize('label,expected',
                             [('c_thermo_press', 'xyzl_xl.png'),
                              ('v_vol', 'xyzl_yl.png')])
    def testSubplot(self, label, expected, modulus, tmp_dir):
        modulus.read()
        modulus.setAve()
        with plotutils.pyplot(inav=False) as plt:
            ax = plt.subplot()
            modulus.subplot(ax, label)
            assert 2 == len(ax.lines)
            assert 1 == len(ax.collections)


class TestFactor:

    @pytest.fixture
    def fac(self, press):
        return lmpfunc.Factor(press, PRESS_DATA)

    @pytest.mark.parametrize('press,expected', [(-2937, 1), (100, 0.995),
                                                (-5000, 1.005)])
    def testRun(self, fac, expected):
        np.testing.assert_almost_equal(fac.run(), expected)


class TestFunc:

    def testGetL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getL(XYZL_DATA), 4.81020585)

    def testGetXL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getXL(XYZL_DATA), 4.81020585)

    def testGetYL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getYL(XYZL_DATA), 4.81020585)

    def testGetZL(self, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.getZL(XYZL_DATA), 4.81020585)

    def testGetPress(self):
        press = lmpfunc.getPress(PRESS_DATA)
        np.testing.assert_almost_equal(press, -2937.6461839464882)

    def testGetModulus(self, tmp_dir):
        modulus = lmpfunc.getModulus(PRESS_DATA, 100)
        np.testing.assert_almost_equal(modulus, 67468.3215132747)

    @pytest.mark.parametrize('press,expected', [(-2937, 1), (100, 0.995),
                                                (-5000, 1.005)])
    def testGetVolFac(self, press, expected, tmp_dir):
        factor = lmpfunc.getVolFac(press, PRESS_DATA)
        np.testing.assert_almost_equal(factor, expected)

    @pytest.mark.parametrize('press,expected', [(-2937, 1),
                                                (100, 0.9983305478136913),
                                                (-5000, 1.001663896579312)])
    def testgetBdryFac(self, press, expected, tmp_dir):
        factor = lmpfunc.getBdryFac(press, PRESS_DATA)
        np.testing.assert_almost_equal(factor, expected)
