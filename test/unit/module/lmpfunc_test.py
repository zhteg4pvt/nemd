import os

import conftest
import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfunc
from nemd import plotutils

PRESS_DATA = envutils.test_data('water', 'defm_00', 'press_vol.data')


@conftest.require_src
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


@conftest.require_src
@pytest.mark.parametrize('xyzl', [envutils.test_data('water', 'xyzl.data')])
class TestLength:

    @pytest.fixture
    def length(self, xyzl, last_pct, ending):
        return lmpfunc.Length(xyzl, last_pct=last_pct, ending=ending)

    @pytest.mark.parametrize('last_pct,ending', [(0.2, 'xl'), (0.1, 'yl'),
                                                 (0.4, 'zl')])
    def testRead(self, length):
        length.read()
        assert (999, 3) == length.data.shape

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 4.81020585), (0.1, 'yl', 4.8101973),
                              (0.4, 'zl', 4.8101627)])
    def testSetAve(self, length, expected):
        length.read()
        length.setAve()
        np.testing.assert_almost_equal(length.ave, expected)

    @pytest.mark.parametrize('last_pct,ending,expected',
                             [(0.2, 'xl', 'xyzl_xl.png'),
                              (0.1, 'yl', 'xyzl_yl.png'),
                              (0.4, 'zl', 'xyzl_zl.png')])
    def testPlot(self, length, expected, tmp_dir):
        length.read()
        length.setAve()
        length.plot()
        os.path.isfile(expected)

    def testGet(self, xyzl, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.Length.get(xyzl), 4.81020585)

    def testGetX(self, xyzl, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.Length.getX(xyzl), 4.81020585)

    def testGetY(self, xyzl, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.Length.getY(xyzl), 4.81020585)

    def testGetZ(self, xyzl, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.Length.getZ(xyzl), 4.81020585)


@conftest.require_src
@pytest.mark.parametrize('data', [PRESS_DATA])
class TestPress:

    @pytest.fixture
    def press(self, data):
        return lmpfunc.Press(data)

    def testSetAve(self, press):
        press.read()
        press.setAve()
        np.testing.assert_almost_equal(press.ave, -2937.646184)

    def testGet(self, data):
        np.testing.assert_almost_equal(lmpfunc.Press.get(data), -2937.646184)


@conftest.require_src
@pytest.mark.parametrize('data', [PRESS_DATA])
class TestFactor:

    @pytest.fixture
    def fac(self, press, data):
        return lmpfunc.Factor(press, data)

    @pytest.mark.parametrize('press,expected', [(-2937, 1), (100, 0.995),
                                                (-5000, 1.005)])
    def testRun(self, fac, expected):
        np.testing.assert_almost_equal(fac.run(), expected)

    @pytest.mark.parametrize('press,expected', [(-2937, 1), (100, 0.995),
                                                (-5000, 1.005)])
    def testGetVol(self, press, data, expected, tmp_dir):
        factor = lmpfunc.Factor.getVol(press, data)
        np.testing.assert_almost_equal(factor, expected)

    @pytest.mark.parametrize('press,expected', [(-2937, 1),
                                                (100, 0.9983305478136913),
                                                (-5000, 1.001663896579312)])
    def testgetBdry(self, press, expected, data, tmp_dir):
        factor = lmpfunc.Factor.getBdry(press, data)
        np.testing.assert_almost_equal(factor, expected)


@conftest.require_src
@pytest.mark.parametrize('data', [PRESS_DATA])
class TestModulus:

    @pytest.fixture
    def modulus(self, data):
        return lmpfunc.Modulus(data, 100)

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

    def testGetModulus(self, data, tmp_dir):
        np.testing.assert_almost_equal(lmpfunc.Modulus.get(data, 100),
                                       67468.3215133)
