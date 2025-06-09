import os

import numpy as np
import pytest
import recip_sp_driver as driver

from nemd import recip_sp


class TestReciprocal:

    @pytest.fixture
    def recip(self, miller):
        options = driver.validate_options(['-miller_indices', *miller])
        return recip_sp.Reciprocal(options)

    @pytest.mark.parametrize(
        ('miller,vecs'),
        [(['0.5', '2'], [[1.5, 0.8660254], [1.5, -0.8660254]])])
    def testSetReal(self, recip, vecs):
        recip.setReal()
        np.testing.assert_almost_equal(np.array(vecs).T, recip.real)

    @pytest.mark.parametrize(
        ('miller,vecs'),
        [(['0.5', '2'], [[2.0943951, 3.62759873], [2.0943951, -3.62759873]])])
    def testSetReciprocal(self, recip, vecs):
        recip.setReal()
        recip.setReciprocal()
        np.testing.assert_almost_equal(np.array(vecs).T, recip.recip)

    @pytest.mark.parametrize(('miller'), [(['0', '1']), (['0', '2']),
                                          (['2', '4'])])
    def testPlot(self, recip, tmp_dir):
        recip.setReal()
        recip.setReciprocal()
        recip.plot()
        assert os.path.isfile('recip_sp.png')
