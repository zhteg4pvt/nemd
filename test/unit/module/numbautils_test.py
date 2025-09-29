import types
from unittest import mock

import numba
import numpy as np
import pytest

from nemd import dist
from nemd import envutils
from nemd import numbautils
from nemd import pbc


class TestFunc:

    TRJ = np.array(
        [[[1, 2, 3], [8, 9, 10], [6, 4, 1]], [[4, 5, 6], [6, 4, 1], [6, 4, 1]],
         [[7, 8, 9], [6, 4, 1], [6, 4, 1]]],
        dtype=np.float32)

    @pytest.fixture
    def cell(self, xyzs, span, cut):
        box = pbc.Box.fromParams(*span)
        return dist.Cell(dist.Frame(xyzs, box=box, cut=cut))

    @pytest.mark.parametrize('ekey', ['PYTHON'])
    @pytest.mark.parametrize('evalue,dtype,ctype', [
     ('-1', types.FunctionType, types.NoneType),
     ('1', numba.core.registry.CPUDispatcher, numba.core.caching.NullCache),
     ('2', numba.core.registry.CPUDispatcher, numba.core.caching.FunctionCache)
    ]) # yapf: disable
    def testJit(self, dtype, ctype, env):

        @numbautils.jit
        def direct():
            return 1

        @numbautils.jit(parallel=False)
        def bracketed():
            return 1

        for decorated in (direct, bracketed):
            assert isinstance(decorated, dtype)
            assert isinstance(getattr(decorated, '_cache', None), ctype)

    @pytest.mark.parametrize(
        "dists,span,expected",
        [([[0.5, 2, 3], [2.5, 3.5, -0.5]], [2.5, 2.5, 2.5], [0.8660, 1.1180]),
         ([[0.5, 2], [2.5, -0.1]], [1, 2], [0.5, 0.5099])])
    def testNorms(self, dists, span, expected):
        norm = numbautils.norms(np.array(dists), np.array(span))
        np.testing.assert_almost_equal(norm, expected, decimal=4)

    @pytest.mark.parametrize(
        'gids,wt,expected', [([0, 1, 2], [1, 1, 1], [0., 23.555555, 69.77778]),
                             ([0, 1], [1, 2, 1], [0., 29.9375, 92.25]),
                             ([1], [1, 1, 1], [0., 38.11111, 103.55555])])
    def testMsd(self, gids, wt, expected):
        gids, wt = np.array(gids), np.array(wt, dtype=np.float32)
        msd = numbautils.msd(self.TRJ, gids, wt)
        np.testing.assert_almost_equal(msd, expected, 5)
