import types

import numba
import numpy as np
import pytest

from nemd import numbautils


class TestFunc:

    @pytest.mark.parametrize('ekey', ['PYTHON'])
    @pytest.mark.parametrize('evalue,instance',
                             [('-1', types.FunctionType),
                              ('1', numba.core.registry.CPUDispatcher),
                              ('2', numba.core.registry.CPUDispatcher)])
    def testJit(self, instance, evalue, env):

        @numbautils.jit
        def direct():
            return 1

        @numbautils.jit(parallel=False)
        def bracketed():
            return 1

        for decorated in (direct, bracketed):
            assert isinstance(decorated, instance)
            if evalue == '-1':
                continue
            is_null = isinstance(decorated._cache,
                                 numba.core.caching.NullCache)
            assert (evalue == '1') == is_null

    @pytest.mark.parametrize(
        "dists,span,expected",
        [([[0.5, 2, 3], [2.5, 3.5, -0.5]], [2.5, 2.5, 2.5], [0.8660, 1.1180]),
         ([[0.5, 2], [2.5, -0.1]], [1, 2], [0.7071, 0.5099])])
    def testNorn(self, dists, span, expected):
        remained = numbautils.norm(np.array(dists), np.array(span))
        np.testing.assert_almost_equal(remained, expected, decimal=4)
