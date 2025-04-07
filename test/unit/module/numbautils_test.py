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
        with (mock.patch('nemd.numbautils.NOPYTHON', envutils.nopython()),
              mock.patch('nemd.numbautils.JIT_KWARGS', envutils.jit_kwargs())):

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
