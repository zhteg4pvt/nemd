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

    @pytest.mark.parametrize(
        "arrays,gid,expected",
        [([[2., 2.25, 2.2], [5, 4, 5], [[1, 4, 1]]], 0, [0, 2, 0])])
    def testGetIds(self, arrays, gid, expected):
        ids = numbautils.get_ids(*[np.array(x) for x in arrays], gid)
        np.testing.assert_almost_equal(ids, expected)

    @pytest.mark.parametrize('span,cut', [([10, 9, 11], 2)])
    @pytest.mark.parametrize(
        "xyzs,gids,expected",
        [([[1, 4, 1]], [0], [[0, 2, 0, 0]]),
         ([[1, 4, 1], [2, 3, 6]], [0, 1], [[0, 2, 0, 0], [1, 1, 3, 1]])])
    def testSet(self, cell, xyzs, gids, expected):
        args = [cell, cell.grids, cell.dims, cell.frm, np.array(gids)]
        numbautils.set(*args)
        np.testing.assert_almost_equal(np.array(cell.nonzero()).T, expected)
        numbautils.set(*args, state=False)
        assert not cell.nonzero()[0].size
