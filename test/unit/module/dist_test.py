import math
import os

import numpy as np
import pytest

from nemd import dist
from nemd import envutils
from nemd import lmpfull
from nemd import pbc

NACL_DATA = envutils.test_data('itest', '0027_test', 'workspace',
                               '062200efd143bd63bc59842f7ffb56d5',
                               'amorp_bldr.data')
NACL_RDR = lmpfull.Reader(NACL_DATA)
HE_RDR = lmpfull.Reader(envutils.test_data('he', 'mol_bldr.data'))

HEX = envutils.test_data('hexane_liquid')
HEXANE_READER = lmpfull.Reader(os.path.join(HEX, 'polymer_builder.data'))
HEXANE_FRAME = os.path.join(HEX, 'dump.custom')


class TestRadius:

    @pytest.mark.parametrize('struct,num,expected', [(None, 3, [1, 3]),
                                                     (NACL_RDR, 1, [2, 20]),
                                                     (HE_RDR, 1, [1, 1])])
    def testNew(self, struct, num, expected):
        radii = dist.Radius(struct=struct, num=num)
        assert expected == [radii.shape[0], radii.map.shape[0]]

    @pytest.fixture
    def radii(self, struct):
        return dist.Radius(struct=struct)

    @pytest.mark.parametrize('struct,args,expected',
                             [(HE_RDR, (0, 0), 1.4), (NACL_RDR, (0, 0), 2.231),
                              (NACL_RDR, (10, 10), 1.682),
                              (NACL_RDR, (9, 10), 1.937),
                              (NACL_RDR, (0, [1, 11]), [2.231, 1.937]),
                              (NACL_RDR, ([12, 19], [1, 11]), [1.937, 1.682])])
    def testGet(self, radii, args, expected):
        np.testing.assert_almost_equal(radii.get(*args), expected, decimal=3)


class TestCell:

    @pytest.fixture
    def cell(self, num, span, cut):
        return dist.Cell(num, np.array(span), cut)

    @pytest.mark.parametrize('num,span,cut,expected',
                             [(4, [10, 9, 11], 2, (5, 4, 5, 4))])
    def testNew(self, span, cut, cell, expected):
        assert expected == cell.shape
        np.testing.assert_almost_equal(span, cell.grids * cell.shape[:-1])

    @pytest.mark.parametrize('num,span,cut', [(4, [10, 9, 11], 2)])
    @pytest.mark.parametrize('xyzs', [([1, 4, 1]), ([[1, 4, 1]])])
    @pytest.mark.parametrize('gids', [(1), ([1])])
    def testSet(self, cell, xyzs, gids, span):
        cell.set(xyzs, gids)
        ixs, iys, izs, ids = cell.nonzero()
        assert (gids == ids).all()
        cell.set(xyzs, gids, state=False)
        assert 0 == len(cell.nonzero()[0])

    @pytest.mark.parametrize('num,span,cut', [(4, [10, 9, 11], 2)])
    @pytest.mark.parametrize('xyzs', [([1, 4, 1]), ([[1, 4, 1]])])
    @pytest.mark.parametrize('ids', [([0, 2, 0]), ([[0, 2, 0]])])
    def testGetIds(self, cell, xyzs, ids, span):
        assert (ids == cell.getCids(xyzs)).all()

    @pytest.mark.parametrize('num,span,cut,gids',
                             [(4, [10, 9, 11], 2, [0, 1])])
    @pytest.mark.parametrize('xyzs', [([1, 4, 1])])
    @pytest.mark.parametrize('ids', [([0, 2, 1])])
    def testGet(self, cell, xyzs, gids, ids, span):
        cell.set(xyzs, gids)
        cell.get(xyzs)

    @pytest.mark.parametrize('num,span,cut,gid', [(4, [10, 9, 11], 2, 1)])
    @pytest.mark.parametrize('xyz', [([1, 4, 1]), ([-1, -4, 0]),
                                     ([8, 100, 11])])
    def test(self, cell, xyz, gid, span):
        cell.set(xyz, gid)
        center = np.array(cell.nonzero()[:-1]).transpose()[0] * cell.grids
        # Between 0 and the span
        assert (center < span).all()
        assert (center >= 0).all()
        # Near the original point
        norm = pbc.Box.fromParams(*span).norm(xyz - center)
        assert norm < np.linalg.norm(cell.grids / 2)

    # @pytest.mark.parametrize('file', [HEXANE_FRAME])
    # @pytest.mark.parametrize('grp,grps', [(None, None), ([0, 1], None),
    #                                       ([0], [[1]])])
    # def testPairDists(self, frm, grp, grps):
    #     dists = frm.getDists(grp=grp, grps=grps)
    #     breakpoint()
    #     # np.testing.assert_almost_equal(dists, expected)
