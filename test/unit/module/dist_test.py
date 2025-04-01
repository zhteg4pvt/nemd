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

    @pytest.mark.parametrize('struct,args', [(HE_RDR, (0, 0))])
    def testArrayWrap(self, radii, args):
        isinstance(radii.max(), float)
        isinstance(radii.ravel(), dist.Radius)


class TestCell:

    @pytest.fixture
    def cell(self, xyzs, span, cut):
        box = pbc.Box.fromParams(*span)
        return dist.Cell(dist.Frame(xyzs, box=box, cut=cut))

    @pytest.mark.parametrize('xyzs,span,cut,expected',
                             [([[1, 4, 1]], [10, 9, 11], 2, (5, 4, 5, 1))])
    def testNew(self, span, cut, cell, expected):
        assert expected == cell.shape
        np.testing.assert_almost_equal(span, cell.grids * cell.shape[:-1])

    @pytest.mark.parametrize('data,cut,upper,expected',
                             [([10, 20, 5], 0.3, None, (33, 66, 16)),
                              ([10, 20, 5], 0.3, 20, (20, 20, 16))])
    def testGetShape(self, data, cut, upper, expected):
        assert expected == dist.Cell.getShape(data, cut, upper=upper)

    @pytest.mark.parametrize('xyzs,span,cut', [([[1, 4, 1]], [10, 9, 11], 2)])
    @pytest.mark.parametrize('gids', [0, [0]])
    def testSet(self, cell, gids, span):
        cell.set(gids)
        ixs, iys, izs, ids = cell.nonzero()
        assert (gids == ids).all()
        cell.set(gids, state=False)
        assert 0 == len(cell.nonzero()[0])

    @pytest.mark.parametrize('xyzs,span,cut', [([[1, 4, 1]], [10, 9, 11], 2)])
    @pytest.mark.parametrize('gids,shape', [(0, (3, )), ([0], (1, 3))])
    def testGetCids(self, cell, gids, shape):
        cids = cell.getCids(gids)
        assert shape == cids.shape
        assert ([0, 2, 0] == cids).all()

    @pytest.mark.parametrize(
        'xyzs,span,cut',
        [([[5, 8.5, 5], [4.5, 0.5, 5.5], [4.5, 5, 5.5]], [10, 9, 11], 2)])
    @pytest.mark.parametrize('gids,gid,gt,expected',
                             [([0, 1], 0, False, [0, 1]),
                              ([0, 1], 0, True, [1]), ([0, 1], 2, False, []),
                              ([1], 0, False, [1])])
    def testGet(self, cell, gids, gid, gt, expected):
        cell.set(gids)
        assert expected == cell.get(gid, gt=gt)

    @pytest.mark.parametrize('dims,shape,expected',
                             [((2, 3, 4), (1, 1, 1), 18),
                              ((4, 3, 1), (2, 2, 2), 12)])
    def testGetNbrs(self, dims, shape, expected):
        nx, ny, nz, num, _ = dist.Cell.getNbrs(shape, dims).shape
        assert expected == num
        assert dims == (nx, ny, nz)

    @pytest.mark.parametrize('dims,shape,expected',
                             [((2, 3, 4), (1, 1, 1), 18),
                              ((4, 3, 1), (2, 2, 2), 12)])
    def testGetOrigNbrs(self, dims, shape, expected):
        assert expected == dist.Cell.getOrigNbrs(shape, dims).shape[0]

    @pytest.mark.parametrize('span,cut', [([10, 9, 11], 2)])
    @pytest.mark.parametrize('xyzs',
                             [[[1, 4, 1]], [[-1, -4, 0]], [[8, 100, 11]]])
    def test(self, cell, xyzs, span):
        cell.set(0)
        center = np.array(cell.nonzero()[:-1]).transpose()[0] * cell.grids
        # Between 0 and the span
        assert (center < span).all()
        assert (center >= 0).all()
        # Near the original point
        norm = pbc.Box.fromParams(*span).norm(xyzs - center)
        assert norm < np.linalg.norm(cell.grids / 2)


class TestCellNumba:

    @pytest.fixture
    def cell(self, xyzs, span, cut):
        box = pbc.Box.fromParams(*span)
        return dist.CellNumba(dist.Frame(xyzs, box=box, cut=cut))

    @pytest.mark.parametrize('xyzs,span,cut', [([[1, 4, 1]], [10, 9, 11], 2)])
    @pytest.mark.parametrize('gids', [[0]])
    def testSet(self, cell, gids, span):
        cell.set(gids)
        ixs, iys, izs, ids = cell.nonzero()
        assert (gids == ids).all()
        cell.set(gids, state=False)
        assert 0 == len(cell.nonzero()[0])

    @pytest.mark.parametrize(
        'xyzs,span,cut',
        [([[5, 8.5, 5], [4.5, 0.5, 5.5], [4.5, 5, 5.5]], [10, 9, 11], 2)])
    @pytest.mark.parametrize('gids,gid,gt,expected',
                             [([0, 1], 0, False, [0, 1]),
                              ([0, 1], 0, True, [1]), ([0, 1], 2, False, []),
                              ([1], 0, False, [1])])
    def testGet(self, cell, gids, gid, gt, expected):
        cell.set(gids)
        assert expected == cell.get(gid, gt=gt)

    @pytest.mark.parametrize('dims,shape,expected',
                             [((2, 3, 4), (1, 1, 1), 18),
                              ((4, 3, 1), (2, 2, 2), 12)])
    def testGetNbrs(self, dims, shape, expected):
        nx, ny, nz, num, _ = dist.Cell.getNbrs(shape, dims).shape
        assert expected == num
        assert dims == (nx, ny, nz)


class TestFrame:

    HEX = envutils.test_data('hexane_liquid')
    RDR = lmpfull.Reader(os.path.join(HEX, 'polymer_builder.data'))
    FRM = os.path.join(HEX, 'dump.custom')

    @pytest.fixture
    def fr(self, frm, gids, cut, struct, srch):
        return dist.Frame(frm, gids=gids, cut=cut, struct=struct, srch=srch)

    @pytest.mark.parametrize('file', [FRM])
    @pytest.mark.parametrize('gids,cut,struct,srch,expected',
                             [([1, 2], 1000, None, None, (2, 23.92, False)),
                              (None, 1000, None, False, (0, 1000, False)),
                              (None, 1000, None, True, (0, 23.92, True)),
                              (None, None, RDR, None, (0, 1.97, True))])
    def testInit(self, fr, expected):
        assert expected[0] == len(fr.gids.values)
        np.testing.assert_almost_equal(fr.cut, expected[1], decimal=2)
        assert expected[2] == (fr.cell is not None)

    @pytest.mark.parametrize('file,gids,cut,srch', [(FRM, None, None, None)])
    @pytest.mark.parametrize('struct,expected', [(None, 1.4), (RDR, 1.97)])
    def testRadii(self, fr, expected):
        np.testing.assert_almost_equal(fr.radii.max(), expected, decimal=2)

    @pytest.mark.parametrize('span,cut,expected', [([10, 10, 10], 1, True),
                                                   ([10, 10, 9], 1, False)])
    def testUseCell(self, span, cut, expected):
        assert expected == dist.Frame.useCell(span, cut)

    @pytest.mark.parametrize('file,gids,cut,struct,srch', [(FRM, None, None, None, None)])
    @pytest.mark.parametrize('grp,grps', [(None, None), ([0, 1], None),
                                          ([0], [[1]])])
    def testGetDists(self, fr, grp, grps):
        dists = fr.getDists(grp, grps=grps)
        breakpoint()
        # np.testing.assert_almost_equal(dists, expected)
