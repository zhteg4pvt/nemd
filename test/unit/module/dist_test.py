import copy
import os

import numpy as np
import pytest

from nemd import dist
from nemd import envutils
from nemd import lmpfull

HEX = envutils.test_data('hexane_liquid')
HEX_FRM = os.path.join(HEX, 'dump.custom')


class TestRadius:

    NACL = lmpfull.Reader(
        envutils.test_data('itest', '0027_test', 'workspace',
                           '062200efd143bd63bc59842f7ffb56d5',
                           'amorp_bldr.data'))
    HE = lmpfull.Reader(envutils.test_data('he', 'mol_bldr.data'))

    @pytest.mark.parametrize('struct,num,expected', [(None, 3, [1, 3]),
                                                     (NACL, 1, [2, 20]),
                                                     (HE, 1, [1, 1])])
    def testNew(self, struct, num, expected):
        radii = dist.Radius(struct=struct, num=num)
        assert expected == [radii.shape[0], radii.map.shape[0]]

    @pytest.fixture
    def radii(self, struct):
        return dist.Radius(struct=struct, num=15)

    @pytest.mark.parametrize('struct,args,expected',
                             [(HE, (0, 0), 1.4), (NACL, (0, 0), 2.231),
                              (NACL, (10, 10), 1.682), (NACL, (9, 10), 1.937),
                              (NACL, (0, [1, 11]), [2.231, 1.937]),
                              (None, (0, [1, 11]), [1.4, 1.4]),
                              (NACL, ([12, 19], [1, 11]), [1.937, 1.682])])
    def testGet(self, radii, args, expected):
        np.testing.assert_almost_equal(radii.get(*args), expected, decimal=3)

    @pytest.mark.parametrize('struct,args', [(HE, (0, 0))])
    def testArrayWrap(self, radii, args):
        isinstance(radii.max(), float)
        isinstance(radii.ravel(), dist.Radius)


class TestCellOrig:

    @pytest.fixture
    def cell(self, frm, cut):
        return dist.CellOrig(frm, frm.box.span, cut)

    @pytest.mark.parametrize('file', [HEX_FRM])
    @pytest.mark.parametrize(
        'cut,shape,grids',
        [(10, (4, 4, 4, 3000), [11.9610732, 11.9610732, 11.9610732]),
         (2, (20, 20, 20, 3000), [2.39221463, 2.39221463, 2.39221463])])
    def testInit(self, cell, shape, grids):
        assert shape == cell.cell.shape
        np.testing.assert_almost_equal(cell.grids, grids)

    @pytest.mark.parametrize('file,cut,gids', [(HEX_FRM, 10, [0, 6])])
    def testSet(self, cell, gids):
        cell.set(gids)
        assert set(gids) == set(cell.cell.nonzero()[-1])
        cell.set(gids, state=False)
        assert 0 == cell.cell.nonzero()[-1].size

    @pytest.mark.parametrize('file,cut,gids,expected',
                             [(HEX_FRM, 10, [0, 6], [[3, 2, 2], [2, 0, 2]])])
    def testGetCids(self, cell, gids, expected):
        np.testing.assert_almost_equal(cell.getCid(gids), expected)

    @pytest.mark.parametrize('file,cut,gids,gid',
                             [(HEX_FRM, 10, [0, 1, 2], 1)])
    @pytest.mark.parametrize('less,expected', [(False, [0, 1, 2]),
                                               (True, [0])])
    def testGet(self, cell, gids, gid, less, expected):
        cell.set(gids)
        assert expected == cell.get(gid, less=less)

    @pytest.mark.parametrize('dims,expected',
                             [((2, 2, 2), (2, 2, 2, 27, 3)),
                              ((10, 10, 10), (10, 10, 10, 27, 3))])
    def testGetNbrs(self, dims, expected):
        assert expected == dist.CellOrig.getNbrs(*dims).shape

    @pytest.mark.parametrize('file,cut', [(HEX_FRM, 10)])
    @pytest.mark.parametrize('gid', [0, 6, 24, 54, 100, 2777])
    def test(self, cell, gid):
        cell.set(gid)
        center = np.array(cell.cell.nonzero())[:-1].transpose() * cell.grids
        # Between 0 and the span
        assert (center < cell.frm.box.span).all()
        assert (center >= 0).all()
        # Near the original point
        norm = cell.frm.box.norms(cell.frm[gid] - center)
        assert (norm < np.linalg.norm(cell.grids / 2)).all()


class TestCellNumba:

    @pytest.fixture
    def cell(self, frm, cut):
        return dist.CellNumba(frm, frm.box.span, cut)

    @pytest.mark.parametrize('file,cut,gids',
                             [(HEX_FRM, 10, np.array([0, 6]))])
    def testSet(self, cell, gids):
        cell.set(gids)
        assert set(gids) == set(cell.cell.nonzero()[-1])
        cell.set(gids, False)
        assert 0 == cell.cell.nonzero()[-1].size

    @pytest.mark.parametrize(
        'file,cut,gids,expected',
        [(HEX_FRM, 10, [0, 6], np.array([[3, 2, 2], [2, 0, 2]]))])
    def testGetCids(self, cell, gids, expected):
        cid = cell.getCid(np.array(gids, dtype=np.int64))
        np.testing.assert_almost_equal(cid, expected)

    @pytest.mark.parametrize('file,cut,gids,gid',
                             [(HEX_FRM, 10, np.array([0, 1, 2]), 1)])
    @pytest.mark.parametrize('less,expected', [(False, [0, 1, 2]),
                                               (True, [0])])
    def testGet(self, cell, gids, gid, less, expected):
        cell.set(gids)
        assert expected == cell.get(gid, less)

    @pytest.mark.parametrize('dims,expected',
                             [((2, 2, 2), (2, 2, 2, 27, 3)),
                              ((10, 10, 10), (10, 10, 10, 27, 3))])
    def testGetNbrs(self, dims, expected):
        assert expected == dist.CellOrig.getNbrs(*dims).shape


class TestFrame:

    HEX_RDR = lmpfull.Reader(os.path.join(HEX, 'polymer_builder.data'))
    MODIFIED = copy.deepcopy(HEX_RDR)
    MODIFIED.pair_coeffs.dist = 18

    @pytest.fixture
    def fr(self, frm, gids, cut, struct, srch):
        return dist.Frame(frm, gids=gids, cut=cut, struct=struct, srch=srch)

    @pytest.mark.parametrize('file', [HEX_FRM])
    @pytest.mark.parametrize('gids,cut,struct,srch,expected',
                             [([1, 2], 1000, None, None, (2, 23.92, False)),
                              (None, 1000, None, False, (0, 1000, False)),
                              (None, 1000, None, True, (0, 23.92, True)),
                              (None, None, HEX_RDR, None, (0, 1.97, True))])
    def testSetUp(self, fr, expected):
        assert expected[0] == len(fr.gids.on)
        np.testing.assert_almost_equal(fr.cut, expected[1], decimal=2)
        assert expected[2] == (fr.cell is not None)

    @pytest.mark.parametrize('file,gids,cut,srch',
                             [(HEX_FRM, None, None, None)])
    @pytest.mark.parametrize('struct,expected', [(None, 1.4), (HEX_RDR, 1.97)])
    def testRadii(self, fr, expected):
        np.testing.assert_almost_equal(fr.radii.max(), expected, decimal=2)

    @pytest.mark.parametrize('span,cut,expected', [([10, 10, 10], 1, True),
                                                   ([10, 10, 9], 1, False)])
    def testUseCell(self, span, cut, expected):
        assert expected == dist.Frame.useCell(span, cut)

    @pytest.mark.parametrize('file,gids,cut,struct',
                             [(HEX_FRM, [0, 1, 11], None, None)])
    @pytest.mark.parametrize(
        'srch,grp,grps,less,expected',
        [(True, [0, 1], [[11], [11]], True, [15.3781802, 15.8164904]),
         (True, [0], None, True, []), (True, [0], None, False, [0., 1.524349]),
         (False, [0], None, False, [0., 1.524349, 15.3781802]),
         (True, [1], None, True, [1.52434905])])
    def testGetDists(self, fr, grp, grps, less, expected):
        dists = fr.getDists(grp, grps=grps, less=less)
        np.testing.assert_almost_equal(dists, expected)

    @pytest.mark.parametrize('file,gids,cut,struct',
                             [(HEX_FRM, [0, 1, 11], None, None)])
    @pytest.mark.parametrize('srch,gid,less,expected',
                             [(True, 11, True, []), (False, 11, True, [0, 1]),
                              (True, 11, False, [11]),
                              (False, 11, False, [0, 1, 11])])
    def testGetGrp(self, fr, gid, less, expected):
        np.testing.assert_almost_equal(fr.getGrp(gid, less=less), expected)

    @pytest.mark.parametrize('file,gids,cut,struct',
                             [(HEX_FRM, [0, 1, 5, 203], 5.2, MODIFIED)])
    @pytest.mark.parametrize(
        'srch,grp,grps,less,expected',
        [(True, [0, 1], [[5], [5]], True, [6.2680227, 5.1520511]),
         (True, [0, 1], None, True, []),
         (True, [0, 1], None, False, [6.2680227, 5.1520511]),
         (True, [5, 203], None, False, [6.2680227, 5.1520511]),
         (False, [5, 203], None, False, [6.2680227, 5.1520511, 8.8596222])])
    def testGetClashes(self, fr, grp, grps, less, expected):
        clashes = fr.getClashes(grp, grps=grps, less=less)
        np.testing.assert_almost_equal(clashes, expected)

    @pytest.mark.parametrize('file,gids,gid,cut,struct',
                             [(HEX_FRM, [0, 1, 5, 203], 0, 5.2, MODIFIED)])
    @pytest.mark.parametrize('srch,grp,less,expected',
                             [(True, [[5]], True, [6.2680227]),
                              (True, None, True, []),
                              (True, None, False, [6.2680227]),
                              (False, None, False, [6.2680227, 8.8596222])])
    def testGetClash(self, fr, gid, grp, less, expected):
        clash = fr.getClash(gid, grp=grp, less=less)
        np.testing.assert_almost_equal(clash, expected)

    @pytest.mark.parametrize('file,cut,struct', [(HEX_FRM, 5.2, MODIFIED)])
    @pytest.mark.parametrize('srch,gids,grp,expected',
                             [(True, [0, 1], [0, 1], False),
                              (True, [0, 1, 5], [0, 1, 5], True),
                              (True, [0, 1], [203], False),
                              (False, [0, 1], [203], True)])
    def testHasClash(self, fr, grp, expected):
        assert expected == fr.hasClash(grp)

    @pytest.mark.parametrize('file,cut,gids,srch',
                             [(HEX_FRM, None, None, None)])
    @pytest.mark.parametrize('struct,expected', [(None, 1), (HEX_RDR, 4)])
    def testExcluded(self, fr, expected):
        assert expected == len(fr.excluded[0])

    @pytest.mark.parametrize('struct,num,incl14,expected',
                             [(None, 3, True, 1), (HEX_RDR, 3, True, 4),
                              (HEX_RDR, 3, False, 3)])
    def testGetExcluded(self, struct, num, incl14, expected):
        incl = dist.Frame.getExcluded(struct=struct, num=num, incl14=incl14)
        assert expected == len(incl[0])

    @pytest.mark.parametrize(
        'file,cut,gids,to_set,srch,struct',
        [(HEX_FRM, None, [], np.array([1, 12]), None, None)])
    def testSet(self, fr, to_set):
        assert not fr.cell.cell.any()
        fr.set(to_set)
        assert 2 == len(fr.gids.on)
        assert 2 == len(fr.cell.cell.nonzero()[0])
        fr.set(to_set, state=False)
        assert 0 == len(fr.gids.on)
        assert 0 == len(fr.cell.cell.nonzero()[0])
