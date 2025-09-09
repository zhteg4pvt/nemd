import os.path

import numpy as np
import pytest

from nemd import envutils
from nemd import frame
from nemd import lmpfull
from nemd import pbc


class TestBase:

    @pytest.mark.parametrize('data,box,shape',
                             [(None, None, (0, )),
                              (np.array([[1, 2], [3, 4]]), pbc.Box(), None)])
    def testInit(self, data, box, shape):
        base = frame.Base(data, box=box, shape=shape)
        assert base.box is box
        expected = np.zeros(shape) == data if data is None else data
        np.testing.assert_almost_equal(base, expected)

    @pytest.mark.parametrize('span,cut,expected', [([10, 10, 10], 1, True),
                                                   ([10, 10, 9], 1, False)])
    def testLarge(self, span, cut, expected):
        base = frame.Base(None, box=pbc.Box.fromParams(*span))
        assert expected == base.large(cut)


class TestFrame:

    AR_RDR = lmpfull.Reader.fromTest('ar', 'ar100.data')
    TWO_FRMS = envutils.test_data('ar', 'two_frames.custom')
    BROKEN_HEADER = envutils.test_data('ar', 'broken_header.custom')
    BROKEN_ATOMS = envutils.test_data('ar', 'broken_atoms.custom')
    HEX_RDR = lmpfull.Reader.fromTest('hexane_liquid', 'polymer_builder.data')
    HEX_FRM = envutils.test_data('hexane_liquid', 'dump.custom')
    SUB_FRM = envutils.test_data('si', 'sub', 'amorp_bldr.custom')

    @pytest.mark.parametrize('data,step,expected',
                             [(None, None, [(), None]),
                              ([1, 2], 3, [(2, ), 3]),
                              (frame.Frame([4], step=2), None, [(1, ), 2])])
    def testInit(self, data, step, expected):
        frm = frame.Frame(data, step=step)
        assert expected == [frm.shape, frm.step]

    @pytest.mark.parametrize('file,start,expected',
                             [(TWO_FRMS, 0, (1000, 100)),
                              (TWO_FRMS, 1, (1000, 100)),
                              (BROKEN_HEADER, 0, EOFError),
                              (BROKEN_ATOMS, 1, EOFError),
                              (SUB_FRM, 0, (1, 6))])
    def testRead(self, file, start, expected, raises):
        with open(file) as fh:
            frm = frame.Frame.read(fh, start=start)
            assert 0 == frm.step
            assert not start == isinstance(frm, frame.Frame)
            with raises:
                frm = frame.Frame.read(fh, start=start)
                assert expected == (frm.step, frm.shape[0])

    @pytest.mark.parametrize('file', [TWO_FRMS])
    def testGetCopy(self, frm):
        copied = frm.getCopy()
        assert hasattr(copied, 'box')

    @pytest.mark.parametrize('file,dreader', [(HEX_FRM, HEX_RDR)])
    @pytest.mark.parametrize('broken_bonds,expected', [(True, 0.0002537),
                                                       (False, -2.25869)])
    def testWrap(self, frm, broken_bonds, dreader, expected):
        frm.wrap(broken_bonds=broken_bonds, dreader=dreader)
        np.testing.assert_almost_equal(frm.min(), expected)
        # Scientific test: bond length; atom or molecule centroid within box
        inbox = (frm.max(axis=0) - frm.min(axis=0)) < frm.box.span
        assert broken_bonds == inbox.all()
        atom1s, atom2s = dreader.bonds[['atom1', 'atom2']].transpose().values
        mbond = np.linalg.norm(frm[atom1s, :] - frm[atom2s, :], axis=1).max()
        assert broken_bonds == (mbond > 2)
        if mbond > 2:
            return
        centers = [frm[x, :].mean(axis=0) for x in dreader.mols.values()]
        centers = np.array(centers)
        inbox = (centers.max(axis=0) - centers.min(axis=0)) < frm.box.span
        assert inbox.all()

    @pytest.mark.parametrize('file', [HEX_FRM])
    @pytest.mark.parametrize('expected', [-40.274934])
    def testCenter(self, frm, expected):
        frm.center()
        np.testing.assert_almost_equal(frm.min(), expected, decimal=6)

    @pytest.mark.parametrize('file', [TWO_FRMS])
    @pytest.mark.parametrize('dreader,visible,points,expected',
                             [(None, [3, 4], None, [4, 'X', 'X']),
                              (AR_RDR, None, [[1, 2, 3]], [103, 'Ar', 'X'])])
    def testWrite(self, frm, dreader, visible, points, expected, tmp_dir):
        with open('file', 'w') as fh:
            frm.write(fh, dreader=dreader, visible=visible, points=points)
        with open('file', 'r') as fh:
            lines = fh.readlines()
        assert expected == [len(lines)] + [x.split()[0] for x in lines[-2:]]
