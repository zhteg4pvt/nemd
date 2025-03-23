import os.path

import numpy as np
import pytest

from nemd import envutils
from nemd import frame
from nemd import lmpfull
from nemd import pbc


class TestBase:

    @pytest.fixture
    def base(self, data, box, shape):
        return frame.Base(data=data, box=box, shape=shape)

    @pytest.mark.parametrize('data,box,shape',
                             [(None, None, (0, )),
                              (np.array([[1, 2], [3, 4]]), pbc.Box(), None)])
    def testInit(self, data, box, shape, base):
        assert base.box is box
        expected = np.zeros(shape) == data if data is None else data
        np.testing.assert_almost_equal(base, expected)

    @pytest.mark.parametrize('data,box,shape,expected',
                             [([[0.1, 0.2, 0.4], [0.9, 0.8, 0.6]],
                               pbc.Box.fromParams(1.), None, [0.48989795])])
    @pytest.mark.parametrize('grp,grps', [(None, None), ([0, 1], None),
                                          ([0], [[1]])])
    def testPairDists(self, base, grp, grps, expected):
        dists = base.pairDists(grp=grp, grps=grps)
        np.testing.assert_almost_equal(dists, expected)


class TestFrame:

    AR_READER = lmpfull.Reader(envutils.test_data('ar', 'ar100.data'))
    TWO_FRAMES = envutils.test_data('ar', 'two_frames.custom')
    BROKEN_HEADER = envutils.test_data('ar', 'broken_header.custom')
    BROKEN_ATOMS = envutils.test_data('ar', 'broken_atoms.custom')
    HEX = envutils.test_data('hexane_liquid')
    HEXANE_READER = lmpfull.Reader(os.path.join(HEX, 'polymer_builder.data'))
    HEXANE_FRAME = os.path.join(HEX, 'dump.custom')

    @pytest.fixture
    def frm(self, file):
        with open(file, 'r') as fh:
            return frame.Frame.read(fh)

    @pytest.mark.parametrize('data,step,expected',
                             [(None, None, [(0, ), None]),
                              ([1, 2], 3, [(2, ), 3]),
                              (frame.Frame([4], step=2), None, [(1, ), 2])])
    def testInit(self, data, step, expected):
        frm = frame.Frame(data=data, step=step)
        assert expected == [frm.shape, frm.step]

    @pytest.mark.parametrize('file,start,expected',
                             [(TWO_FRAMES, 0, True), (TWO_FRAMES, 1, False),
                              (BROKEN_HEADER, 0, EOFError),
                              (BROKEN_ATOMS, 1, EOFError)])
    def testRead(self, file, start, expected, raises):
        with open(file) as fh:
            frms = [frame.Frame.read(fh, start=start)]
            with raises:
                frms.append(frame.Frame.read(fh, start=start))
        assert 0 == frms[0].step
        assert not start == isinstance(frms[0], frame.Frame)
        if not isinstance(expected, bool):
            return
        assert 1000 == frms[1].step
        assert 100 == frms[1].shape[0]

    @pytest.mark.parametrize('file', [TWO_FRAMES])
    @pytest.mark.parametrize('array', [True, False])
    def testCopy(self, frm, array):
        copied = frm.copy(array=array)
        assert not array == isinstance(copied, frame.Frame)

    @pytest.mark.parametrize('file,dreader', [(HEXANE_FRAME, HEXANE_READER)])
    @pytest.mark.parametrize('broken_bonds,expected', [(True, 0.0002537),
                                                       (False, -2.25869)])
    def testWrap(self, frm, broken_bonds, dreader, expected):
        frm.wrap(broken_bonds=broken_bonds, molecules=dreader.molecules)
        np.testing.assert_almost_equal(frm.min(), expected)
        # Scientific test: bond length; atom or molecule centroid within box
        inbox = (frm.max(axis=0) - frm.min(axis=0)) < frm.box.span
        assert broken_bonds == inbox.all()
        atom1s, atom2s = dreader.bonds[['atom1', 'atom2']].transpose().values
        mbond = np.linalg.norm(frm[atom1s, :] - frm[atom2s, :], axis=1).max()
        assert broken_bonds == (mbond > 2)
        if mbond > 2:
            return
        centers = [frm[x, :].mean(axis=0) for x in dreader.molecules.values()]
        centers = np.array(centers)
        inbox = (centers.max(axis=0) - centers.min(axis=0)) < frm.box.span
        assert inbox.all()

    @pytest.mark.parametrize('file', [HEXANE_FRAME])
    @pytest.mark.parametrize('dreader,expected',
                             [(None, -37.6929), (HEXANE_READER, -40.8407627)])
    def testGlue(self, frm, dreader, expected):
        frm.glue(molecules=dreader.molecules if dreader else None)
        np.testing.assert_almost_equal(frm.min(), expected)

    @pytest.mark.parametrize('file', [TWO_FRAMES])
    @pytest.mark.parametrize(
        'dreader,visible,points,expected',
        [(None, [3, 4], None, [4, 'X', 'X']),
         (AR_READER, None, [[1, 2, 3]], [103, 'Ar', 'X'])])
    def testWrite(self, frm, dreader, visible, points, expected, tmp_dir):
        with open('file', 'w') as fh:
            frm.write(fh, dreader=dreader, visible=visible, points=points)
        with open('file', 'r') as fh:
            lines = fh.readlines()
        assert expected == [len(lines)] + [x.split()[0] for x in lines[-2:]]
