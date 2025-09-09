import os

import conftest
import numpy as np
import pytest

from nemd import envutils
from nemd import frame
from nemd import parserutils
from nemd import traj

ARGS = ['-last_pct', '0.8', '-task', 'xyz']
XTC = envutils.test_data('hexane_xtc', 'amorp_bldr.xtc')


class TestBox:

    @pytest.fixture
    def box(self, vecs):
        return traj.Box(np.transpose(vecs))

    @pytest.mark.parametrize('vecs,expected',
                             [([[1, 0, 0], [0, 2, 0], [0, 0, 3]], 6),
                              ([[1, 0, 0], [1, 1, 0], [1, 2, 3]], 3)])
    def testVolume(self, box, expected):
        np.testing.assert_almost_equal(box.volume, expected)

    @pytest.mark.parametrize('vecs,expected',
                             [([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [1, 2, 3]),
                              ([[1, 0, 0], [1, 1, 0], [1, 2, 3]], [1, 1, 3])])
    def testSpan(self, box, expected):
        np.testing.assert_almost_equal(box.span, expected)

    @pytest.mark.parametrize(
        'vecs,vec,expected',
        [([[1, 0, 0], [0, 2, 0], [0, 0, 3]], [0.5, 1, 1.5], 1.87082869),
         ([[1, 0, 0], [1, 1, 0], [1, 2, 3]], [0.5, 1, 1.5], 1.58113883)])
    def testNorms(self, box, vec, expected):
        np.testing.assert_almost_equal(box.norms(np.array([vec]))[0], expected)

    @pytest.mark.parametrize('vecs,expected',
                             [([[1, 0, 0], [0, 2, 0], [0, 0, 3]], (12, 2, 3)),
                              ([[1, 0, 0], [1, 1, 0], [1, 2, 3]], (12, 2, 3))])
    def testEdges(self, box, expected):
        assert expected == box.edges.shape


@conftest.require_src
class TestTime:

    @pytest.mark.parametrize('time,args,expected',
                             [([], None, None), ([0, 1, 2], None, 0),
                              ([0, 1, 2], [XTC, *ARGS], 1)])
    def testNew(self, time, args, expected):
        options = None if args is None else parserutils.LmpTraj().parse_args(
            args)
        time = traj.Time(time, options=options)
        assert expected == time.start


@conftest.require_src
class TestTraj:

    FRM = envutils.test_data('hexane_liquid', 'dump.custom')
    GZ = envutils.test_data('hexane_liquid', 'dump.custom.gz')
    EMPTY = envutils.test_data('ar', 'empty.custom')
    ONE = envutils.test_data('ar', 'one_frame.custom')

    @pytest.fixture
    def trj(self, file, args, start, delay):
        options = None if args is None else parserutils.LmpTraj().parse_args(
            [file] + args)
        return traj.Traj(file, options=options, start=start, delay=delay)

    @pytest.mark.parametrize('delay', [False])
    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, ['-task', 'xyz'], 0, (1, 105.0)),
                              (GZ, ARGS, 0, (46, 105.0)),
                              (GZ, ARGS, None, (38, 105.0)),
                              (XTC, ARGS, None, (255, 254.0))])
    def testSetUp(self, trj, expected):
        num = len([x for x in trj if isinstance(x, frame.Frame)])
        assert expected == (num, trj.time[-1])

    @pytest.mark.parametrize('delay', [True])
    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, None, 0, 0), (FRM, None, None, 0),
                              (EMPTY, ['-task', 'xyz'], None, 0),
                              (ONE, ['-task', 'xyz'], None, 0),
                              (GZ, ARGS, None, 7000), (XTC, ARGS, None, 0)])
    def testSetStart(self, trj, expected):
        trj.setStart()
        assert expected == trj.start

    @pytest.mark.parametrize('delay', [True])
    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, None, 105000, True),
                              (FRM, None, 105001, False)])
    def testFrame(self, trj, expected):
        frm = next(trj.frame)
        assert expected == isinstance(frm, frame.Frame)

    @pytest.mark.parametrize('delay', [False])
    @pytest.mark.parametrize('file,args,start,expected',
                             [(GZ, None, 0, 46), (GZ, ARGS, None, 37)])
    def testSel(self, trj, expected):
        assert expected == len(trj.sel)
