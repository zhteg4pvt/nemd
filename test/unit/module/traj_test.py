import os

import numpy as np
import pytest

from nemd import envutils
from nemd import frame
from nemd import parserutils
from nemd import traj

XTC = envutils.test_data('hexane_xtc', 'amorp_bldr.xtc')
ARGS = ['-last_pct', '0.8', '-task', 'xyz']


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


class TestTime:

    @pytest.mark.parametrize('time,args,expected',
                             [([], None, None), ([0, 1, 2], None, 0),
                              ([0, 1, 2], [XTC, *ARGS], 1)])
    def testNew(self, time, args, expected):
        options= None if args is None else parserutils.LmpTraj().parse_args(args)
        time = traj.Time(time, options=options)
        assert expected == time.start


class TestTraj:

    HEX = envutils.test_data('hexane_liquid')
    FRM = os.path.join(HEX, 'dump.custom')
    GZ = os.path.join(HEX, 'dump.custom.gz')

    @pytest.fixture
    def trj(self, file, args, start):
        options = None if args is None else parserutils.LmpTraj().parse_args([file] + args)
        return traj.Traj(file, options=options, start=start, delay=True)

    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, ['-task', 'xyz'], 0, (1, 105.0)),
                              (GZ, ARGS, 0, (46, 105.0)),
                              (GZ, ARGS, None, (38, 105.0)),
                              (XTC, ARGS, None,
                               (1402, 1400.0))])
    def testSetUp(self, trj, expected):
        trj.setUp()
        num = len([x for x in trj if isinstance(x, frame.Frame)])
        assert expected == (num, trj.time[-1])

    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, None, 0, 0), (FRM, None, None, 0),
                              (GZ,ARGS, None, 7000),
                              (XTC, ARGS, None, 0)])
    def testSetStart(self, trj, expected):
        trj.setStart()
        assert expected == trj.start

    @pytest.mark.parametrize('file,args,start,expected',
                             [(FRM, None, 105000, True),
                              (FRM, None, 105001, False)])
    def testFrame(self, trj, expected):
        frm = next(trj.frame)
        assert expected == isinstance(frm, frame.Frame)

    @pytest.mark.parametrize('file,args,start,expected',
                             [(GZ, None, 0, 46),
                              (GZ, ARGS, None, 37)])
    def testSel(self, trj, expected):
        trj.setUp()
        assert expected == len(trj.sel)
