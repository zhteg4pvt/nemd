import os

import numpy as np
import pytest

from nemd import envutils
from nemd import frame
from nemd import parserutils
from nemd import traj


@pytest.fixture
def options(opts):
    if opts is None:
        return
    parser = parserutils.LmpTraj(delay=True)
    parser.add(parser)
    return parser.parse_args(opts)


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

    @pytest.mark.parametrize('args,opts,expected',
                             [([], None, None), ([0, 1, 2], None, 0),
                              ([0, 1, 2], ['-last_pct', '0.8'], 1)])
    def testNew(self, args, options, expected):
        time = traj.Time(args, options=options)
        assert expected == time.start


class TestTraj:

    HEX = envutils.test_data('hexane_liquid')
    FRM = os.path.join(HEX, 'dump.custom')
    GZ = os.path.join(HEX, 'dump.custom.gz')
    XTC = envutils.test_data('hexane_xtc', 'amorp_bldr.xtc')

    @pytest.fixture
    def trj(self, file, options, start):
        return traj.Traj(file, options=options, start=start, delay=True)

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(FRM, None, 0, (1, 105.0)),
                              (GZ, ['-last_pct', '0.8'], 0, (46, 105.0)),
                              (GZ, ['-last_pct', '0.8'], None, (38, 105.0)),
                              (XTC, ['-last_pct', '0.8'], None,
                               (1402, 1400.0))])
    def testSetup(self, trj, expected):
        trj.setUp()
        num = len([x for x in trj if isinstance(x, frame.Frame)])
        assert expected == (num, trj.time[-1])

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(FRM, None, 0, 0), (FRM, None, None, 0),
                              (GZ, ['-last_pct', '0.8'], None, 7000),
                              (XTC, ['-last_pct', '0.8'], None, 0)])
    def testSetStart(self, trj, expected):
        trj.setStart()
        assert expected == trj.start

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(FRM, None, 105000, True),
                              (FRM, None, 105001, False)])
    def testFrame(self, trj, expected):
        frm = next(trj.frame)
        assert expected == isinstance(frm, frame.Frame)

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(GZ, None, None, 46),
                              (GZ, ['-last_pct', '0.8'], None, 37)])
    def testSel(self, trj, expected):
        trj.setUp()
        assert expected == len(trj.sel)
