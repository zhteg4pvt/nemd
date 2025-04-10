import os

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

    @pytest.fixture
    def trj(self, file, options, start):
        return traj.Traj(file, options=options, start=start, delay=True)

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(FRM, None, 0, 1),
                              (GZ, ['-last_pct', '0.8'], 0, 46),
                              (GZ, ['-last_pct', '0.8'], None, 38)])
    def testSetup(self, trj, expected):
        trj.setUp()
        assert expected == len([x for x in trj if isinstance(x, frame.Frame)])

    @pytest.mark.parametrize('file,opts,start,expected',
                             [(FRM, None, 0, 0), (FRM, None, None, 0),
                              (GZ, ['-last_pct', '0.8'], None, 7000)])
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
