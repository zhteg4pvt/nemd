import datetime

import pytest

from nemd import timeutils


class TestFunc:

    def testCtime(self):
        assert 19 == len(timeutils.ctime())

    def testDtime(self):
        dtime = timeutils.dtime('13:56:07 02/14/2025')
        assert 7 == dtime.second

    @pytest.mark.parametrize('delta,expected',
                             [(datetime.timedelta(1, 2, 3), '00:00:02'),
                              (None, None)])
    def testDelta2Str(self, delta, expected):
        assert expected == timeutils.delta2str(delta)

    def testStr2Delta(self):
        delta = timeutils.str2delta('12:34:56')
        assert 45296 == delta.seconds
