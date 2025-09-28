import datetime

import pytest

from nemd import timeutils


class TestFunc:

    def testCtime(self):
        assert 19 == len(timeutils.ctime())

    def testDtime(self):
        dtime = timeutils.dtime('13:56:07 02/14/2025')
        assert 7 == dtime.second

    @pytest.mark.parametrize(
        'delta,expected', [(datetime.timedelta(1, 2, 3), '1 days, 00:00:02'),
                           (datetime.timedelta(0, 1, 2), '00:00:01'),
                           (None, 'nan')])
    def testDelta2Str(self, delta, expected):
        assert expected == timeutils.delta2str(delta)

    @pytest.mark.parametrize('delta,expected', [('12:34:56', 45296.0),
                                                ('1 days, 00:00:02', 86402.0)])
    def testStr2Delta(self, delta, expected):
        delta = timeutils.str2delta(delta)
        assert expected == delta.total_seconds()
