import datetime

import numpy as np
import pytest

from nemd import timeutils


class TestFunc:

    def testCtime(self):
        assert 19 == len(timeutils.ctime())

    def testDtime(self):
        dtime = timeutils.dtime('13:56:07 02/14/2025')
        assert 7 == dtime.second

    def testDelta2Str(self):
        dtime = datetime.timedelta(1, 2, 3)
        assert '00:00:02' == timeutils.delta2str(dtime)

    def testStr2Delta(self):
        delta = timeutils.str2delta('12:34:56')
        assert 45296 == delta.seconds
