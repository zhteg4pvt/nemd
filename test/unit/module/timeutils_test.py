import pytest

from nemd import timeutils


class TestDateTime:

    @pytest.fixture
    def now(self):
        return timeutils.Date.now()

    def testStrftime(self, now):
        assert 19 == len(now.strftime())

    @pytest.mark.parametrize('input,expected', [('13:56:07 02/14/2025', 7)])
    def testStrptime(self, input, expected):
        assert expected == timeutils.Date.strptime(input).second


class TestTimeDelta:

    @pytest.mark.parametrize('args,expected', [((1, 2, 3), '1 days, 00:00:02'),
                                               ((0, 1, 2), '00:00:01')])
    def testToStr(self, args, expected):
        assert expected == timeutils.Delta(*args).toStr()

    @pytest.mark.parametrize('delta,expected', [('12:34:56', 45296.0),
                                                ('1 days, 00:00:02', 86402.0)])
    def testFromStr(self, delta, expected):
        assert expected == timeutils.Delta.fromStr(delta).total_seconds()
