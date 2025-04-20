import pytest

from nemd import envutils
from nemd import test


@pytest.fixture
def cmd(dirname):
    return test.Cmd(envutils.test_data('itest', dirname))


class TestBase:

    @pytest.fixture
    def base(self, dirname):
        Cmd = type('Cmd', (test.Base, ), {})
        return Cmd(envutils.test_data('itest', dirname))

    @pytest.mark.parametrize('dirname,expected', [('0001', ['0001', 2, 1]),
                                                  ('0000', ['0000', 2, 2]),
                                                  ('empty', ['empty', 0, 0])])
    def testInit(self, base, expected):
        assert expected == [base.jobname, len(base), len(base.args)]

    @pytest.mark.parametrize('dirname,expected',
                             [('0001', '0001: Amorphous builder on C'),
                              ('0000', '0000: echo hi echo wa'),
                              ('empty', 'empty')])
    def testPrefix(self, base, expected):
        assert expected == base.prefix

    @pytest.mark.parametrize('dirname,expected', [('0001', 1), ('0000', 0),
                                                  ('empty', 0)])
    def testCmts(self, base, expected):
        assert expected == len(list(base.cmts))


class TestCmd:

    @pytest.mark.parametrize('dirname,expected',
                             [('0001', 'echo "0001: Amorphous builder on C"')])
    def testPrefix(self, cmd, expected):
        assert expected == cmd.prefix
