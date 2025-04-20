import pytest

from nemd import envutils
from nemd import test


@pytest.fixture
def cmd(dirname):
    return test.Cmd(envutils.test_data('itest', dirname))


class TestCmd:

    @pytest.mark.parametrize('dirname,expected', [('0001', ['0001', 2, 1]),
                                                  ('0000', ['0000', 2, 2]),
                                                  ('empty', ['empty', 0, 0])])
    def testInit(self, cmd, expected):
        assert expected == [cmd.jobname, len(cmd.raw), len(cmd.args)]

    @pytest.mark.parametrize('dirname,expected',
                             [('0001', 'echo "0001: Amorphous builder on C"'),
                              ('0000', 'echo "0000"'),
                              ('empty', 'echo "empty"')])
    def testPrefix(self, cmd, expected):
        assert expected == cmd.prefix

    @pytest.mark.parametrize('dirname,expected', [('0001', 1), ('0000', 0),
                                                  ('empty', 0)])
    def testCmts(self, cmd, expected):
        assert expected == len(list(cmd.cmts))


class TestParam:

    @pytest.fixture
    def param(self, cmd):
        return test.Cmd(cmd)

    @pytest.mark.parametrize('dirname,expected', [('0001', ['0001', 2, 1]),
                                                  ('0000', ['0000', 2, 2]),
                                                  ('empty', ['empty', 0, 0])])
    def testInit(self, cmd, expected):
        assert expected == [cmd.jobname, len(cmd.raw), len(cmd.args)]
