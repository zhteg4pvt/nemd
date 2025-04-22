import os
import shutil
from unittest import mock

import pytest
import test_workflow

from nemd import envutils
from nemd import test


class TestBase:

    @pytest.fixture
    def base(self, dirname):
        Cmd = type('Cmd', (test.Base, ), {})
        return Cmd(envutils.test_data('itest', dirname))

    @pytest.mark.parametrize(
        'dirname,args,expected',
        [('empty', None, 'empty'),
         ('0000', None, '0000: $param;echo hi;echo wa'),
         ('0001', ['1', '2'], '0001: 1;2'),
         ('0001', None,
          '0001: run_nemd amorp_bldr_driver.py C -mol_num 10 -seed 5678')])
    def testGetHeader(self, base, args, expected):
        assert expected == base.getHeader(args=args)

    @pytest.mark.parametrize('dirname,expected', [('0001', 1), ('0000', 3),
                                                  ('empty', 0)])
    def testArgs(self, base, expected):
        assert expected == len(base.args)

    @pytest.mark.parametrize('dirname,expected', [('0001', 2), ('0000', 3),
                                                  ('empty', 0)])
    def testRaw(self, base, expected):
        assert expected == len(base.raw)

    @pytest.mark.parametrize('dirname,expected', [('0001', 1), ('0000', 0),
                                                  ('empty', 0)])
    def testCmts(self, base, expected):
        assert expected == len(list(base.cmts))


class TestCmd:

    @pytest.fixture
    def cmd(self, dirname):
        return test.Cmd(envutils.test_data('itest', dirname))

    @pytest.mark.parametrize('dirname,expected',
                             [('0000', 'echo "0000"'),
                              ('0001', 'echo "0001: Amorphous builder on C"')])
    def testPrefix(self, cmd, expected):
        assert expected == cmd.prefix


class TestParam:

    @pytest.fixture
    def param(self, dirname, args):
        options = test_workflow.Parser().parse_args(args)
        dirpath = envutils.test_data('itest', dirname)
        cmd = test.Cmd(dirpath, options=options)
        return test.Param(cmd)

    @pytest.mark.parametrize('args', [[]])
    @pytest.mark.parametrize('dirname,expected', [
        ('0001', 1),
        ('0000', 2),
        ('empty', 0),
        ('0049', 2),
    ])
    def testCmds(self, param, expected):
        assert expected == len(param.cmds)

    @pytest.mark.parametrize('dirname,args,expected',
                             [('empty', [], 0), ('0049', [], 2),
                              ('0049', ['-slow', '2'], 1)])
    def testArgs(self, param, expected):
        assert expected == len(param.args)

    @pytest.mark.parametrize('args', [[]])
    @pytest.mark.parametrize('dirname,expected',
                             [('empty', 'param'),
                              ('0049', 'number_of_molecules'),
                              ('no_label', 'param'), ('cmd_label', 'mol_num')])
    def testLabel(self, param, expected, tmp_dir):
        assert expected == param.label


class TestCheck:

    @pytest.fixture
    def check(self, dirname, tmp_dir):
        dirpath = envutils.test_data('itest', dirname)
        shutil.copytree(dirpath, os.curdir, dirs_exist_ok=True)
        return test.Check(dirpath, logger=mock.Mock())

    @pytest.mark.parametrize(
        'dirname,expected',
        [('empty', None),
         ('0001', 'polymer_builder.data is different from amorp_bldr.data.\n'),
         ('0046_test', None)])
    def testRun(self, check, expected):
        with open('amorp_bldr.data', 'w'):
            pass
        msg = check.run()
        assert (msg.endswith(expected) if expected else msg is None)
