import os
import shutil
from unittest import mock

import numpy as np
import pytest
import test_workflow

from nemd import envutils
from nemd import test


class TestBase:

    @pytest.fixture
    def base(self, dirname):
        Cmd = type('Cmd', (test.Base, ), {})
        return Cmd(envutils.test_data(dirname))

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
        return test.Cmd(envutils.test_data(dirname))

    @pytest.mark.parametrize('dirname,expected',
                             [('0000', 'echo "0000"'),
                              ('0001', 'echo "0001: Amorphous builder on C"')])
    def testPrefix(self, cmd, expected):
        assert expected == cmd.prefix


class TestParam:

    @pytest.fixture
    def param(self, dirname, args):
        options = test_workflow.Parser().parse_args(args)
        dirpath = envutils.test_data(dirname)
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
                              ('0049', 'Number_of_Molecules'),
                              ('no_label', 'param'), ('cmd_label', 'mol_num')])
    def testLabel(self, param, expected, tmp_dir):
        assert expected == param.label


class TestCheck:

    @pytest.fixture
    def check(self, dirname, tmp_dir):
        dirpath = envutils.test_data(dirname)
        shutil.copytree(dirpath, os.curdir, dirs_exist_ok=True)
        return test.Check(dirpath, logger=mock.Mock())

    @pytest.mark.parametrize(
        'dirname,expected',
        [('empty', '| finished   |\n|------------|\n'), ('empty_check', None),
         ('0001', 'polymer_builder.data is different from amorp_bldr.data.\n'),
         ('0046_test', None), ('returncode', 'non-zero return code')])
    def testRun(self, check, expected):
        with open('amorp_bldr.data', 'w'):
            pass
        msg = check.run()
        assert (msg.endswith(expected) if expected else msg is None)

    @pytest.mark.parametrize(
        'dirname,expected',
        [('empty', ['nemd_check collect finished dropna=False']),
         ('empty_check', []),
         ('0001', [
             '# polymers are built the same',
             'nemd_check cmp polymer_builder.data amorp_bldr.data'
         ])])
    def testRaw(self, check, expected):
        assert expected == check.raw


class TestTag:

    TEST0001 = os.path.join('0001_test', 'workspace',
                            '0aee44e791ffa72655abcc90e25355d8')
    TEST0049 = os.path.join('0049_test', 'workspace',
                            '3ec5394f589c9363bd15af35d45a7c44')

    @pytest.fixture
    def tag(self, dirname, args, logger):
        dirname = envutils.test_data(dirname)
        options = test_workflow.Parser().parse_args(args)
        return test.Tag(dirname, options=options, logger=logger)

    @pytest.fixture
    def ntag(self, copied, tmp_dir, logger):
        return test.Tag(os.curdir, logger=logger)

    @pytest.mark.parametrize(
        'dirname,expected', [('empty', None),
                             (TEST0001, ['amorp_bldr', '00:00:01']),
                             (TEST0049, [100, '00:00:01', 50000, '00:00:02'])])
    def testSetSlow(self, ntag, expected):
        ntag.setSlow()
        if expected is None:
            assert ntag.tags.get('slow') is None
            return
        to_compare = ntag.tags.get('slow')
        for idx in range(2):
            np.testing.assert_equal(to_compare[idx::2], expected[idx::2])

    @pytest.mark.parametrize('dirname,expected', [('empty', 0), (TEST0001, 1),
                                                  (TEST0049, 2)])
    def testCollected(self, ntag, expected):
        assert expected == len(ntag.collected)

    @pytest.mark.parametrize('args', [[]])
    @pytest.mark.parametrize('dirname,expected', [('empty', 0), ('0001', 2),
                                                  ('0049', 2)])
    def testTags(self, tag, expected):
        assert expected == len(tag.tags)

    @pytest.mark.parametrize('dirname,expected',
                             [('empty', None), (TEST0001, ['amorp_bldr']),
                              (TEST0049, ['number_of_molecules'])])
    def testSetLabel(self, ntag, expected):
        ntag.setLabel()
        assert expected == ntag.tags.get('label')

    @pytest.mark.parametrize('dirname,expected', [('empty', 0), (TEST0001, 2),
                                                  (TEST0049, 2)])
    def testWrite(self, ntag, expected):
        ntag.setSlow()
        ntag.setLabel()
        ntag.write()
        tag = test.Tag(os.curdir)
        assert expected == len(tag.tags)

    @pytest.mark.parametrize('dirname,args,expected', [
        ('0000', [], True),
        ('0000', ['-label', 'mol_bldr'], False),
        ('0001', ['-label', 'amorp_bldr', '-slow', '1'], False),
        ('0001', ['-label', 'amorp_bldr', '-slow', '4'], True),
    ])
    def testSelected(self, tag, expected):
        assert expected == tag.selected

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0000', ['-slow', '4'], None),
                              ('0001', [], None),
                              ('0001', ['-slow', '4'], {'bldr'}),
                              ('0001', ['-slow', '1'], set())])
    def testFast(self, tag, expected):
        assert expected == tag.fast

    @pytest.mark.parametrize('dirname,args,expected', [
        ('0000', [], True),
        ('0000', ['-label', 'amorp_bldr'], False),
        ('0001', ['-label', 'amorp_bldr'], True),
        ('0001', ['-label', 'mol', 'amorp'], True),
        ('0001', ['-label', 'mol'], False),
    ])
    def testLabeled(self, tag, expected):
        assert expected == tag.labeled
