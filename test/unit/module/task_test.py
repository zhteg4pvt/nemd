import glob
import types
from unittest import mock

import pytest
import test_workflow

from nemd import envutils
from nemd import task

TEST0001 = envutils.test_data('0001')


class TestLmpLog:

    @pytest.fixture
    def lmp_log(self, jobs):
        return task.LmpLog(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('dirname,expected',
                             [('0046_test', 'mol_bldr.data')])
    def testAddfiles(self, lmp_log, expected):
        lmp_log.addfiles()
        assert expected == lmp_log.args[-1]


class TestLmpTraj:

    @pytest.fixture
    def lmp_traj(self, jobs):
        return task.LmpTraj(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('dirname,expected',
                             [('0045_test', 'amorp_bldr.custom.gz')])
    def testAddfiles(self, lmp_traj, expected):
        lmp_traj.addfiles()
        assert expected == lmp_traj.args[0]


class TestCmd:

    @pytest.fixture
    def cmd(self, jobs):
        options = test_workflow.Parser().parse_args([])
        return task.Cmd(*jobs, options=options)

    @pytest.fixture
    def manual(self, args):
        options = test_workflow.Parser().parse_args([])
        cmd = task.Cmd(options=options)
        cmd.args = args
        return cmd

    @pytest.mark.parametrize('dirname,expected', [('0001_test', 1)])
    def testSetArgs(self, cmd, expected):
        cmd.setArgs()
        assert expected == len(cmd.args)

    @pytest.mark.parametrize('dirname,expected', [('0001_test', 0)])
    def testParam(self, cmd, expected):
        assert expected == len(cmd.param.args)

    @pytest.mark.parametrize('dirname,expected', [('0001_test', 1)])
    def testCmd(self, cmd, expected):
        assert expected == len(cmd.cmd.args)

    @pytest.mark.parametrize(
        'args,expected',
        [(["echo hi; CC(C)O -mol_num 1"], ['echo hi; "CC(C)O" -mol_num 1'])])
    def testAddQuot(self, manual, expected):
        manual.addQuot()
        assert expected == manual.args

    @pytest.mark.parametrize('doc_args', [(['-CPU', '2'])])
    @pytest.mark.parametrize(
        'args,options,expected',
        [(["echo hi"], None, ['echo hi']),
         (["nemd_run"], None, ['nemd_run -CPU 2']),
         (["nemd_run -CPU 3"], [], ['nemd_run -CPU 3']),
         (["nemd_run -CPU 3"], [6, 3], ['nemd_run -CPU 2'])])
    def testNumCpu(self, manual, doc_args, options, expected):
        manual.doc = dict(args=doc_args)
        if options:
            manual.options.CPU = options
        manual.numCpu()
        assert expected == manual.args

    @pytest.mark.parametrize(
        'args,options,expected',
        [(["echo hi"], None, ['echo hi']), (["nemd_run"], None, ['nemd_run']),
         (["nemd_run -DEBUG"], None, ['nemd_run -DEBUG']),
         (["nemd_run -DEBUG False"], None, ['nemd_run -DEBUG False']),
         (["nemd_run"], True, ['nemd_run -DEBUG True']),
         (["nemd_run -DEBUG"], False, ['nemd_run -DEBUG False']),
         (["nemd_run -DEBUG True"], False, ['nemd_run -DEBUG False'])])
    def testSetDebug(self, manual, options, expected):
        manual.options.DEBUG = options
        manual.setDebug()
        assert expected == manual.args

    @pytest.mark.parametrize(
        'args,options,expected',
        [(["echo hi"], dict(screen='off', DEBUG=None), False),
         (["nemd_run"], dict(screen='off', DEBUG=None), True),
         (["nemd_run"], dict(screen='off', DEBUG=True), False),
         (["nemd_run"], dict(screen='job', DEBUG=None), False)])
    def testSetScreen(self, manual, options, expected):
        manual.options = types.SimpleNamespace(**options)
        manual.setScreen()
        assert expected == manual.args[0].endswith('/dev/null')

    @pytest.mark.parametrize('dirname,expected', [('0049_test', True),
                                                  ('0049_cmd', False),
                                                  ('0001_cmd', False),
                                                  ('0001_test', True)])
    def testOut(self, cmd, expected):
        assert expected == cmd.out

    @pytest.mark.parametrize('dirname', ['0047_test'])
    def testClean(self, cmd):
        assert cmd.post()
        assert glob.glob('workspace/*/workspace')
        cmd.clean()
        assert not cmd.post()
        assert not glob.glob('workspace/*/workspace')

    @pytest.mark.parametrize('dirname,expected', [('0049_test', 332),
                                                  ('0001_cmd', 161)])
    def testGetCmd(self, cmd, expected):
        cmd.setArgs()
        assert expected == len(cmd.getCmd())
