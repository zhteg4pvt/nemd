import glob
import os

import pytest
import test_workflow as workflow

from nemd import envutils
from nemd import logutils

SRC = envutils.Src()


@pytest.mark.skipif(not SRC, reason="test dir not found")
class TestRunner:

    @pytest.fixture
    def runner(self, args, logger, tmp_dir):
        options = workflow.Parser().parse_args(args +
                                               ['-CPU', '1', '-screen', 'off'])
        return workflow.Runner(options=options, args=args, logger=logger)

    @pytest.mark.parametrize(
        'args,expected',
        [(['-task', 'cmd', 'check'], ['cmd', 'check', 1, 2]),
         (['-task', 'cmd'], ['cmd', 0, 1]),
         (['-task', 'check'], ['check', 0, 1]),
         (['-task', 'cmd', 'check', 'tag'], ['cmd', 'check', 'tag', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize('args,expected',
                             [(['1'], (1, )),
                              (['1', '2', '-copy'], (2, '0001', '0002'))])
    def testOpenJobs(self, runner, expected, flow_opr):
        runner.setProj()
        runner.openJobs()
        jobdirs = glob.glob('workspace/*')
        copied = glob.glob('workspace/*/000*')
        copied = sorted([os.path.basename(x) for x in copied])
        assert expected == (len(jobdirs), *copied)

    @pytest.mark.parametrize('args,expected', [(['-name', 'integration'], 62),
                                               (['-name', 'scientific'], 20),
                                               (['-name', 'performance'], 11)])
    def testNames(self, runner, expected):
        assert expected == len(runner.names)

    @pytest.mark.parametrize('args,expected', [
        (['1', '-name', 'integration', '-jtype', 'task'
          ], '2 / 2 succeed sub-jobs.'),
        (['1', '-dirname',
          envutils.Src().test(), '-jtype', 'task'], '0 / 1 succeed sub-jobs.')
    ])
    def testLogStatus(self, runner, expected, tmp_dir, flow_opr):
        with logutils.redirect():
            runner.run()
        runner.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('dirname', ['0001_0002'])
    @pytest.mark.parametrize('args,expected',
                             [(['1', '-name', 'integration', '-prj_path', '.'
                                ], [True, 'test_agg', 0, 1]),
                              (['-prj_path', '.'], [False, 'test_agg', 0, 1])])
    def testSetAggs(self, runner, expected, check_flow, copied):
        runner.setAggs()
        assert expected[0] == bool(
            check_flow._OPERATION_FUNCTIONS[0][1]._flow_aggregate._select)
        # This following runs the def select in the def setAggs. But signac
        # cannot clear previous defined def select. (always run the fist filter)
        runner.setAggProj()
        runner.findJobs()
        runner.runProj()

    @pytest.mark.parametrize('args', [[]])
    @pytest.mark.parametrize('dirname,expected', [('0001_cmd', 1),
                                                  ('empty', SystemExit)])
    def testSetAggProj(self, runner, expected, copied, raises):
        with raises:
            runner.setAggProj()
            assert expected == len(runner.proj.find_jobs())

    @pytest.mark.parametrize('args,dirname,expected',
                             [(['1'], '0001_cmd', 1), (['2'], '0001_cmd', 0),
                              (['2', '1'], '0001_cmd', 1)])
    def testFindJobs(self, runner, expected, copied):
        runner.setAggProj()
        runner.findJobs()
        assert expected == len(runner.jobs)


@pytest.mark.skipif(not SRC, reason="test dir not found")
class TestParser:

    @pytest.fixture
    def parser(self, error):
        parser = workflow.Parser()
        parser.error = error
        return parser

    @pytest.mark.parametrize('ekey', ['NEMD_SRC'])
    @pytest.mark.parametrize(
        'args,evalue,expected',
        [(['-name', 'integration'], SRC, envutils.Src().test(
            os.pardir, 'integration')),
         (['-dirname', os.curdir], None, os.curdir), ([], None, SystemExit),
         (['9999', '-name', 'integration'], SRC, SystemExit)])
    def testParseArgs(self, parser, args, expected, env, raises):
        with raises:
            assert parser.parse_args(args).dirname.samefile(expected)
