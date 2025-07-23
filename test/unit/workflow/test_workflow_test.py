import os

import pytest
import test_workflow as workflow

from nemd import envutils

SRC = envutils.get_src()


@pytest.mark.skipif(SRC is None, reason="test dir not found")
class TestRunner:

    @pytest.fixture
    def runner(self, original, logger):
        options = workflow.Parser().parse_args(original + ['-CPU', '1'])
        return workflow.Runner(options=options,
                               original=original,
                               logger=logger)

    @pytest.mark.parametrize(
        'original,expected',
        [(['-task', 'cmd', 'check'], ['cmd', 'check', 1, 2]),
         (['-task', 'cmd'], ['cmd', 0, 1]),
         (['-task', 'check'], ['check', 0, 1]),
         (['-task', 'cmd', 'check', 'tag'], ['cmd', 'check', 'tag', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize('original,expected',
                             [(['-name', 'integration'], 57),
                              (['-name', 'scientific'], 19),
                              (['-name', 'performance'], 10)])
    def testNames(self, runner, expected):
        assert expected == len(runner.names)

    @pytest.mark.parametrize('original,expected', [
        (['1', '-name', 'integration', '-jtype', 'task'
          ], '2 / 2 succeed sub-jobs.'),
        (['1', '-dirname',
          envutils.test_data(), '-jtype', 'task'], '0 / 2 succeed sub-jobs.')
    ])
    def testLogStatus(self, runner, expected, tmp_dir, flow_opr):
        runner.run()
        runner.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize(
        'original,expected',
        [(['1', '-name', 'integration'], [True, 'test_agg', 0, 1]),
         ([], [False, 'test_agg', 0, 1])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()
        assert expected[0] == bool(
            check_flow._OPERATION_FUNCTIONS[0][1]._flow_aggregate._select)

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize('dirname,expected', [('0001_cmd', 1),
                                                  ('empty', SystemExit)])
    def testSetAggProj(self, runner, expected, copied, raises):
        with raises:
            runner.setAggProj()
            assert expected == len(runner.proj.find_jobs())

    @pytest.mark.parametrize('original,dirname,expected',
                             [(['1'], '0001_cmd', 1), (['2'], '0001_cmd', 0),
                              (['2', '1'], '0001_cmd', 1)])
    def testFindJobs(self, runner, expected, copied):
        runner.setAggProj()
        runner.findJobs()
        assert expected == len(runner.jobs)


@pytest.mark.skipif(SRC is None, reason="test dir not found")
class TestParser:

    @pytest.fixture
    def parser(self):
        return workflow.Parser()

    @pytest.mark.parametrize('ekey', ['NEMD_SRC'])
    @pytest.mark.parametrize('args,evalue,expected',
                             [(['-name', 'integration'], SRC,
                               envutils.get_src('test', 'integration')),
                              (['-dirname', os.curdir], None, os.curdir),
                              ([], None, SystemExit)])
    def testParseArgs(self, parser, args, expected, env, raises):
        with raises:
            assert parser.parse_args(args).dirname.samefile(expected)
