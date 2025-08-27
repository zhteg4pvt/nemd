import cb_lmp_log_workflow as workflow
import numpy as np
import pytest


class TestRunner:

    @pytest.fixture
    def runner(self, args, logger):
        options = workflow.Parser().parse_args(args)
        return workflow.Runner(options=options, args=args, logger=logger)

    @pytest.mark.parametrize(
        'args,expected',
        [([], ['crystal_builder', 'lammps_runner', 'lmp_log', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize('args,expected', [
        (['-scale_range', '0.95', '1.05', '5'], [0.95, 0.975, 1., 1.025, 1.05])
    ])
    def testState(self, runner, expected):
        np.testing.assert_almost_equal(runner.state['-scale_factor'], expected)

    @pytest.mark.parametrize('args,expected',
                             [([], ['lmp_log_agg', 'time_agg', 0, 2])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()


class TestParser:

    @pytest.fixture
    def parser(self):
        return workflow.Parser()

    @pytest.mark.parametrize('args,expected',
                             [(['-scale_range', '0.9', '1.2', '4'],
                               (0.9, 1.2, 4)),
                              (['-scale_range', '0.9', '1.2'], (0.9, 1.2, 31)),
                              (['-scale_range', '0.9'], SystemExit)])
    def testParseArgs(self, parser, args, expected, raises):
        with raises:
            options = parser.parse_args(args)
            assert expected == options.scale_range
