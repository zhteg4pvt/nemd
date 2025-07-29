import cb_lmp_log_workflow as workflow
import pytest

import numpy as np


class TestRunner:

    @pytest.fixture
    def runner(self, original, logger):
        options = workflow.Parser().parse_args(original)
        return workflow.Runner(options=options,
                               original=original,
                               logger=logger)

    @pytest.mark.parametrize(
        'original,expected',
        [([], ['crystal_builder', 'lammps_runner', 'lmp_log', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize(
        'original,expected',
        [(['-scale_range', '0.95', '1.05', '0.05'], [0.95, 1.0, 1.05])])
    def testState(self, runner, expected):
        np.testing.assert_almost_equal(runner.state['-scale_factor'], expected)

    @pytest.mark.parametrize('original,expected',
                             [([], ['lmp_log_agg', 'time_agg', 0, 2])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()


class TestParser:

    @pytest.fixture
    def parser(self):
        return workflow.Parser()

    @pytest.mark.parametrize(
        'args,expected',
        [(['-scale_range', '0.9', '1.2', '0.1'], [0.9, 1.2, 0.1]),
         (['-scale_range', '0.9', '1.2'], SystemExit)])
    def testParseArgs(self, parser, args, expected, raises):
        with raises:
            options = parser.parse_args(args)
            assert expected == options.scale_range
