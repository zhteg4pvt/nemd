import ab_lmp_traj_workflow as workflow
import pytest


class TestRunner:

    @pytest.fixture
    def runner(self, original, logger):
        options = workflow.Parser().parse_args(original)
        return workflow.Runner(options=options,
                               original=original,
                               logger=logger)

    @pytest.mark.parametrize(
        'original,expected',
        [(['C'], ['amorp_bldr', 'lammps', 'lmp_traj', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize('original,expected',
                             [(['C'], ['lmp_traj_agg', 'time_agg', 0, 2])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()


class TestParser:

    @pytest.fixture
    def parser(self):
        return workflow.Parser()

    @pytest.mark.parametrize('args,expected', [(['C'], ['C']),
                                               ([], SystemExit)])
    def testParseArgs(self, parser, args, expected, raises):
        with raises:
            options = parser.parse_args(args)
            assert expected == options.cru
