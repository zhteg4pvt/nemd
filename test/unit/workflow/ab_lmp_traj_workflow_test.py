import ab_lmp_traj_workflow as workflow
import pytest


class TestRunner:

    @pytest.fixture
    def runner(self, args, logger):
        options = workflow.Parser().parse_args(args)
        return workflow.Runner(options=options, args=args, logger=logger)

    @pytest.mark.parametrize(
        'args,expected', [(['C'], ['amorp_bldr', 'lammps', 'lmp_traj', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize('args,expected',
                             [(['C'], ['lmp_traj_agg', 'time_agg', 0, 2])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()


class TestParser:

    @pytest.fixture
    def parser(self, error):
        parser = workflow.Parser()
        parser.error = error
        return parser

    @pytest.mark.parametrize('args,expected', [(['C'], ['C']),
                                               ([], SystemExit)])
    def testParseArgs(self, parser, args, expected, raises):
        with raises:
            options = parser.parse_args(args)
            assert expected == options.cru
