import mb_lmp_log_workflow as workflow
import pytest


class TestRunner:

    @pytest.fixture
    def runner(self, args, logger):
        options = workflow.Parser().parse_args(args)
        return workflow.Runner(options=options,
                               args=args,
                               logger=logger)

    @pytest.mark.parametrize(
        'args,expected',
        [(['CCC'], ['mol_bldr', 'lammps', 'lmp_log', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize(
        'args,expected',
        [(['CCC'], None), (['CCC', '-struct_rg', 'CC'], ['CC']),
         (['CCC', '-struct_rg', 'CC', '1', '2', '0.5'], ['CC 1.0', 'CC 1.5'])])
    def testState(self, runner, expected):
        assert expected == runner.state.get('-substruct')

    @pytest.mark.parametrize('args,expected',
                             [(['CCC'], ['lmp_log_agg', 'time_agg', 0, 2])])
    def testSetAggs(self, runner, expected, check_flow):
        runner.setAggs()


class TestParser:

    @pytest.fixture
    def parser(self):
        return workflow.Parser()

    @pytest.mark.parametrize(
        'args,expected',
        [(['CCC', '-struct_rg', 'CC', '1', '2', '0.5'], ('CC', 1.0, 2.0, 0.5)),
         (['CCC', '-struct_rg', 'CC'], ('CC', )),
         (['CCC', '-struct_rg', 'CC', '1', '2'], SystemExit)])
    def testParseArgs(self, parser, args, expected, raises):
        with raises:
            options = parser.parse_args(args)
            assert expected == options.struct_rg
