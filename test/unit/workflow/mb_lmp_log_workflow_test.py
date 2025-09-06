import os

import mb_lmp_log_workflow as workflow
import pytest

from nemd import analyzer
from nemd import envutils


class TestRunner:

    @pytest.fixture
    def runner(self, args, logger):
        options = workflow.Parser().parse_args(args)
        return workflow.Runner(options=options, args=args, logger=logger)

    @pytest.mark.parametrize(
        'args,expected', [(['CCC'], ['mol_bldr', 'lammps', 'lmp_log', 2, 3])])
    def testSetJobs(self, runner, check_flow):
        runner.setJobs()

    @pytest.mark.parametrize(
        'args,expected',
        [(['CCC'], None), (['CCC', '-struct_rg', 'CC'], ['CC']),
         (['CCC', '-struct_rg', 'CC', '1', '2', '0.5'], ['CC 1.0', 'CC 1.5'])])
    def testState(self, runner, expected):
        assert expected == runner.state.get('-substruct')

    @pytest.mark.parametrize('args,expected',
                             [(['CCC'], ['lmp_log_agg', 'time_agg', 1, 2])])
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


class TestRead:

    @pytest.fixture
    def rdr(self, file):
        return workflow.Reader(file)

    @pytest.mark.parametrize(
        'file,smiles,expected',
        [(envutils.test_data('0003_test', 'mol_bldr.log'), 'CCC', '112.40')])
    def testGetSubstruct(self, rdr, smiles, expected, tmp_dir):
        assert expected == rdr.getSubstruct(smiles)


class TestMerger:
    TEST0047 = os.path.join('0047_test', 'workspace',
                            'ecd6407852986c68a9fcc4390d67f50c')

    @pytest.fixture
    def merger(self, tsk, jobs, args):
        options = workflow.Parser().parse_args(args)
        groups = workflow.LmpAgg(*jobs).groups
        Anlz = next(x for x in analyzer.THERMO if x.name == tsk)
        return workflow.Merger(Anlz, groups=groups, options=options)

    @pytest.mark.parametrize('args,tsk',
                             [(['CCCC', '-NAME', 'lmp_log'], 'toteng')])
    @pytest.mark.parametrize('dirname,expected',
                             [('0046_test', 'CCCC dihedral (degree)'),
                              (TEST0047, 'CCCC dihedral (degree)')])
    def testRun(self, tsk, merger, expected):
        merger.run()
        assert expected == merger.data.index.name
