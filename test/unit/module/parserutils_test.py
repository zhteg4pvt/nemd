import argparse
import os
from unittest import mock

import pytest

from nemd import envutils
from nemd import parserutils
from nemd import pytestutils

AR_DIR = envutils.test_data('ar')
IN_FILE = os.path.join(AR_DIR, 'ar100.in')
EMPTY_IN = os.path.join(AR_DIR, 'empty.in')
MISS_DATA_IN = os.path.join(AR_DIR, 'single.in')
DATA_FILE = os.path.join(AR_DIR, 'ar100.data')
MY_DATA_FILE = os.path.join(AR_DIR, 'mydata.in')
LOG_FILE = os.path.join(AR_DIR, 'lammps.log')
TRAJ_FILE = os.path.join(AR_DIR, 'ar100.custom.gz')

RAISED = argparse.ArgumentTypeError


@pytestutils.Raises
class TestType:

    @pytest.mark.parametrize('arg,expected', [('not_existing', RAISED),
                                              (TRAJ_FILE, TRAJ_FILE)])
    def testFile(self, arg, expected):
        assert expected == parserutils.type_file(arg)

    @pytest.mark.parametrize('arg,expected', [(AR_DIR, None),
                                              ('not_existing', RAISED),
                                              (TRAJ_FILE, RAISED)])
    def testDir(self, arg, expected):
        parserutils.type_dir(arg)

    @pytest.mark.parametrize('arg,expected', [('y', True), ('n', False),
                                              ('wa', RAISED)])
    def testBool(self, arg, expected):
        assert expected == parserutils.type_bool(arg)

    @pytest.mark.parametrize('arg,expected', [('123', 123), ('5.6', 5.6),
                                              ('wa', RAISED),
                                              ('None', RAISED)])
    def testFloat(self, arg, expected):
        assert expected == parserutils.type_float(arg)

    @pytest.mark.parametrize('arg,expected', [('0', 0), ('1.12', RAISED),
                                              ('-1', -1)])
    def testInt(self, arg, expected):
        assert expected == parserutils.type_int(arg)

    @pytest.mark.parametrize('bottom,top', [(100, 200)])
    @pytest.mark.parametrize('value,included_bottom,include_top,expected',
                             [(123, True, True, 123), (100, True, True, 100),
                              (100, False, True, RAISED),
                              (200, False, True, 200),
                              (200, False, False, RAISED)])
    def testRanged(self, value, bottom, top, included_bottom, include_top,
                   expected):
        assert expected == parserutils.type_ranged(
            value,
            bottom=bottom,
            top=top,
            included_bottom=included_bottom,
            include_top=include_top)

    @pytest.mark.parametrize('bottom,top', [(1.12, 3.45)])
    @pytest.mark.parametrize('arg,included_bottom,include_top,expected',
                             [('2', True, True, 2), ('1.12', True, True, 1.12),
                              ('1.12', False, True, RAISED),
                              ('3.45', False, True, 3.45),
                              ('3.45', False, False, RAISED)])
    def testRangedFloat(self, arg, bottom, top, included_bottom, include_top,
                        expected):
        assert expected == parserutils.type_ranged_float(
            arg,
            bottom=bottom,
            top=top,
            included_bottom=included_bottom,
            include_top=include_top)

    @pytest.mark.parametrize('arg,expected', [('0', 0), ('1.12', 1.12),
                                              ('-1.12', RAISED)])
    def testNonnegativeFloat(self, arg, expected):
        assert expected == parserutils.type_nonnegative_float(arg)

    @pytest.mark.parametrize('arg,expected', [('0', RAISED), ('1.12', 1.12),
                                              ('-1.12', RAISED)])
    def testPositiveFloat(self, arg, expected):
        parserutils.type_positive_float(arg)

    @pytest.mark.parametrize('arg,expected', [('1', 1), ('0', RAISED),
                                              ('-1', RAISED)])
    def testPositiveInt(self, arg, expected):
        assert expected == parserutils.type_positive_int(arg)

    @pytest.mark.parametrize('arg,expected', [('1', 1), ('0', 0),
                                              ('-1', RAISED)])
    def testNonnegativeInt(self, arg, expected):
        assert expected == parserutils.type_nonnegative_int(arg)

    @pytest.mark.parametrize('ekey', ['TQDM_DISABLE'])
    @pytest.mark.parametrize('evalue,arg,expected', [('', 'job', ''),
                                                     ('', 'progress', ''),
                                                     ('1', 'job', '1'),
                                                     ('1', 'progress', '')])
    def testScreen(self, arg, expected, env):
        parserutils.type_screen(arg)
        assert expected == os.environ.get('TQDM_DISABLE')

    @pytest.mark.parametrize('arg,expected', [('0', 0), ('1234', 1234),
                                              ('-1', RAISED), (2**31, RAISED)])
    def testRandomSeed(self, arg, expected):
        assert expected == parserutils.type_random_seed(arg)

    @pytest.mark.parametrize('arg,expected', [('C', 1), ('not_valid', RAISED)])
    def testTypeSmiles(self, arg, expected):
        assert expected == parserutils.type_smiles(arg).GetNumAtoms()

    @pytest.mark.parametrize('arg,allow_reg,expected',
                             [('C', True, 'C'), ('C', False, RAISED),
                              ('*C*', False, '*C[*:1]'),
                              ('*C*.C', False, RAISED), ('C*', True, RAISED)])
    def testCruSmiles(self, arg, allow_reg, expected):
        assert expected == parserutils.type_cru_smiles(arg,
                                                       allow_reg=allow_reg)


class TestLastPct:

    @pytest.fixture
    def last_ptc(self, arg):
        return parserutils.LastPct.type(arg)

    @pytest.mark.parametrize('arg,expected', [('0.2', 0.2), ('0', RAISED),
                                              ('0.99', 0.99), ('1', RAISED)])
    @pytestutils.Raises
    def testType(self, arg, expected):
        assert expected == parserutils.LastPct.type(arg)

    @pytest.mark.parametrize('data', [[0, 1, 2, 3, 4]])
    @pytest.mark.parametrize('arg,buffer,expected', [('0.7', 0, 2),
                                                     ('0.6', 0, 2),
                                                     ('0.6', 1, 1)])
    def testGetSidx(self, data, last_ptc, buffer, expected):
        assert expected == last_ptc.getSidx(data, buffer=buffer)


def error(x):
    raise RAISED(x)


@pytestutils.Raises
class TestAction:

    @pytest.fixture
    def parser(self, action, dtype):
        parser = argparse.ArgumentParser()
        parser.add_argument('dest', nargs='+', type=dtype, action=action)
        parser.error = error
        return parser

    @pytest.mark.parametrize('action,dtype,args,expected',
                             [(parserutils.Action, str, ['1', '2'],
                               ('1', '2'))])
    def testDoTyping(self, parser, args, expected):
        assert expected == parser.parse_args(args).dest

    @pytest.mark.parametrize('action,dtype',
                             [(parserutils.ForceFieldAction, str)])
    @pytest.mark.parametrize('args,expected',
                             [(['SW'], ('SW', )),
                              (['OPLSUA'], ('OPLSUA', 'TIP3P')),
                              (['OPLSUA', 'SPCE'], ('OPLSUA', 'SPCE')),
                              (['OPLSUA', 'NOT_VALID'], RAISED)])
    def testForceField(self, parser, args, expected):
        assert expected == parser.parse_args(args).dest

    @pytest.mark.parametrize(
        'action,dtype',
        [(parserutils.SliceAction, parserutils.type_nonnegative_int)])
    @pytest.mark.parametrize('args,expected', [(['9'], (0, 9, 1)),
                                               (['9', '1'], RAISED),
                                               (['1', '9', '3'], (1, 9, 3)),
                                               (['1', '9', '0'], RAISED)])
    def testSlice(self, parser, args, expected):
        assert expected == parser.parse_args(args).dest

    @pytest.mark.parametrize('action,dtype', [(parserutils.StructAction, str)])
    @pytest.mark.parametrize('args,expected', [(['CCC', '36'], ('CCC', 36)),
                                               (['CC'], ('CC', None)),
                                               (['CCC', 'non_float'], RAISED),
                                               (['non_smiles'], RAISED)])
    def testStruct(self, parser, args, expected):
        assert expected == parser.parse_args(args).dest

    @pytest.mark.parametrize('action,dtype', [(parserutils.ThreeAction, str)])
    @pytest.mark.parametrize('args,expected',
                             [(['1', '2', '3', '4'], ('1', '2', '3')),
                              (['1'], ('1', '1', '1')), (['1', '2'], RAISED),
                              ([], RAISED)])
    def testThree(self, args, parser, expected):
        assert expected == parser.parse_args(args).dest


@pytestutils.Raises
class TestValidator:

    @pytest.fixture
    def parser(self, valid, flags, kwargss):
        if kwargss is None:
            kwargss = [{}]
        if isinstance(kwargss, dict):
            kwargss = [kwargss]
        if len(kwargss) == 1:
            kwargss = kwargss * len(flags)
        parser = parserutils.Driver(valids={valid})
        for flag, kwargs in zip(flags, kwargss):
            parser.add_argument(flag, **(kwargs if kwargs else {}))
        parser.error = error
        return parser

    @pytest.fixture
    def args(self, flags, values):
        fvals = [[x, y] for x, y in zip(flags, values) if y is not None]
        fvals = [[x, *y] if isinstance(y, list) else [x, y] for x, y in fvals]
        return [y for x in fvals for y in x]

    @pytest.mark.parametrize('valid', [parserutils.MolValid])
    @pytest.mark.parametrize('flags', [('-cru', '-cru_num', '-mol_num')])
    @pytest.mark.parametrize('kwargss', [{'nargs': '+'}])
    @pytest.mark.parametrize('values,expected', [
        ((['C'], None, None), ['C', 1, 1]),
        ((['C', 'O'], None, None), ['C', 'O', 1, 1, 1, 1]),
        ((['C', 'O'], ['1', '2'], None), ['C', 'O', '1', '2', 1, 1]),
        ((['C', 'O'], ['1', '2'], ['4', '5']), ['C', 'O', '1', '2', '4', '5']),
        ((['C', 'O'], ['1'], ['4', '5']), RAISED),
        ((['C', 'O'], ['1', '2'], ['5']), RAISED)
    ])
    def testMol(self, parser, args, expected):
        options = parser.parse_args(args)
        assert expected == [*options.cru, *options.cru_num, *options.mol_num]

    @pytest.mark.parametrize('valid', [parserutils.LmpValid])
    @pytest.mark.parametrize('flags', [('-inscript', '-data_file')])
    @pytest.mark.parametrize('kwargss', [None])
    @pytest.mark.parametrize(
        'values,expected',
        [((IN_FILE, None), [IN_FILE, DATA_FILE]),
         ((MISS_DATA_IN, None), RAISED),
         ((MISS_DATA_IN, 'my.data'), [MISS_DATA_IN, 'my.data']),
         ((MY_DATA_FILE, None), [MY_DATA_FILE, None]),
         ((EMPTY_IN, None), [EMPTY_IN, None])])
    def testLmp(self, values, parser, expected, args, tmp_dir):
        options = parser.parse_args(args)
        assert expected == [options.inscript, options.data_file]

    @pytest.mark.parametrize('valid', [parserutils.TrajValid])
    @pytest.mark.parametrize('flags', [(['-task', '-data_file'])])
    @pytest.mark.parametrize('kwargss', [([{'nargs': '+'}, None])])
    @pytest.mark.parametrize(
        'values,expected',
        [((['xyz'], None), ['xyz', None]), ((['density'], None), RAISED),
         ((['density'], 'my.data'), ['density', 'my.data'])])
    def testTraj(self, parser, args, expected):
        options = parser.parse_args(args)
        assert expected == [*options.task, options.data_file]


class TestDriver:

    @pytest.fixture
    def mocked(self, to_mock):
        with mock.patch(f'nemd.parserutils.{to_mock}', mock.Mock()):
            yield parserutils.Driver(valids={parserutils.Valid})

    @pytest.fixture
    def parser(self):
        return parserutils.Driver(valids={parserutils.Valid})

    @pytest.mark.parametrize('to_mock', ['Driver.setUp'])
    def testSetUp(self, mocked):
        mocked.setUp.assert_called_with()

    @pytest.mark.parametrize('to_mock', ['Driver.add'])
    def testAdd(self, mocked):
        mocked.add.assert_called_with(mocked, positional=True)

    @pytest.mark.parametrize('ekey', ['JOBNAME'])
    @pytest.mark.parametrize(
        'evalue,args,expected',
        [(None, [], ['job', 'job']),
         (None, ['-NAME', 'mol_bldr'], ['mol_bldr', 'job']),
         (None, ['-JOBNAME', 'xtal_bldr'], ['job', 'xtal_bldr']),
         ('lammps', [], ['lammps', 'lammps']),
         ('lammps', ['-NAME', 'mol_bldr'], ['mol_bldr', 'lammps']),
         ('lammps', ['-JOBNAME', 'xtal_bldr'], ['lammps', 'xtal_bldr'])])
    def testAddJobname(self, args, expected, env):
        parser = parserutils.Driver(valids={parserutils.Valid})
        options = parser.parse_args(args)
        assert expected == [options.NAME, options.JOBNAME]

    @pytest.mark.parametrize('ekey', ['INTERAC'])
    @pytest.mark.parametrize('evalue,args,expected',
                             [(None, [], False), ('', [], False),
                              ('False', ['-INTERAC'], True),
                              ('True', ['-INTERAC', 'False'], False)])
    def testAddInterac(self, args, expected, env):
        parser = parserutils.Driver(valids={parserutils.Valid})
        options = parser.parse_args(args)
        assert expected == options.INTERAC

    @pytest.mark.parametrize('ekey', ['PYTHON'])
    @pytest.mark.parametrize('evalue,args,expected',
                             [(None, [], '2'), ('1', [], '1'),
                              ('1', ['-PYTHON', '0'], '0')])
    def testAddPython(self, args, expected, env):
        parser = parserutils.Driver(valids={parserutils.Valid})
        options = parser.parse_args(args)
        assert expected == options.PYTHON

    @pytest.mark.parametrize('args,expected', [([], None),
                                               (['-CPU', '2'], [2]),
                                               (['-CPU', '2', '3'], [2, 3])])
    def testAddCpu(self, parser, args, expected):
        options = parser.parse_args(args)
        assert expected == options.CPU

    @pytest.mark.parametrize('ekey', ['DEBUG'])
    @pytest.mark.parametrize('evalue,args,expected',
                             [(None, [], False), ('', [], False),
                              (None, ['-DEBUG'], True),
                              ('1', ['-DEBUG', 'False'], False)])
    def testAddDebug(self, args, expected, env):
        parser = parserutils.Driver(valids={parserutils.Valid})
        options = parser.parse_args(args)
        assert expected == options.DEBUG

    @pytest.mark.parametrize('args,expected', [([], None),
                                               (['-bool', ''], False),
                                               (['-bool'], True),
                                               (['-bool', 'yes'], True),
                                               (['-bool', 'false'], False),
                                               (['-bool', 'False'], False)])
    def testAddBool(self, parser, args, expected):
        parser.addBool('-bool')
        options = parser.parse_args(args)
        assert expected == options.bool

    @pytest.mark.parametrize('args,expected', [([], None),
                                               (['-seed', '12'], 12)])
    def testAddSeed(self, parser, args, expected):
        parser.addSeed(parser)
        seed = parser.parse_args(args).seed
        assert (expected == seed) if expected else isinstance(seed, int)

    @pytest.mark.parametrize('flags,kwargs', [(['-JOBNAME'], {'-NAME': 'hi'})])
    def testSuppress(self, flags, kwargs, parser):
        parser.suppress(flags=flags, **kwargs)
        for dest in ['NAME', 'JOBNAME']:
            action = next(x for x in parser._actions if x.dest == dest)
            assert '==SUPPRESS==' == action.help
            assert not ((dest == 'NAME') ^ ('hi' == action.default))

    @pytest.mark.parametrize('to_mock', ['Valid.run'])
    def testParseArgs(self, mocked):
        options = mocked.parse_args(['-JOBNAME', 'hi'])
        assert 'hi' == options.JOBNAME
        assert next(iter(mocked.valids)).run.called


class TestAdd:

    @pytest.fixture
    def parser(self):
        return parserutils.Driver()

    def testBldr(self, parser):
        parserutils.Bldr.addBldr(parser)
        options = parser.parse_args(['-substruct', 'C', '-force_field', 'SW'])
        assert ('C', None) == options.substruct
        assert ('SW', ) == options.force_field

    def testMolBase(self, parser):
        parserutils.MolBase.add(parser)
        options = parser.parse_args(
            ['*C*', '-cru_num', '2', '-mol_num', '3', '-buffer', '4'])
        assert ['*C[*:1]'] == options.cru
        assert [2] == options.cru_num
        assert [3] == options.mol_num
        assert 4.0 == options.buffer

    @pytest.mark.parametrize(
        'args,expected',
        [([], [1, 10, 300, 1, 1000, 1, 1, 'NVE', False, None, None]),
         ([
             '-timestep', '1.2', '-stemp', '3.1', '-temp', '275', '-press',
             '3.6', '-pdamp', '777', '-relax_time', '4', '-prod_time', '5',
             '-prod_ens', 'NVT', '-no_minimize', '-rigid_bond', '4', '6',
             '-rigid_angle', '3', '5'
         ], [1.2, 3.1, 275, 3.6, 777, 4, 5, 'NVT', True, [4, 6], [3, 5]])])
    def testMd(self, args, expected, parser):
        parserutils.Md.add(parser)
        options = parser.parse_args(args)
        assert expected == [
            options.timestep, options.stemp, options.temp, options.press,
            options.pdamp, options.relax_time, options.prod_time,
            options.prod_ens, options.no_minimize, options.rigid_bond,
            options.rigid_angle
        ]

    def testMolBldr(self, parser):
        parserutils.MolBldr.add(parser)
        options = parser.parse_args(['CCC'])
        assert [44.0, [1], 0, 1, 1, 0, 0, 'NVE'] == [
            options.buffer, options.mol_num, options.temp, options.timestep,
            options.press, options.relax_time, options.prod_time,
            options.prod_ens
        ]

    @pytest.mark.parametrize(
        'args,expected',
        [(['CCC'], [0.5, 'grow']),
         (['CCC', '-density', '1.2', '-method', 'pack'], [1.2, 'pack'])])
    def testAmorpBldr(self, args, expected, parser):
        parserutils.AmorpBldr.add(parser)
        options = parser.parse_args(args)
        assert expected == [options.density, options.method]

    @pytest.mark.parametrize(
        'args,expected',
        [([], ['Si', (1, 1, 1), (1, 1, 1), ['SW']]),
         (['-dimension', '2', '-scale_factor', '0.95', '0.98', '1.02'
           ], ['Si', (2, 2, 2), (0.95, 0.98, 1.02), ['SW']])])
    def testXtalBldr(self, args, expected, parser):
        parserutils.XtalBldr.add(parser)
        options = parser.parse_args(args)
        assert expected == [
            options.name, options.dimension, options.scale_factor,
            options.force_field
        ]

    @pytest.mark.parametrize(
        'args,expected',
        [([EMPTY_IN], [EMPTY_IN, None]),
         ([MISS_DATA_IN, '-data_file', DATA_FILE], [MISS_DATA_IN, DATA_FILE])])
    def testLammps(self, args, expected, parser):
        parserutils.Lammps.add(parser)
        options = parser.parse_args(args)
        assert expected == [options.inscript, options.data_file]

    @pytest.mark.parametrize(
        'args,positional,expected',
        [([], False, [['toteng'], None,
                      parserutils.LastPct(0.2), None]),
         ([
             LOG_FILE, '-task', 'temp', 'e_mol', '-data_file', DATA_FILE,
             '-last_pct', '0.66', '-slice', '1', '8', '3'
         ], True, [['temp', 'e_mol'], DATA_FILE, 0.66, (1, 8, 3)])])
    def testLog(self, args, expected, positional, parser):
        parserutils.LmpLog.add(parser, positional=positional)
        options = parser.parse_args(args)
        assert positional == hasattr(options, 'log')
        assert expected == [
            options.task, options.data_file, options.last_pct, options.slice
        ]

    @pytest.mark.parametrize(
        'args,positional,expected',
        [([], False, [['density'], None,
                      parserutils.LastPct(0.2), None]),
         ([
             TRAJ_FILE, '-task', 'rdf', 'msd', '-data_file', DATA_FILE,
             '-last_pct', '0.66', '-slice', '1', '8', '3'
         ], True, [['rdf', 'msd'], DATA_FILE, 0.66, (1, 8, 3)])])
    def testTraj(self, args, expected, positional, parser):
        parserutils.LmpTraj.add(parser, positional=positional)
        options = parser.parse_args(args)
        assert positional == hasattr(options, 'traj')
        assert expected == [
            options.task, options.data_file, options.last_pct, options.slice
        ]


class TestWorkflow:

    @pytest.mark.parametrize('ekey', ['TQDM_DISABLE'])
    @pytest.mark.parametrize('evalue', [None, '1'])
    @pytest.mark.parametrize(
        'args,expected',
        [([], [1, ['task', 'aggregator'], None, False, None]),
         ([
             '-state_num', '3', '-jtype', 'task', '-prj_path', os.curdir,
             '-clean', '-screen', 'progress', 'job'
         ], [3, ['task'], os.curdir, True, ['progress', 'job']])])
    def testAddWorkflow(self, args, expected, env):
        parser = parserutils.Workflow()
        options = parser.parse_args(args)
        assert expected == [
            options.state_num, options.jtype, options.prj_path, options.clean,
            options.screen
        ]
