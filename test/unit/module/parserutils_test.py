import argparse
import os
from unittest import mock

import pytest

from nemd import envutils
from nemd import parserutils

AR_DIR = envutils.test_data('ar')
IN_FILE = os.path.join(AR_DIR, 'ar100.in')
MISS_DATA_IN = os.path.join(AR_DIR, 'single.in')
TRAJ_FILE = os.path.join(AR_DIR, 'ar100.custom.gz')


@pytest.mark.parametrize('raise_type', [argparse.ArgumentTypeError])
class TestType:

    @pytest.mark.parametrize('file,is_raise', [('not_existing', True),
                                               (
                                                   TRAJ_FILE,
                                                   False,
                                               )])
    def testFile(self, file, check_raise):
        with check_raise():
            parserutils.type_file(file)

    @pytest.mark.parametrize('file,is_raise', [(AR_DIR, False),
                                               ('not_existing', True),
                                               (TRAJ_FILE, True)])
    def testDir(self, file, check_raise):
        with check_raise():
            parserutils.type_dir(file)

    @pytest.mark.parametrize('arg,val,is_raise', [('y', True, False),
                                                  ('n', False, False),
                                                  ('wa', None, True)])
    def testBool(self, arg, val, check_raise):
        with check_raise():
            assert val == parserutils.type_bool(arg)

    @pytest.mark.parametrize('arg,val,is_raise', [('123', 123, False),
                                                  ('5.6', 5.6, False),
                                                  ('wa', None, True),
                                                  ('None', None, True)])
    def testFloat(self, arg, val, check_raise):
        with check_raise():
            assert val == parserutils.type_float(arg)

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1.12', True),
                                              ('-1', False)])
    def testInt(self, arg, check_raise):
        with check_raise():
            parserutils.type_int(arg)

    @pytest.mark.parametrize('bottom,top', [(100, 200)])
    @pytest.mark.parametrize('value,included_bottom,include_top,is_raise',
                             [(123, True, True, False),
                              (100, True, True, False),
                              (100, False, True, True),
                              (200, False, True, False),
                              (200, False, False, True)])
    def testRanged(self, value, bottom, top, included_bottom, include_top,
                   check_raise):
        with check_raise():
            parserutils.type_ranged(value,
                                    bottom=bottom,
                                    top=top,
                                    included_bottom=included_bottom,
                                    include_top=include_top)

    @pytest.mark.parametrize('bottom,top', [(1.12, 3.45)])
    @pytest.mark.parametrize('arg,included_bottom,include_top,is_raise',
                             [('2', True, True, False),
                              ('1.12', True, True, False),
                              ('1.12', False, True, True),
                              ('3.45', False, True, False),
                              ('3.45', False, False, True)])
    def testRangedFloat(self, arg, bottom, top, included_bottom, include_top,
                        check_raise):
        with check_raise():
            parserutils.type_ranged_float(arg,
                                          bottom=bottom,
                                          top=top,
                                          included_bottom=included_bottom,
                                          include_top=include_top)

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1.12', False),
                                              ('-1.12', True)])
    def testNonnegativeFloat(self, arg, check_raise):
        with check_raise():
            parserutils.type_nonnegative_float(arg)

    @pytest.mark.parametrize('arg,is_raise', [('0', True), ('1.12', False),
                                              ('-1.12', True)])
    def testPositiveFloat(self, arg, check_raise):
        with check_raise():
            parserutils.type_positive_float(arg)

    @pytest.mark.parametrize('arg,is_raise', [('1', False), ('0', True),
                                              ('-1', True)])
    def testPositiveInt(self, arg, check_raise):
        with check_raise():
            parserutils.type_positive_int(arg)

    @pytest.mark.parametrize('arg,is_raise', [('1', False), ('0', False),
                                              ('-1', True)])
    def testNonnegativeInt(self, arg, check_raise):
        with check_raise():
            parserutils.type_nonnegative_int(arg)

    @pytest.mark.parametrize('is_raise,ekey', [(False, 'TQDM_DISABLE')])
    @pytest.mark.parametrize('evalue', ['', '1'])
    @pytest.mark.parametrize('arg', ['wa', 'progress'])
    def testScreen(self, arg, evalue, env, check_raise):
        parserutils.type_screen(arg)
        tqdm = arg == 'progress' or not evalue
        assert bool(tqdm) == (not os.environ.get('TQDM_DISABLE'))

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1234', False),
                                              ('-1', True), (2**31, True)])
    def testRandomSeed(self, arg, check_raise):
        with check_raise():
            parserutils.type_random_seed(arg)

    @pytest.mark.parametrize('arg,is_raise', [('C', False),
                                              ('not_valid', True)])
    def testTypeSmiles(self, arg, check_raise):
        with check_raise():
            parserutils.type_smiles(arg)

    @pytest.mark.parametrize('arg,allow_reg,is_raise', [('C', True, False),
                                                        ('C', False, True),
                                                        ('*C*', False, False),
                                                        ('*C*.C', False, True),
                                                        ('C*', True, True)])
    def testCruSmiles(self, arg, allow_reg, check_raise):
        with check_raise():
            parserutils.type_cru_smiles(arg, allow_reg=allow_reg)

    @pytest.mark.parametrize('arg,is_raise', [('0.2', False), ('0', True),
                                              ('0.99', False), ('1', True)])
    def testLastPctType(self, arg, check_raise):
        with check_raise():
            parserutils.LastPct.type(arg)


class TestLastPct:

    @pytest.mark.parametrize('data', [[0, 1, 2, 3, 4]])
    @pytest.mark.parametrize('arg,buffer,sidx', [('0.7', 0, 2), ('0.6', 0, 2),
                                                 ('0.6', 1, 1)])
    def testLastPctGetSidx(self, data, arg, buffer, sidx):
        ptc = parserutils.LastPct.type(arg)
        assert sidx == ptc.getSidx(data, buffer=buffer)


class TestAction:

    class ArgumentParser(argparse.ArgumentParser):

        def check(self, args, err=False, expected=None):
            values = self.parse_args(args=['-flag', *args]).flag
            assert err == self.error.called
            if err:
                return
            assert (tuple(args) if expected is None else expected) == values

    @pytest.fixture
    def parser(self, action, dtype):
        parser = self.ArgumentParser()
        parser.add_argument('-flag', nargs='+', type=dtype, action=action)
        parser.error = mock.Mock()
        return parser

    @mock.patch('nemd.parserutils.Action.doTyping')
    @pytest.mark.parametrize('action,dtype', [(parserutils.Action, str)])
    def testDoTyping(self, mocked, parser):
        parser.parse_args(['-flag', '1', '2'])
        assert mocked.called

    @pytest.mark.parametrize('action,dtype',
                             [(parserutils.ForceFieldAction, str)])
    @pytest.mark.parametrize('args,expected,err',
                             [(['SW'], None, False),
                              (['OPLSUA'], ('OPLSUA', 'TIP3P'), False),
                              (['OPLSUA', 'SPCE'], None, False),
                              (['OPLSUA', 'NOT_VALID'], None, True)])
    def testForceField(self, args, parser, expected, err):
        parser.check(args, err, expected)

    @pytest.mark.parametrize(
        'action,dtype',
        [(parserutils.SliceAction, parserutils.type_positive_int)])
    @pytest.mark.parametrize('args,expected,err',
                             [(['9'], (0, 9, 1), False),
                              (['9', '1'], None, True),
                              (['1', '9', '3'], (1, 9, 3), False),
                              (['1', '9', '0'], None, True)])
    def testSlice(self, args, parser, expected, err):
        parser.check(args, err, expected)

    @pytest.mark.parametrize('action,dtype', [(parserutils.StructAction, str)])
    @pytest.mark.parametrize('args,expected,err',
                             [(['CCC', '36'], ('CCC', 36), False),
                              (['CC'], ('CC', None), False),
                              (['CCC', 'non_float'], None, True),
                              (['non_smiles'], None, True)])
    def testStruct(self, args, parser, expected, err):
        parser.check(args, err, expected)

    @pytest.mark.parametrize('action,dtype', [(parserutils.ThreeAction, str)])
    @pytest.mark.parametrize('args,expected,err',
                             [(['1', '2', '3', '4'], ('1', '2', '3'), False),
                              (['1'], ('1', '1', '1'), False),
                              (['1', '2'], None, True), ([], None, True)])
    def testThree(self, args, parser, expected, err):
        parser.check(args, err, expected)


class TestValidator:

    @pytest.fixture
    def parser(self, valid, flags, kwargss):
        if flags is None:
            flags = []
        if kwargss is None:
            kwargss = [{}]
        if isinstance(kwargss, dict):
            kwargss = [kwargss]
        if len(kwargss) == 1:
            kwargss = kwargss * len(flags)
        parser = parserutils.Driver(valids=set([valid]))
        parser.error = mock.Mock()
        for flag, kwargs in zip(flags, kwargss):
            if kwargs is None:
                kwargs = {}
            parser.add_argument(flag, **kwargs)
        return parser

    @pytest.fixture
    def args(self, flags, values):
        fvals = [[x, y] for x, y in zip(flags, values) if y is not None]
        fvals = [[x, *y] if isinstance(y, list) else [x, y] for x, y in fvals]
        return [y for x in fvals for y in x]

    @pytest.mark.parametrize('valid', [parserutils.MolValid])
    @pytest.mark.parametrize('flags', [('-cru', '-cru_num', '-mol_num')])
    @pytest.mark.parametrize('kwargss', [{'nargs': '+'}])
    @pytest.mark.parametrize('values,err',
                             [((['C'], None, None), False),
                              ((['C', 'O'], None, None), False),
                              ((['C', 'O'], ['1', '2'], None), False),
                              ((['C', 'O'], ['1', '2'], ['4', '5']), False),
                              ((['C', 'O'], ['1'], ['4', '5']), True),
                              ((['C', 'O'], ['1', '2'], ['5']), True)])
    def testMol(self, parser, err, args):
        parser.parse_args(args)
        assert err == parser.error.called

    @pytest.mark.parametrize('valid', [parserutils.LmpValid])
    @pytest.mark.parametrize('flags', [('-inscript', '-data_file')])
    @pytest.mark.parametrize('kwargss', [None])
    @pytest.mark.parametrize(
        'values,err', [((IN_FILE, None), False), ((MISS_DATA_IN, None), True),
                       ((MISS_DATA_IN, 'my.data'), False),
                       ((os.path.join(AR_DIR, 'mydata.in'), None), False),
                       ((os.path.join(AR_DIR, 'empty.in'), None), False)])
    def testLmp(self, values, parser, err, args, tmp_dir):
        with open('my.data', 'w') as fh:
            fh.write('This is a data file')
        parser.parse_args(args)
        assert err == parser.error.called

    @pytest.mark.parametrize('valid', [parserutils.TrajValid])
    @pytest.mark.parametrize('flags', [(['-task', '-data_file'])])
    @pytest.mark.parametrize('kwargss', [([{'nargs': '+'}, None])])
    @pytest.mark.parametrize('values,err', [((['density'], None), True),
                                            ((['xyz'], None), False),
                                            ((['density'], 'my.data'), False)])
    def testTraj(self, values, parser, err, args):
        parser.parse_args(args)
        assert err == parser.error.called


class TestDriver:

    @pytest.fixture
    def parser(self):
        return parserutils.Driver(valids=[parserutils.Valid])

    def testSetUp(self):
        with mock.patch('nemd.parserutils.Driver.setUp') as mocked:
            parserutils.Driver()
        assert mocked.called

    def testAdd(self):
        with mock.patch('nemd.parserutils.Driver.add') as mocked:
            parser = parserutils.Driver()
        mocked.assert_called_with(parser, append=False)

    @pytest.mark.parametrize('ekey', ['DEBUG'])
    @pytest.mark.parametrize('evalue', [None, '1'])
    @pytest.mark.parametrize('DEBUG,expected',
                             [(None, None), ('NO_VALUE', True), ('True', True),
                              ('False', False), ('on', True), ('off', False)])
    def testAddJob(self, DEBUG, expected, evalue, env):
        if expected is None:
            expected = bool(evalue)
        with mock.patch('nemd.parserutils.DEBUG', bool(evalue)):
            parser = parserutils.Driver(valids=[parserutils.Valid])
        args = [
            '-JOBNAME', 'hi', '-cpu', '3', '1', '-PYTHON', '-1', '-NAME', 'wa'
        ]
        if DEBUG is not None:
            args += ['-DEBUG'] if DEBUG == 'NO_VALUE' else ['-DEBUG', DEBUG]
        options = parser.parse_args(args)
        assert 'hi' == options.JOBNAME
        assert [3, 1] == options.cpu
        assert '-1' == options.PYTHON
        assert 'wa' == options.NAME
        assert expected == options.DEBUG

    def testAddSeed(self, parser):
        parser.addSeed(parser)
        options = parser.parse_args([])
        assert isinstance(options.seed, int)
        options = parser.parse_args(['-seed', '12'])
        assert 12 == options.seed

    def testParseArgs(self):
        parser = parserutils.Driver(valids=[parserutils.Valid])
        with mock.patch('nemd.parserutils.Valid.run') as mocked:
            options = parser.parse_args(['-JOBNAME', 'hi'])
        assert mocked.called
        assert 'hi' == options.JOBNAME
