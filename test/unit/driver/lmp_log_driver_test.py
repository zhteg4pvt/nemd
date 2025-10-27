import os
from unittest import mock

import conftest
import lmp_log_driver as driver
import pytest

from nemd import envutils
from nemd import jobutils
from nemd import parserutils


@conftest.require_src
class TestLmpLog:

    LOG = envutils.Src().test('ar', 'lammps.log')
    EMPTY = envutils.Src().test('ar', 'empty.log')
    DATA = envutils.Src().test('ar', 'ar100.data')
    NO_VOL = envutils.Src().test('0046_test', 'workspace',
                                 '67f59ab2eb89dab89c2f1b16e4fc1776',
                                 'lammps_lmp.log')

    @pytest.fixture
    def lmplog(self, args, logger):
        args = jobutils.Args(args).set('-JOBNAME', 'name')
        options = parserutils.LmpLog().parse_args(args)
        return driver.LmpLog(options=options, logger=logger)

    @pytest.mark.parametrize("args,expected",
                             [([LOG], 'name_toteng'),
                              ([LOG, '-task', 'e_pair'], 'name_e_pair')])
    def testRun(self, lmplog, expected, tmp_dir):
        lmplog.run()
        assert os.path.exists(f'{expected}.csv')
        assert os.path.exists(f'{expected}.svg')

    @pytest.mark.parametrize("args,expected",
                             [([LOG], None),
                              ([LOG, '-data_file', DATA], (100, 7))])
    def testSetStruct(self, lmplog, expected):
        lmplog.setStruct()
        assert expected == (lmplog.rdr.atoms.shape if expected else lmplog.rdr)

    @pytest.mark.parametrize("args,expected", [([LOG], 264),
                                               ([EMPTY], SystemExit)])
    def testSetThermo(self, lmplog, expected, raises):
        with raises:
            lmplog.setThermo()
            assert expected == lmplog.thermo.shape[0]

    @mock.patch('nemd.logutils.Base.error', side_effect=ValueError)
    @mock.patch('nemd.logutils.Base.warning')
    @pytest.mark.parametrize("args,expected",
                             [([LOG, '-task', 'temp', 'e_pair'], (2, False)),
                              ([LOG, '-task', 'all', 'e_pair'], (6, False)),
                              ([NO_VOL, '-task', 'e_pair', 'volume'],
                               (1, True)),
                              ([NO_VOL, '-task', 'volume'], ValueError)])
    def testSetTasks(self, warn_mocked, mocked, lmplog, expected, raises):
        lmplog.setThermo()
        with raises:
            lmplog.setTasks()
            assert expected == (len(lmplog.task), warn_mocked.called)

    @pytest.mark.parametrize(
        "args,expected",
        [([LOG, '-task', 'temp'], ['temp']),
         ([LOG, '-task', 'temp', 'e_pair'], ['temp', 'e_pair'])])
    def testAnalyze(self, lmplog, expected, tmp_dir):
        lmplog.setThermo()
        lmplog.setTasks()
        lmplog.analyze()
        for basename in expected:
            assert os.path.exists(f'{lmplog.options.JOBNAME}_{basename}.csv')
            assert os.path.exists(f'{lmplog.options.JOBNAME}_{basename}.svg')
