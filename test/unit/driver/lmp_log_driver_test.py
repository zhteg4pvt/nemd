import os

import lmp_log_driver as driver
import pandas as pd
import pytest

from nemd import envutils

LOG = envutils.test_data('ar', 'gas', 'lammps.log')
DATA_FILE = envutils.test_data('ar', 'gas', 'ar100.data')


class TestLmpLog:

    @pytest.fixture
    def log(self, argv):
        options = driver.validate_options(argv)
        return driver.LmpLog(options)

    @pytest.mark.parametrize("argv,num",
                             [([LOG], None),
                              ([LOG, '-data_file', DATA_FILE], 100)])
    def testSetStruct(self, log, num):
        log.setStruct()
        val = log.rdf if num is None else log.rdf.atoms.shape[0]
        assert num == val

    @pytest.mark.parametrize("argv,num", [([LOG], None)])
    def testSetThermo(self, log, num):
        log.setThermo()
        assert 264 == log.lmp_log.thermo.shape[0]

    @pytest.mark.parametrize("argv,num",
                             [([LOG, '-task', 'temp', 'e_pair'], None)])
    def testSetTasks(self, log, num):
        log.setThermo()
        log.setTasks()
        assert 2 == len(log.tasks)

    @pytest.mark.parametrize(
        "argv,idx",
        [([LOG, '-task', 'temp', 'e_pair'], '211'),
         ([LOG, '-task', 'temp', 'e_pair', '-last_pct', '0.01'], '261')])
    def testAnalyze(self, log, idx, tmp_dir):
        log.setThermo()
        log.setTasks()
        log.analyze()
        assert os.path.exists('lmp_log_temp.png')
        assert idx in pd.read_csv('lmp_log_e_pair.csv').columns[0]
