import os
import types
from unittest import mock

import lammps_driver as driver
import pytest

from nemd import envutils
from nemd import task

AR_IN = envutils.test_data('ar', 'single', 'single.in')
MISSING_DAT_IN = envutils.test_data('ar', 'single', 'missing_dat.in')
AR_DATA = envutils.test_data('ar', 'single', 'single.data')
SI_IN = envutils.test_data('si', 'crystal_builder.in')
SI_DATA = envutils.test_data('si', 'crystal_builder.data')


class TestValidator:

    @pytest.fixture
    def validator(self, argv):
        options = task.LammpsJob.get_parser().parse_args(argv)
        return driver.Validator(options)

    @pytest.mark.parametrize(
        "argv,is_raise,raise_type",
        [([AR_IN], False, None),
         ([AR_DATA, '-data_file', AR_DATA], False, None),
         ([MISSING_DAT_IN, '-data_file', AR_DATA], False, None),
         ([MISSING_DAT_IN], True, FileNotFoundError)])
    def testDataFile(self, validator, check_raise):
        with check_raise():
            validator.dataFile()


class TestLammps:

    @pytest.fixture
    def lmp(self, argv):
        options = driver.validate_options(argv)
        return driver.Lammps(options)

    @pytest.mark.parametrize(
        "argv,expected", [([AR_IN], ['-log', 'lammps.log', '-screen', 'none']),
                          ([AR_IN, '-screen', 'my.out', '-log', 'my.log'
                            ], ['-log', 'my.log', '-screen', 'my.out'])])
    def testSetArgs(self, lmp, expected):
        lmp.setArgs()
        assert expected == lmp.args[-4:]

    @pytest.mark.parametrize("argv", [([AR_IN])])
    def testReadContents(self, lmp):
        lmp.setArgs()
        lmp.readContents()
        assert lmp.contents

    @pytest.mark.parametrize("argv", [([SI_IN])])
    def testSetPairCoeff(self, lmp):
        lmp.setArgs()
        lmp.readContents()
        lmp.addPath()
        lmp.setPairCoeff()
        assert envutils.test_data('si', 'Si.sw') in lmp.contents

    @pytest.mark.parametrize("argv", [([SI_IN])])
    def testWriteIn(self, lmp, tmp_dir):
        lmp.readContents()
        lmp.writeIn()
        assert os.path.exists(lmp.options.inscript)

    @mock.patch('subprocess.run')
    @pytest.mark.parametrize("argv", [([SI_IN])])
    def testSetGpu(self, run_mock, lmp):
        run_mock.return_value = types.SimpleNamespace(stdout='')
        lmp.setGpu()
        assert 'gpu' not in lmp.args
        run_mock.return_value = types.SimpleNamespace(stdout='GPU')
        lmp.setGpu()
        assert 'gpu' in lmp.args

    @mock.patch('lammps_driver.logger')
    @pytest.mark.parametrize("argv", [([SI_IN])])
    def testRunLammps(self, logger_mock, lmp, tmp_dir):
        lmp.setArgs()
        lmp.readContents()
        lmp.addPath()
        lmp.setPairCoeff()
        lmp.writeIn()
        lmp.setGpu()
        lmp.runLammps()
        assert os.path.exists('lammps.log')