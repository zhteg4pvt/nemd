import os
import types
from unittest import mock

import lmp_traj_driver as driver
import pytest

from nemd import envutils
from nemd import task
from nemd import traj

AR_IN = envutils.test_data('ar', 'gas', 'ar100.in')
AR_DATA = envutils.test_data('ar', 'gas', 'ar100.data')
AR_CUSTOM = envutils.test_data('ar', 'gas', 'ar100.custom')
AR_CUSTOM_GZ = envutils.test_data('ar', 'gas', 'ar100.custom.gz')


class TestValidator:

    @pytest.fixture
    def validator(self, argv):
        parser = task.TrajJob.get_parser()
        options = parser.parse_args(argv)
        return driver.Validator(options)

    @pytest.mark.parametrize(
        "argv,is_raise,raise_type",
        [([AR_CUSTOM_GZ, '-data_file', AR_DATA], False, None),
         ([AR_CUSTOM_GZ], True, ValueError),
         ([AR_CUSTOM_GZ, '-task', 'xyz'], False, ValueError)])
    def testTask(self, validator, check_raise):
        with check_raise():
            validator.task()


class TestTraj:

    @pytest.fixture
    def obj(self, argv):
        options = driver.validate_options(argv)
        return driver.Traj(options)

    @pytest.mark.parametrize("argv", [([AR_CUSTOM_GZ, '-data_file', AR_DATA])])
    def testSetStruct(self, obj):
        obj.setStruct()
        assert obj.rdf is not None

    @pytest.mark.parametrize(
        "argv,num",
        [([AR_CUSTOM_GZ, '-data_file', AR_DATA, '-sel', 'O'], 0),
         ([AR_CUSTOM_GZ, '-data_file', AR_DATA, '-sel', 'Ar'], 100),
         ([AR_CUSTOM_GZ, '-data_file', AR_DATA], 100)])
    def testSetAtoms(self, obj, num):
        obj.setStruct()
        obj.setAtoms()
        assert num == len(obj.gids)

    @pytest.mark.parametrize(
        "argv,idx,fake,num",
        [([AR_CUSTOM_GZ, '-data_file', AR_DATA, '-task', 'msd'
           ], 189, True, 236),
         ([AR_CUSTOM_GZ, '-data_file', AR_DATA, '-task', 'density'
           ], 189, False, 236),
         ([
             AR_CUSTOM_GZ, '-data_file', AR_DATA, '-task', 'rdf', '-slice',
             '10:20:2'
         ], 4, False, 5)])
    def testSetFrames(self, obj, idx, fake, num):
        obj.setFrames()
        assert num == len(obj.frms)
        assert idx == int(obj.time.name.split()[-1].strip('()'))
        is_traj = isinstance(obj.frms[idx - 2], traj.Frame)
        assert fake != is_traj

    @pytest.mark.parametrize("argv", [([AR_CUSTOM_GZ, '-data_file', AR_DATA])])
    def testAnalyze(self, obj, tmp_dir):
        obj.setStruct()
        obj.setAtoms()
        obj.setFrames()
        obj.analyze()
        assert os.path.exists('lmp_traj_density.csv')
