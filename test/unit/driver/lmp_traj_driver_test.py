import os

import lmp_traj_driver as driver
import pytest

from nemd import envutils
from nemd import frame
from nemd import parserutils


class TestTraj:

    AR_IN = envutils.test_data('ar', 'ar100.in')
    AR_DATA = envutils.test_data('ar', 'ar100.data')
    AR_TRJ = envutils.test_data('ar', 'ar100.custom')
    ARGS = [AR_TRJ, '-data_file', AR_DATA, '-JOBNAME', 'name']

    @pytest.fixture
    def trj(self, args, logger):
        options = parserutils.LmpTraj().parse_args(args)
        return driver.Traj(options, logger=logger)

    @pytest.mark.parametrize("args,expected",
                             [([AR_TRJ, '-task', 'xyz'], None),
                              (ARGS, (100, 7))])
    def testSetStruct(self, trj, expected):
        trj.setStruct()
        assert expected == (trj.rdr.atoms.shape if expected else trj.rdr)

    @pytest.mark.parametrize("args,expected", [([*ARGS, '-sel', 'O'], 0),
                                               ([*ARGS, '-sel', 'Ar'], 100),
                                               (ARGS, 100)])
    def testSetAtoms(self, trj, expected):
        trj.setStruct()
        trj.setAtoms()
        assert expected == len(trj.gids)

    @pytest.mark.parametrize(
        "args,expected",
        [([*ARGS, '-task', 'msd'], (236, 189, True)),
         ([*ARGS, '-task', 'density'], (236, 189, False)),
         ([*ARGS, '-task', 'rdf', '-slice', '10', '20', '2'], (5, 4, False))])
    def testSetFrames(self, trj, expected):
        trj.setFrames()
        assert expected[:2] == (len(trj.trj), trj.trj.time.sidx)
        assert expected[2] != isinstance(trj.trj[expected[1] - 2], frame.Frame)

    @pytest.mark.parametrize("args,expected", [(ARGS, 'name_density.csv')])
    def testAnalyze(self, trj, expected, tmp_dir):
        trj.setStruct()
        trj.setAtoms()
        trj.setFrames()
        trj.analyze()
        assert os.path.exists(expected)
