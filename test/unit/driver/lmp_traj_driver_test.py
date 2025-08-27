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
    NO_DAT = [AR_TRJ, '-JOBNAME', 'name']
    ARGS = NO_DAT + ['-data_file', AR_DATA]
    XYZ = [AR_TRJ, '-JOBNAME', 'name'] + ['-task', 'xyz']
    XTC = [envutils.test_data('ar', 'amorp_bldr.xtc')] + NO_DAT[1:]
    EMPTY = [envutils.test_data('ar', 'empty.custom')] + NO_DAT[1:]

    @pytest.fixture
    def trj(self, args, logger):
        options = parserutils.LmpTraj().parse_args(args)
        return driver.Traj(options, logger=logger)

    @pytest.mark.parametrize("args,expected", [(ARGS, 'name_density.csv')])
    def testRun(self, trj, expected, tmp_dir):
        trj.run()
        assert os.path.exists(expected)

    @pytest.mark.parametrize("args,expected",
                             [([AR_TRJ, '-task', 'xyz'], None),
                              (ARGS, (100, 7))])
    def testSetStruct(self, trj, expected):
        trj.setStruct()
        assert expected == (trj.rdr.atoms.shape if expected else trj.rdr)

    @pytest.mark.parametrize("args,expected", [([*ARGS, '-sel', 'O'], 0),
                                               ([*ARGS, '-sel', 'Ar'], 100),
                                               (ARGS, 100), (XYZ, None)])
    def testSetAtoms(self, trj, expected):
        trj.setStruct()
        trj.setAtoms()
        assert expected == (None if trj.gids is None else len(trj.gids))

    @pytest.mark.parametrize(
        "args,expected",
        [([*ARGS, '-task', 'msd'], (236, 189, False)),
         ([*XTC, '-task', 'msd'], (1, 0, True)),
         ([*EMPTY, '-task', 'msd'], SystemExit),
         ([*ARGS, '-task', 'density'], (236, 189, True)),
         ([*ARGS, '-task', 'rdf', '-slice', '10', '20', '2'], (5, 4, True))])
    def testSetFrames(self, trj, expected, raises):
        with raises:
            trj.setFrames()
            assert expected[:2] == (len(trj.trj), trj.trj.time.sidx)
            index = expected[1] - 2
            frm = trj.trj[index if index > 0 else 0]
            assert expected[2] == isinstance(frm, frame.Frame)

    @pytest.mark.parametrize("args,expected", [(ARGS, 'name_density.csv')])
    def testAnalyze(self, trj, expected, tmp_dir):
        trj.setStruct()
        trj.setAtoms()
        trj.setFrames()
        trj.analyze()
        assert os.path.exists(expected)
