import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from nemd import analyzer
from nemd import envutils
from nemd import lmpfull
from nemd import parserutils
from nemd import traj

TEST0027 = envutils.test_data('itest', '0027_test')
TEST0045 = envutils.test_data('itest', '0045_test')
TEST0037 = envutils.test_data('itest', '0037_test')
AR_DIR = os.path.join(TEST0045, 'workspace',
                      '3e1866ded1c2eea09dfe0a34482ecca2')
AR_TRJ = os.path.join(AR_DIR, 'amorp_bldr.custom.gz')
AR_DAT = os.path.join(AR_DIR, 'amorp_bldr.data')
AR_RDR = lmpfull.Reader(AR_DAT)
HEX = envutils.test_data('hexane_liquid')
HEXANE_RDR = lmpfull.Reader(os.path.join(HEX, 'polymer_builder.data'))
HEXANE_FRM = os.path.join(HEX, 'dump.custom')


class TestBase:

    EMPTY = pd.DataFrame()
    DATA = pd.DataFrame({'density': [1, 0, 2]})
    DATA.index = pd.Index([5, 2, 6], name='time')
    TWO_COLS = DATA.copy()
    TWO_COLS['std'] = 1

    @pytest.fixture
    def base(self, args, data):
        options = parserutils.Driver().parse_args(args) if args else None
        base = analyzer.Base(options=options, logger=mock.Mock())
        base.data = data
        return base

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize(
        'data,expected,exist',
        [(EMPTY, 'WARNING: Empty Result for base', False),
         (DATA, 'Base data written into name_base.csv', True)])
    def testSave(self, base, expected, exist, tmp_dir):
        base.save()
        base.logger.log.assert_called_with(expected)
        assert exist == os.path.exists('name_base.csv')

    def testFull(self):
        assert 'base' == analyzer.Base().full

    def testName(self):
        assert 'base' == analyzer.Base.name

    @pytest.mark.parametrize('data', [None])
    @pytest.mark.parametrize('args,expected',
                             [(['-JOBNAME', 'name'], 'name_base.csv'),
                              (['-JOBNAME', 'job'], 'job_base.csv')])
    def testOutfile(self, base, expected):
        assert expected == base.outfile

    @pytest.mark.parametrize('args', [None])
    @pytest.mark.parametrize(
        'data,expected',
        [(EMPTY, False),
         (DATA, 'The minimum density of 1 is found with the time being 2')])
    def testFit(self, base, expected):
        base.fit()
        assert bool(expected) == base.logger.log.called
        if expected:
            base.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize('data,line,marker,selected,expected',
                             [(EMPTY, None, None, None, (0, 0, False)),
                              (DATA, '-', '*', None, (1, 0, True)),
                              (TWO_COLS, '-', '*', None, (1, 1, True)),
                              (DATA, '--', '*', pd.DataFrame([[1], [2]]),
                               (2, 0, True))])
    def testPlot(self, base, line, marker, selected, expected, tmp_dir):
        base.plot(line=line, marker=marker, selected=selected)
        line_num = len(base.fig.axes[0].lines) if base.fig else 0
        col_num = len(base.fig.axes[0].collections) if base.fig else 0
        assert expected == (line_num, col_num, os.path.exists('name_base.png'))

    @pytest.mark.parametrize('name,expected',
                             [('r', ('r', None, None)),
                              ('r (g/m^3)', ('r', 'g/m^3', None)),
                              ('r (g/m^3) (num=3)', ('r', 'g/m^3', 'num=3'))])
    def testParse(self, name, expected):
        assert expected == analyzer.Base.parse(name)


class TestJob:

    PARM = pd.Series(['CCCC 5.0'], index=pd.Index(['substruct'], name=1))

    @pytest.fixture
    def job(self, args, parm, jobs):
        options = parserutils.Driver().parse_args(args)
        return analyzer.Job(options=options, parm=parm, jobs=jobs)

    @pytest.fixture
    def density(self, args, parm, jobs):
        options = parserutils.Driver().parse_args(args)
        Density = type('density', (analyzer.Job, ), {})
        return Density(options=options,
                       parm=parm,
                       jobs=jobs,
                       logger=mock.Mock())

    @pytest.mark.parametrize('args,dirname', [(['-JOBNAME', 'name'], None)])
    @pytest.mark.parametrize('parm,expected',
                             [(None, 'name_job.csv'),
                              (PARM, 'workspace/name_job_1.csv')])
    def testOutfile(self, job, expected):
        assert expected == job.outfile

    @pytest.mark.parametrize('args,parm', [(['-NAME', 'lmp_traj'], None)])
    @pytest.mark.parametrize('dirname,expected',
                             [(TEST0027, (0, 0, 0, None)),
                              (TEST0037, (3, 1, 1, None)),
                              (TEST0045, (145, 2, 116, None))])
    def testRead(self, density, expected):
        density.read()
        assert expected == (*density.data.shape, density.sidx, density.eidx)

    @pytest.mark.parametrize('args,parm,dirname', [([], None, None)])
    @pytest.mark.parametrize('name,expected',
                             [('r (Å)', ('r', 'Å', 0, None)),
                              ('Time (ps) (1)', ('Time', 'ps', 1, None)),
                              ('Tau (ps) (0 2)', ('Tau', 'ps', 0, 2))])
    def testParseIndex(self, job, name, expected):
        assert expected == job.parseIndex(name)

    @pytest.mark.parametrize('args,parm,dirname', [([], None, None)])
    def testLabel(self, job):
        assert 'Job (a.u.)' == job.label

    @pytest.mark.parametrize('args,parm', [(['-NAME', 'lmp_traj'], None)])
    @pytest.mark.parametrize('dirname,expected',
                             [(TEST0045, (145, 2, 116, None))])
    def testFit(self, density, expected):
        density.read()
        density.fit()

    @pytest.mark.parametrize('args,parm', [(['-NAME', 'lmp_traj'], None)])
    @pytest.mark.parametrize('dirname,expected',
                             [(TEST0045, (145, 2, 116, None))])
    def testPlot(self, density, expected, tmp_dir):
        density.read()
        density.plot()
        assert 2 == len(density.fig.axes[0].lines)

    @pytest.mark.parametrize(
        'name,unit,label,names,err,expected',
        [(None, None, None, None, False, 'job (a.u.)'),
         ('name', 'm', 'r (g/m^3) (num=3)', None, False, 'name (m) (num=3)'),
         ('r', None, None, ['r', 'St Dev of r'], False, 'r'),
         ('r', None, None, ['r', 'St Dev of r'], True, 'St Dev of r'),
         ('r', None, None, ['r', 'St Err of r'], True, 'St Err of r')])
    def testGetName(self, name, unit, label, names, err, expected):
        assert expected == analyzer.Job.getName(name=name,
                                                unit=unit,
                                                label=label,
                                                names=names,
                                                err=err)


class TestDensity:

    @pytest.mark.parametrize('trj,gids,expected', [(None, None, (0, None)),
                                                   (AR_TRJ, None, (29, 100)),
                                                   (AR_TRJ, [0, 1], (29, 2))])
    def testInit(self, trj, gids, expected):
        options = trj and parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'xyz'])
        if trj is not None:
            trj = traj.Traj(trj, options=options)
        job = analyzer.Density(trj=trj, gids=gids)
        assert expected == (job.sidx, job.gids and len(job.gids))

    @pytest.mark.parametrize('trj,rdr,expected', [(AR_TRJ, AR_RDR, 0.0016524)])
    def testSet(self, trj, rdr, expected):
        job = analyzer.Density(trj=traj.Traj(trj), rdr=rdr)
        job.set()
        np.testing.assert_almost_equal(job.data.max(), [expected])


class TestXYZ:

    @pytest.mark.parametrize('trj,rdr', [(HEXANE_FRM, HEXANE_RDR)])
    @pytest.mark.parametrize('center,wrapped,broken_bonds,expected',
                             [(False, True, False, (-2.2587, 50.6048)),
                              (True, True, True, (0.0055, 47.839))])
    def testRun(self, trj, rdr, center, wrapped, broken_bonds, expected,
                tmp_dir):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-JOBNAME', 'xyz', '-task', 'xyz'])
        job = analyzer.XYZ(trj=traj.Traj(trj),
                           rdr=rdr,
                           options=options,
                           logger=mock.Mock())
        job.run(center=center, wrapped=wrapped, broken_bonds=broken_bonds)
        xyzs = np.loadtxt('xyz_xyz.xyz', skiprows=2, usecols=(1, 2, 3))
        np.testing.assert_almost_equal([xyzs.min(), xyzs.max()], expected)
        assert job.logger.log.called
