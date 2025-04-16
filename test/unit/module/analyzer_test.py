import copy
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from nemd import analyzer
from nemd import envutils
from nemd import lmpfull
from nemd import lmplog
from nemd import parserutils
from nemd import traj

TEST0027 = envutils.test_data('itest', '0027_test')
TEST0045 = envutils.test_data('itest', '0045_test')
TEST0037 = envutils.test_data('itest', '0037_test')
AR_DIR = os.path.join(TEST0045, 'workspace',
                      '6fd1b87409fbb60c6612569e187f59fc')
AR_TRJ = os.path.join(AR_DIR, 'amorp_bldr.custom.gz')
AR_DAT = os.path.join(AR_DIR, 'amorp_bldr.data')
AR_RDR = lmpfull.Reader(AR_DAT)
AR_LOG = os.path.join(AR_DIR, 'lammps_lmp.log')
HEX = envutils.test_data('hexane_liquid')
HEX_TRJ = os.path.join(HEX, 'dump.custom')
HEX_DAT = os.path.join(HEX, 'polymer_builder.data')
HEX_RDR = lmpfull.Reader(HEX_DAT)
MODIFIED = copy.deepcopy(HEX_RDR)
MODIFIED.pair_coeffs.dist = 18
FOUR_TRJ = os.path.join(HEX, 'four_frames.custom')


class TestBase:

    EMPTY = pd.DataFrame()
    DATA = pd.DataFrame({'density': [1, 0, 2]})
    DATA.index = pd.Index([5, 2, 6], name='time (ps)')
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
        'data,expected', [(EMPTY, False),
                          (DATA, 'The minimum density of 0 found at 2 ps.')])
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
    @pytest.mark.parametrize('dirname,expected', [(TEST0027, (0, 0, 0, None)),
                                                  (TEST0037, (3, 1, 1, None)),
                                                  (TEST0045, (7, 2, 3, None))])
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
    @pytest.mark.parametrize(
        'dirname,expected',
        [(TEST0045, 'Density: 0.00164 ± 0 g/cm^3 ∈ [2.0000, 5.0000] ps')])
    def testFit(self, density, expected):
        density.read()
        density.fit()
        density.logger.log.assert_called_with(expected)

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
                                                   (AR_TRJ, None, (1, 10)),
                                                   (AR_TRJ, [0, 1], (1, 2))])
    def testInit(self, trj, gids, expected):
        options = trj and parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'xyz'])
        if trj is not None:
            trj = traj.Traj(trj, options=options)
        job = analyzer.Density(trj=trj, gids=gids)
        assert expected == (job.sidx, job.gids and len(job.gids))

    @pytest.mark.parametrize('trj,rdr,expected', [(AR_TRJ, AR_RDR, 0.00164)])
    def testSet(self, trj, rdr, expected):
        job = analyzer.Density(trj=traj.Traj(trj), rdr=rdr)
        job.set()
        np.testing.assert_almost_equal(job.data.max(), [expected])


class TestXYZ:

    @pytest.mark.parametrize('trj', [HEX_TRJ])
    @pytest.mark.parametrize('rdr,center,wrapped,broken_bonds,expected',
                             [(HEX_RDR, False, True, False,
                               (-2.2587, 50.6048)),
                              (None, False, True, False, (-37.6929, 67.0373)),
                              (None, True, True, True, (0.0055, 47.839))])
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
        job.logger.log.assert_called_with(
            'XYZ coordinates are written into xyz_xyz.xyz')


class TestView:

    TRJ = envutils.test_data('water', 'three.custom')
    RDR = lmpfull.Reader(envutils.test_data('water', 'polymer_builder.data'))

    @pytest.mark.parametrize('trj', [TRJ])
    @pytest.mark.parametrize('rdr', [(None), (RDR)])
    def testRun(self, trj, rdr, tmp_dir):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-JOBNAME', 'view', '-task', 'view'])
        job = analyzer.View(trj=traj.Traj(trj),
                            rdr=rdr,
                            options=options,
                            logger=mock.Mock())
        job.run()
        assert os.path.isfile('view_view.html')
        job.logger.log.assert_called_with(
            'Trajectory visualization are written into view_view.html')


class TestClash:

    @pytest.fixture
    def clash(self, trj, gids, rdr):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'clash'])
        return analyzer.Clash(trj=traj.Traj(trj, options=options),
                              gids=gids,
                              rdr=rdr,
                              logger=mock.Mock())

    @pytest.mark.parametrize(
        'trj,gids,cut,srch,rdr,expected',
        [(None, None, None, None, None, (1.4, None, None)),
         (None, None, None, None, HEX_RDR, (1.9724464, None, None)),
         (HEX_TRJ, [3, 1, 6, 2], 1, None, None, (1, 3, 3)),
         (HEX_TRJ, [3, 1, 6, 2], 20, None, None, (20, 4, None)),
         (HEX_TRJ, [3, 1, 6, 2], 20, True, None, (20, 3, 3))])
    def testInit(self, trj, gids, cut, srch, rdr, expected):
        options = trj and parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'clash'])
        trj = trj and traj.Traj(trj, options=options)
        clash = analyzer.Clash(trj=trj,
                               gids=gids,
                               cut=cut,
                               srch=srch,
                               rdr=rdr,
                               logger=mock.Mock())
        np.testing.assert_almost_equal(clash.cut, expected[0])
        assert expected[1] == (None if clash.grp is None else len(clash.grp))
        assert expected[2] == (None if clash.grps is None else len(clash.grps))

    @pytest.mark.parametrize('trj,rdr', [(HEX_TRJ, MODIFIED)])
    @pytest.mark.parametrize('gids,expected', [([1], None), ([5, 0, 1], 2),
                                               ([5, 0], 1)])
    def testSet(self, clash, expected):
        clash.set()
        if expected is None:
            clash.logger.log.assert_called_with(
                'WARNING: Clash requires least two atoms selected.')
            return
        assert not clash.logger.log.called
        assert expected == clash.data.max().max()


class TestRDF:

    @pytest.fixture
    def rdf(self, trj, gids, rdr, jobs):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'rdf', '-NAME', 'lmp_traj'])
        return analyzer.RDF(trj=traj.Traj(trj, options=options),
                            options=options,
                            gids=gids,
                            rdr=rdr,
                            jobs=jobs,
                            logger=mock.Mock())

    @pytest.mark.parametrize('trj,rdr,jobs', [(HEX_TRJ, HEX_RDR, None)])
    @pytest.mark.parametrize('gids,expected', [([1], None),
                                               ([5, 0, 1], 41913.204277),
                                               ([5, 0], 5559.9628738)])
    def testSet(self, rdf, expected):
        rdf.set()
        if expected is None:
            rdf.logger.log.assert_called_with(
                'WARNING: RDF requires least two atoms selected.')
            return
        np.testing.assert_almost_equal(rdf.data.max().max(), expected)

    @pytest.mark.parametrize('trj,rdr,gids,dirname,msg,expected',
                             [(HEX_TRJ, HEX_RDR, [5, 0, 1], None,
                               'RDF peak 4.191e+04 ± nan found at 1.52 Å',
                               (41913.204277, np.nan)),
                              (HEX_TRJ, None, None, TEST0045,
                               'RDF peak 237.1 ± 237.1 found at 4.12 Å',
                               (237.05, 237.05))])
    def testFit(self, rdf, msg, expected):
        rdf.read()
        rdf.set()
        rdf.fit()
        rdf.logger.log.assert_called_with(msg)
        np.testing.assert_almost_equal(rdf.result, expected)

    def testLabel(self):
        assert 'g (r)' == analyzer.RDF().label


class TestMSD:

    @pytest.fixture
    def msd(self, trj, gids, rdr):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'msd'])
        return analyzer.MSD(trj=traj.Traj(trj, options=options),
                            options=options,
                            gids=gids,
                            rdr=rdr,
                            logger=mock.Mock())

    @pytest.mark.parametrize(
        'trj,rdr,gids,spct,epct,expected,name',
        [(FOUR_TRJ, HEX_RDR, None, 0.1, 0.2, 0.4820988, 'Tau (ps) (0 3)'),
         (FOUR_TRJ, None, None, 0.1, 0.2, 0.4803063, 'Tau (ps) (0 3)'),
         (FOUR_TRJ, HEX_RDR, [0, 1], 0.1, 0.2, 0.9921599, 'Tau (ps) (0 3)'),
         (FOUR_TRJ, HEX_RDR, [0, 1], 0.4, 0.2, 0.9921599, 'Tau (ps) (1 3)'),
         (AR_TRJ, None, [], 0.1, 0.2, np.nan, None),
         (HEX_TRJ, None, None, 0.1, 0.2, np.nan, None)])
    def testSet(self, msd, spct, epct, expected, name):
        msd.set(spct=spct, epct=epct)
        np.testing.assert_almost_equal(msd.data.max().max(), expected)
        assert name == msd.data.index.name

    @pytest.mark.parametrize('trj,rdr,gids,spct,epct,expected,msg', [
        (FOUR_TRJ, HEX_RDR, None, 0.1, 0.2, [4.01749017e-06, 8.75353197e-07],
         'Diffusion Coefficient 4.017e-06 ± 8.754e-07 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.955)'
         ),
        (FOUR_TRJ, None, None, 0.1, 0.2, [4.00255214e-06, 8.72030195e-07],
         'Diffusion Coefficient 4.003e-06 ± 8.72e-07 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.955)'
         ),
        (FOUR_TRJ, HEX_RDR, [0, 1], 0.1, 0.2, [8.2679990e-06, 1.5476629e-06],
         'Diffusion Coefficient 8.268e-06 ± 1.548e-06 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.966)'
         ),
        (FOUR_TRJ, HEX_RDR, [0, 1], 0.4, 0.2, [1.09486298e-05, 0.00000e+00],
         'Diffusion Coefficient 1.095e-05 ± 0 cm^2/s calculated by fitting MSD ∈ [1 2] ps. (R-squared: 1)'
         )
    ])
    def testFit(self, msd, spct, epct, expected, msg):
        msd.set(spct=spct, epct=epct)
        msd.fit()
        np.testing.assert_almost_equal(msd.result.values, expected)
        msd.logger.log.assert_called_with(msg)

    def testLabel(self):
        assert 'MSD (Å^2)' == analyzer.MSD().label


class TestTotEng:

    @pytest.fixture
    def tot_eng(self, logfile):
        if logfile is None:
            return analyzer.TotEng()
        options = parserutils.LmpLog().parse_args([AR_LOG])
        lmp_log = lmplog.Log(logfile, options=options)
        return analyzer.TotEng(thermo=lmp_log.thermo)

    @pytest.mark.parametrize('logfile,expected', [(None, 0), (AR_LOG, 7)])
    def testInit(self, tot_eng, expected):
        assert expected == tot_eng.sidx

    @pytest.mark.parametrize('logfile,expected', [(AR_LOG, 8.0481871)])
    def testSet(self, tot_eng, expected):
        tot_eng.set()
        np.testing.assert_almost_equal(tot_eng.data.max().max(), expected)

    def testAnalyzers(self):
        assert 6 == len(analyzer.TotEng.Analyzers)
