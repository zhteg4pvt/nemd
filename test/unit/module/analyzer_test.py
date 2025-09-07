import copy
import glob
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from nemd import analyzer
from nemd import envutils
from nemd import lmpfull
from nemd import lmplog
from nemd import numpyutils
from nemd import parserutils
from nemd import task
from nemd import traj

TEST0027 = envutils.test_data('0027_test')
TEST0037 = envutils.test_data('0037_test')
TEST0045 = envutils.test_data('0045_test')
TEST0046 = envutils.test_data('0046_test')
AR_DIR = os.path.join(TEST0045, 'workspace',
                      '6fd1b87409fbb60c6612569e187f59fc')
AR_TRJ = os.path.join(AR_DIR, 'amorp_bldr.xtc')
AR_DAT = os.path.join(AR_DIR, 'amorp_bldr.data')
AR_RDR = lmpfull.Reader(AR_DAT)
HEX = envutils.test_data('hexane_liquid')
HEX_TRJ = os.path.join(HEX, 'dump.custom')
HEX_DAT = os.path.join(HEX, 'polymer_builder.data')
HEX_RDR = lmpfull.Reader(HEX_DAT)


class TestBase:

    EMPTY = pd.DataFrame()
    DATA = pd.DataFrame({'density (g/cm^3)': [1, 0, 2]})
    DATA.index = pd.Index([5, 2, 6], name='time (ps)')
    TWO_COLS = DATA.copy()
    TWO_COLS['std'] = 1

    @pytest.fixture
    def raw(self):
        return analyzer.Base()

    @pytest.fixture
    def base(self, args, data):
        options = parserutils.Driver().parse_args(args) if args else None
        base = analyzer.Base(options=options, logger=mock.Mock())
        base.data = data
        return base

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize('data,expected', [(DATA, True), (TWO_COLS, True),
                                               (None, False)])
    def testRun(self, base, expected, tmp_dir):
        base.run()
        assert expected == os.path.exists(base.outfile)

    def testMerge(self, raw):
        raw.merge()
        assert raw.data is None

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize('data', [EMPTY, DATA])
    def testSave(self, base, tmp_dir):
        base.save()
        base.logger.log.assert_called_with(
            'Base data written into name_base.csv')
        assert os.path.exists('name_base.csv')

    def testFull(self, raw):
        assert 'base' == raw.full

    def testName(self):
        assert 'base' == analyzer.Base.name

    @pytest.mark.parametrize('data', [None])
    @pytest.mark.parametrize('args,expected',
                             [(['-JOBNAME', 'name'], 'name_base.csv')])
    def testOutfile(self, base, expected):
        assert expected == base.outfile

    @pytest.mark.parametrize('args', [None])
    @pytest.mark.parametrize(
        'data,expected',
        [(EMPTY, None),
         (DATA, 'The minimum density of 0 g/cm^3 is found at 2 ps.')])
    def testFit(self, base, called, expected):
        base.log = called
        base.fit()

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize('data,marker,selected,fitted,expected',
                             [(DATA, '*', None, None, (1, 0, True)),
                              (TWO_COLS, '*', None, np.array([[3, 7], [4, 8]]),
                               (2, 1, True)),
                              (DATA, '*', pd.DataFrame([[1], [2]]), None,
                               (2, 0, True))])
    def testPlot(self, base, marker, selected, fitted, expected, tmp_dir):
        base.fitted = fitted
        base.plot(marker=marker, selected=selected)
        line_num = len(base.fig.axes[0].lines) if base.fig else 0
        col_num = len(base.fig.axes[0].collections) if base.fig else 0
        assert expected == (line_num, col_num, os.path.exists('name_base.png'))

    @pytest.mark.parametrize('args', [(['-JOBNAME', 'name'])])
    @pytest.mark.parametrize('data,expected', [(DATA, False),
                                               (TWO_COLS, True)])
    def testWithErr(self, base, expected):
        assert expected == base.with_err

    @pytest.mark.parametrize('name,expected',
                             [('r', ('r', None, None)),
                              ('r (g/m^3)', ('r', 'g/m^3', None)),
                              ('r (g/m^3) (num=3)', ('r', 'g/m^3', 'num=3'))])
    def testParse(self, name, expected):
        assert expected == analyzer.Base.parse(name)


class TestJob:

    EMPTY = pd.Series()
    PARM = pd.Series(['CCCC 5.0'], index=pd.Index(['substruct'], name=1))

    @pytest.fixture
    def job(self, args, group, jobs):
        options = parserutils.Driver().parse_args(args)
        return analyzer.Job(options=options, parm=group)

    @pytest.fixture
    def density(self, args, parm, jobs, logger):
        Density = type('density', (analyzer.Job, ), {})
        options = parserutils.Driver().parse_args(args)
        if parm is not None:
            parm = task.Group(parm=parm, jobs=jobs)
        return Density(options=options, parm=parm, logger=logger)

    @pytest.mark.parametrize('args,dirname', [([], None)])
    @pytest.mark.parametrize(
        'parm,data,expected',
        [(None, None, 'WARNING: No data calculated for job'),
         (PARM, None, None), (None, pd.DataFrame([1]), None)])
    def testSet(self, job, data, called):
        job.log = called
        job.data = data
        job.set()

    @pytest.mark.parametrize('args,parm,dirname', [([], None, None)])
    def testCalculate(self, job):
        job.calculate()
        assert job.data is None

    @pytest.mark.parametrize('args', [(['-NAME', 'lmp_traj'])])
    @pytest.mark.parametrize('dirname,parm,expected',
                             [(TEST0037, PARM, (3, 1, 1, None)),
                              (TEST0045, PARM, (23, 2, 9, None)),
                              ('empty', PARM, None), (TEST0027, PARM, None),
                              (TEST0027, None, None)])
    def testMerge(self, density, expected):
        density.merge()
        assert expected == (None if density.data is None else
                            (*density.data.shape, density.sidx, density.eidx))

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

    @pytest.mark.parametrize('args,dirname', [(['-JOBNAME', 'name'], None)])
    @pytest.mark.parametrize('parm,expected',
                             [(None, 'name_job.csv'),
                              (EMPTY, 'workspace/name_job_None.csv'),
                              (PARM, 'workspace/name_job_1.csv')])
    def testOutfile(self, job, expected):
        assert expected == job.outfile

    @pytest.mark.parametrize('args,dirname,parm',
                             [(['-NAME', 'lmp_traj'], TEST0045, PARM)])
    @pytest.mark.parametrize(
        'drop,expected',
        [(None, 'Density: 0.001799 ± 4.5e-06 g/cm^3 ∈ [10.0000, 23.0000] ps'),
         ('St Dev of Density (g/cm^3) (num=2)',
          'Density: 0.001799 ± 2.255e-05 g/cm^3 ∈ [10.0000, 23.0000] ps')])
    def testFit(self, density, drop, called):
        density.merge()
        if drop:
            density.data.drop(columns=[drop], inplace=True)
        density.logger.log = called
        density.fit()

    @pytest.mark.parametrize(
        'name,unit,label,names,err,expected',
        [(None, None, None, None, False, 'job (a.u.)'),
         ('name', 'm', 'r (g/m^3) (num=3)', None, False, 'name (m) (num=3)'),
         ('r', None, None, ['r', 'St Dev of r'], False, 'r'),
         ('r', None, None, ['r', 'St Dev of r'], True, 'St Dev of r'),
         ('r', None, None, ['r', 'St Err of r'], True, 'St Err of r'),
         ('rho', None, None, ['r', 'St Err of r'], True, None)])
    def testGetName(self, name, unit, label, names, err, expected):
        assert expected == analyzer.Job.getName(name=name,
                                                unit=unit,
                                                label=label,
                                                names=names,
                                                err=err)

    @pytest.mark.parametrize('args,parm', [(['-NAME', 'lmp_traj'], PARM)])
    @pytest.mark.parametrize('dirname,expected', [(TEST0045, 2)])
    def testPlot(self, density, expected, tmp_dir):
        density.merge()
        density.plot()
        assert expected == len(density.fig.axes[0].lines)

    @pytest.mark.parametrize('args,parm', [(['-NAME', 'lmp_traj'], PARM)])
    @pytest.mark.parametrize('dirname,expected', [(TEST0045, (1, 3))])
    def testResult(self, density, expected, tmp_dir):
        density.run()
        assert expected == density.result.shape


class TestDensity:

    @pytest.mark.parametrize('trj,gids,expected', [(None, None, (0, None)),
                                                   (AR_TRJ, None, (5, 10)),
                                                   (AR_TRJ, [0, 1], (5, 2))])
    def testInit(self, trj, gids, expected):
        if trj is not None:
            args = [trj, '-last_pct', '0.8', '-task', 'xyz']
            options = parserutils.LmpTraj().parse_args(args)
            trj = traj.Traj(trj, options=options)
        job = analyzer.Density(trj=trj, gids=gids)
        assert expected == (job.sidx,
                            None if job.gids is None else len(job.gids))

    @pytest.mark.parametrize('trj,data_file,expected',
                             [(AR_TRJ, AR_DAT, 0.0018261)])
    def testCalculate(self, trj, data_file, expected):
        args = [trj, '-data_file', data_file]
        options = parserutils.LmpTraj().parse_args(args)
        job = analyzer.Density(trj=traj.Traj(trj, options=options),
                               rdr=lmpfull.Reader(data_file))
        job.calculate()
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
        job = analyzer.XYZ(trj=traj.Traj(trj, options=options),
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
    DAT = envutils.test_data('water', 'polymer_builder.data')

    @pytest.mark.parametrize('trj', [TRJ])
    @pytest.mark.parametrize('dat', [(None), (DAT)])
    def testRun(self, trj, dat, tmp_dir):
        args = [trj, '-JOBNAME', 'view', '-task', 'view']
        if dat:
            args += ['-data_file', dat]
        options = parserutils.LmpTraj().parse_args(args)
        job = analyzer.View(trj=traj.Traj(trj, options=options, start=0),
                            rdr=lmpfull.Reader(dat) if dat else None,
                            options=options,
                            logger=mock.Mock())
        job.run()
        assert os.path.isfile('view_view.html')
        job.logger.log.assert_called_with(
            'Trajectory visualization are written into view_view.html')


class TestClash:

    MODIFIED = copy.deepcopy(HEX_RDR)
    MODIFIED.pair_coeffs.dist = 18

    @pytest.fixture
    def clash(self, trj, gids, rdr, logger):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'clash'])
        return analyzer.Clash(trj=traj.Traj(trj, options=options),
                              gids=gids,
                              rdr=rdr,
                              logger=logger)

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
    @pytest.mark.parametrize(
        'gids,expected,data',
        [([1], 'WARNING: Clash requires least two atoms selected.', None),
         ([5, 0, 1], None, [[2]]), ([5, 0], None, [[1]])])
    def testCalculate(self, clash, expected, data, called):
        clash.log = called
        clash.calculate()
        numpyutils.assert_almost_equal(clash.data, data)


class TestRDF:

    @pytest.fixture
    def rdf(self, trj, gids, rdr, group, logger):
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'rdf', '-NAME', 'lmp_traj'])
        return analyzer.RDF(trj=traj.Traj(trj, options=options),
                            options=options,
                            gids=gids,
                            rdr=rdr,
                            parm=group,
                            logger=logger)

    @pytest.mark.parametrize('trj,rdr,parm,dirname',
                             [(HEX_TRJ, HEX_RDR, None, None)])
    @pytest.mark.parametrize('gids,expected,mdata', [
        ([1], 'WARNING: RDF requires least two atoms selected.', None),
        ([5, 0, 1
          ], 'The volume fluctuates: [109519.24 109519.24] Å^3', 41913.204277),
        ([5, 0
          ], 'The volume fluctuates: [109519.24 109519.24] Å^3', 5559.9628738)
    ])
    def testCalculate(self, rdf, expected, mdata, called):
        rdf.log = called
        rdf.calculate()
        numpyutils.assert_almost_equal(
            None if rdf.data is None else rdf.data.max().max(), mdata)

    def testLabel(self):
        assert 'g (r)' == analyzer.RDF().label

    @pytest.mark.parametrize('trj', [(HEX_TRJ)])
    @pytest.mark.parametrize(
        'rdr,gids,dirname,parm,expected,result',
        [(HEX_RDR, [5, 0, 1], None, None,
          'RDF peak 4.191e+04 ± nan found at 1.52 Å', [[41913.2042769, np.nan]
                                                       ]),
         (None, None, TEST0045, pd.Series(index=pd.Index([], name='rdf')),
          'RDF peak 92.65 ± 92.65 found at 3.46 Å', [[92.65, 92.65]])])
    def testFit(self, rdf, result, called):
        rdf.log = called
        rdf.set()
        rdf.merge()
        rdf.fit()
        np.testing.assert_almost_equal(rdf.result, result)

    def testFull(self):
        assert 'radial distribution function' == analyzer.RDF().full


class TestMSD:

    FOUR = os.path.join(HEX, 'four_frames.custom')

    @pytest.fixture
    def msd(self, trj, gids, rdr, logger):
        if gids is not None:
            gids = np.array(gids)
        options = parserutils.LmpTraj().parse_args(
            [trj, '-last_pct', '0.8', '-task', 'msd'])
        return analyzer.MSD(trj=traj.Traj(trj, options=options),
                            options=options,
                            gids=gids,
                            rdr=rdr,
                            logger=logger)

    @pytest.mark.parametrize(
        'trj,rdr,gids,spct,epct,expected,mdata,name',
        [(FOUR, HEX_RDR, None, 0.1, 0.2, None, 0.4820897, 'Tau (ps) (0 3)'),
         (FOUR, None, None, 0.1, 0.2, None, 0.48032087, 'Tau (ps) (0 3)'),
         (FOUR, HEX_RDR, [0, 1], 0.1, 0.2, None, 0.99159753, 'Tau (ps) (0 3)'),
         (FOUR, HEX_RDR, [0, 1], 0.4, 0.2, None, 0.99159753, 'Tau (ps) (1 3)'),
         (AR_TRJ, None, [], 0.1, 0.2, "MSD requires least one atom selected.",
          None, None),
         (HEX_TRJ, None, None, 0.1, 0.2,
          "Only one trajectory frame selected for MSD.", None, None)])
    def testCalculate(self, msd, spct, epct, expected, mdata, name, called):
        msd.warning = called
        msd.calculate(spct=spct, epct=epct)
        if mdata is None:
            return
        np.testing.assert_almost_equal(msd.data.max().max(), mdata, decimal=4)
        assert name == msd.data.index.name

    def testLabel(self):
        assert 'MSD (Å^2)' == analyzer.MSD().label

    @pytest.mark.parametrize('trj,rdr,gids,spct,epct,expected,msg', [
        (FOUR, HEX_RDR, None, 0.1, 0.2, [[4.01749017e-06, 8.75353197e-07]],
         'Diffusion Coefficient 4.018e-06 ± 8.754e-07 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.955)'
         ),
        (FOUR, None, None, 0.1, 0.2, [[4.00255214e-06, 8.72030195e-07]],
         'Diffusion Coefficient 4.003e-06 ± 8.721e-07 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.955)'
         ),
        (FOUR, HEX_RDR, [0, 1], 0.1, 0.2, [[8.2679990e-06, 1.5476629e-06]],
         'Diffusion Coefficient 8.263e-06 ± 1.546e-06 cm^2/s calculated by fitting MSD ∈ [0 2] ps. (R-squared: 0.966)'
         ),
        (FOUR, HEX_RDR, [0, 1], 0.4, 0.2, [[1.09486298e-05, 0.00000e+00]],
         'Diffusion Coefficient 1.094e-05 ± 0 cm^2/s calculated by fitting MSD ∈ [1 2] ps. (R-squared: 1)'
         )
    ])
    def testFit(self, msd, spct, epct, expected, msg):
        msd.calculate(spct=spct, epct=epct)
        msd.fit()
        np.testing.assert_almost_equal(msd.result.values, expected, 4)
        msd.logger.log.assert_called_with(msg)

    def testFull(self):
        assert 'mean squared displacement' == analyzer.MSD().full


class TestTotEng:

    AR_LOG = os.path.join(AR_DIR, 'lammps_lmp.log')

    @pytest.fixture
    def tot_eng(self, logfile):
        if logfile is None:
            return analyzer.TotEng()
        options = parserutils.LmpLog().parse_args([logfile])
        lmp_log = lmplog.Log(logfile, options=options)
        return analyzer.TotEng(thermo=lmp_log.thermo)

    @pytest.mark.parametrize('logfile,expected', [(None, 0), (AR_LOG, 55)])
    def testInit(self, tot_eng, expected):
        assert expected == tot_eng.sidx

    @pytest.mark.parametrize('logfile,expected', [(AR_LOG, 8.338777)])
    def testSet(self, tot_eng, expected):
        tot_eng.set()
        np.testing.assert_almost_equal(tot_eng.data.max().max(), expected)

    def testFull(self):
        assert 'thermodynamic information' == analyzer.TotEng().full


class TestMerger:

    @pytest.fixture
    def merger(self, jobs, tsk, args, logger):
        Anlz = analyzer.Merger.getAnlz(tsk)
        options = parserutils.Workflow().parse_args(args)
        groups = None if jobs is None else task.LmpAgg(*jobs).groups
        return analyzer.Merger(Anlz,
                               options=options,
                               groups=groups,
                               logger=logger)

    @pytest.mark.parametrize('args,dirname,tsk,expected',
                             [(['-NAME', 'lmp_traj'], TEST0045, 'rdf', (1, 2)),
                              ([], 'empty', 'toteng', None),
                              (['-NAME', 'lmp_log'], TEST0046, 'rdf', None),
                              (['-NAME', 'lmp_log'], TEST0046, 'toteng',
                               (2, 2))])
    def testMerge(self, merger, expected):
        merger.merge()
        assert expected == (None if merger.data is None else merger.data.shape)

    @pytest.mark.parametrize('tsk,expected', [('rdf', 'rdf'), ('msd', 'msd')])
    def testName(self, tsk, expected):
        assert expected == analyzer.Merger.getAnlz(tsk).name

    @pytest.mark.parametrize('args,dirname,tsk,expected',
                             [(['-NAME', 'lmp_log'], TEST0046, 'toteng', 2),
                              (['-NAME', 'lmp_traj'], TEST0045, 'rdf', 1),
                              (['-NAME', 'lmp_traj'], TEST0045, 'xyz', 0)])
    def testMain(self, tsk, args, jobs, expected):
        options = parserutils.Workflow().parse_args(['-JOBNAME', 'nm', *args])
        groups = None if jobs is None else task.LmpAgg(*jobs).groups
        analyzer.Merger.main(tsk, options=options, groups=groups)
        assert expected == len(glob.glob('*.csv'))

    @pytest.mark.parametrize('tsk,expected', [('rdf', analyzer.RDF),
                                              ('xyz', None)])
    def testGetAnlz(self, tsk, expected):
        assert expected == analyzer.Merger.getAnlz(tsk)
