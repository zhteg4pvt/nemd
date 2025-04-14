import os
from unittest import mock

import pandas as pd
import pytest

from nemd import analyzer
from nemd import envutils
from nemd import parserutils

TEST0027 = envutils.test_data('itest', '0027_test')
TEST0045 = envutils.test_data('itest', '0045_test')
TEST0037 = envutils.test_data('itest', '0037_test')


class TestBase:

    EMPTY = pd.DataFrame()
    DATA = pd.DataFrame({'density': [1, 0, 2]})
    DATA.index = pd.Index([5, 2, 6], name='time')

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
    @pytest.mark.parametrize('data,expected', [(EMPTY, False), (DATA, True)])
    @pytest.mark.parametrize('line,marker,selected',
                             [(None, None, None),
                              (DATA, '*', pd.DataFrame([[1], [2]]))])
    def testPlot(self, base, line, marker, selected, expected, tmp_dir):
        base.plot(line=line, marker=marker, selected=selected)
        assert expected == os.path.exists('name_base.png')

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
    def testRead(self, job, expected):
        with mock.patch.object(job, 'name', 'density'):
            job.read()
        assert expected == (*job.data.shape, job.sidx, job.eidx)

    @pytest.mark.parametrize('args,parm,dirname', [([], None, None)])
    @pytest.mark.parametrize('name,expected',
                             [('r (Å)', ('r', 'Å', 0, None)),
                              ('Time (ps) (1)', ('Time', 'ps', 1, None)),
                              ('Tau (ps) (0 2)', ('Tau', 'ps', 0, 2))])
    def testParseIndex(self, job, name, expected):
        assert expected == job.parseIndex(name)
