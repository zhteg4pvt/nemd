import glob
import json
import os
from unittest import mock

import pytest

from nemd import envutils
from nemd import jobutils
from nemd import parserutils
from nemd import taskbase

TEST0037 = os.path.join('0037_test', 'workspace',
                        'c57080b7efdda6099a8667ccf28534c9')


class TestJob:
    TEST_0001 = envutils.test_data('itest', '0001_test')
    FAIL_0001 = envutils.test_data('itest', '0001_fail')

    @pytest.fixture
    def job(self, jobs, jobname, status):
        return taskbase.Job(*jobs,
                            jobname=jobname,
                            status=status,
                            logger=mock.Mock())

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testAgg(self, job):
        assert False == job.agg

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testGetOpr(self, job, jobname, flow_opr):
        opr = job.getOpr(jobname=jobname)
        assert jobname == opr.jobname
        assert False == opr.opr._flow_cmd
        assert True == opr.opr._flow_with_job
        assert True == opr.opr._flow_aggregate._is_default_aggregator
        assert issubclass(opr.cls, taskbase.Job)

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testRunOpr(self, job, jobname):
        with mock.patch('nemd.taskbase.Job.run') as mocked:
            assert job.runOpr(jobname=jobname) is None
        assert mocked.called
        assert os.path.isfile('.name_document.json')

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testRun(self, job):
        job.run()
        assert job.out is True

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testOut(self, jobname, job):
        assert job.out is None
        job.out = False
        assert job.out is False
        assert jobutils.Job(jobname=jobname)['status'] is False

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testGetCmd(self, job):
        assert job.getCmd() is None

    @pytest.mark.parametrize('dirname,status,jobname,expected',
                             [(TEST_0001, {}, 'check', True),
                              (TEST_0001, {}, 'tag', False)])
    def testPostOpr(self, jobname, jobs, job, expected):
        assert expected == job.postOpr(jobname=jobname, *jobs)

    @pytest.mark.parametrize('dirname,jobname,status,expected,logged',
                             [(None, None, None, False, False),
                              (None, None, True, True, False),
                              (None, None, False, False, False),
                              (TEST_0001, 'check', None, True, False),
                              (FAIL_0001, 'check', {}, True, True)])
    def testPost(self, jobname, job, status, expected, logged):
        if status is True or status is False:
            job.status = {(job.jobname, job.job.dirname): status}
        assert expected == job.post()
        assert logged == job.logger.log.called
        if not logged:
            return
        job.logger.log.call_args_list[0][0][0].startswith(jobname)
        assert (jobname, job.job.dirname) in status

    @pytest.mark.parametrize('dirname,jobname,status,expected',
                             [(TEST_0001, None, None, 2),
                              (None, None, None, 0)])
    def testGetJobs(self, job, expected):
        assert expected == len(job.getJobs())

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testLog(self, job):
        job.log('msg')
        assert 'msg' == job.out == job.job['status']
        job.log('another')
        assert 'msg\nanother' == job.out == job.job['status']

    @pytest.mark.parametrize("dirname,jobname,status",
                             [('empty', 'name', None)])
    def testClean(self, job):
        job.run()
        assert os.path.isfile(job.job.file)
        job.clean()
        assert not os.path.isfile(job.job.file)


class TestAgg:

    @pytest.fixture
    def agg(self, jobs, jobname, status):
        return taskbase.Agg(*jobs,
                            jobname=jobname,
                            status=status,
                            logger=mock.Mock())

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testAgg(self, agg):
        assert True == agg.agg

    @pytest.mark.parametrize("dirname,jobname,status", [('empty', 'name', {})])
    def testGetOpr(self, jobname, agg):
        opr = agg.getOpr(jobname=jobname)
        assert jobname == opr.opr.keywords['jobname']
        assert False == opr.opr._flow_aggregate._is_default_aggregator

    @pytest.mark.parametrize('dirname,jobname,status,expected,logged',
                             [('0037_test', 'test_agg', {}, True, True),
                              ('0037_test', 'time_agg', True, True, False),
                              ('0037_test', 'time_agg2', {}, False, False),
                              ('0037_test', 'time_agg2', True, True, False)])
    def testPost(self, agg, jobname, status, expected, logged):
        if status is True or status is False:
            agg.status = {(agg.jobname, agg.job.dirname): status}
        assert expected == agg.post()
        assert logged == agg.logger.log.called
        if not logged:
            return
        assert not agg.logger.log.call_args_list[0][0][0].startswith(jobname)
        assert (agg.jobname, agg.job.dirname) in status


class TestCmd:

    THREE = (None, None, None)

    @pytest.fixture
    def cmd(self, name, file, parser, tmpl, jobs):
        attrs = dict(FILE=file, ParserClass=parser, TMPL=tmpl)
        Name = type(name, (taskbase.Cmd, ), attrs)
        return Name(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('dirname,file,parser,tmpl', [('empty', *THREE)])
    @pytest.mark.parametrize('name', ['MolBldr'])
    def testAgg(self, cmd):
        assert False == cmd.agg

    @pytest.mark.parametrize('dirname,file,parser,tmpl', [('empty', *THREE)])
    @pytest.mark.parametrize('name,jobname,expected',
                             [('MolBldr', None, 'mol_bldr'),
                              ('TrajLmp', 'myname', 'myname')])
    def testGetOpr(self, cmd, jobname, expected, flow_opr):
        opr = cmd.getOpr(jobname=jobname)
        assert expected == opr.jobname
        assert True == opr.opr._flow_cmd
        assert True == opr.opr._flow_with_job
        assert True == opr.opr._flow_aggregate._is_default_aggregator
        assert issubclass(opr.cls, taskbase.Cmd)

    @pytest.mark.parametrize('dirname,file,parser,tmpl', [('empty', *THREE)])
    @pytest.mark.parametrize(
        "name,jobname,expected",
        [('Job', None, 'nemd_run -JOBNAME job'),
         ('Job', 'mol_bldr', 'nemd_run -JOBNAME mol_bldr')])
    def testRunOpr(self, cmd, jobname, expected):
        assert expected == cmd.runOpr(jobname=jobname)
        assert not glob.glob('.*_document.json')

    @pytest.mark.parametrize('dirname,file,parser,tmpl',
                             [('0037_test', *THREE)])
    @pytest.mark.parametrize('name,jobname,expected',
                             [('AmorpBldr', 'amorp_bldr', False),
                              ('LmpTraj', 'lmp_traj', True)])
    def testPostOpr(self, jobs, name, jobname, cmd, expected):
        assert expected == cmd.postOpr(*jobs, jobname=jobname)

    @pytest.mark.parametrize('dirname,file,parser,tmpl,',
                             [('0037_test', *THREE)])
    @pytest.mark.parametrize('name,expected',
                             [('AmorpBldr', (False, None)),
                              ('LmpTraj', (True, 'lmp_traj.log'))])
    def testPost(self, cmd, expected):
        assert expected == (cmd.post(), next(iter(cmd.status.values())))
        assert not cmd.logger.log.called

    @pytest.mark.parametrize('dirname', ['0045_test'])
    @pytest.mark.parametrize('name,file,parser,tmpl,expected',
                             [('AmorpBldr', 'amorp_bldr_driver.py',
                               parserutils.AmorpBldr, None, '[Ar]'),
                              ('Lammps', 'lammps_driver.py',
                               parserutils.Lammps, [None], 'amorp_bldr.in')])
    def testAddfiles(self, cmd, expected):
        cmd.addfiles()
        assert cmd.args[0].endswith(expected)

    @pytest.mark.parametrize('dirname', ['0045_test'])
    @pytest.mark.parametrize(
        'name,file,parser,tmpl,expected',
        [('AmorpBldr', 'amorp_bldr_driver.py', parserutils.AmorpBldr, None, 19),
         ('Lammps', 'lammps_driver.py', parserutils.Lammps, [None], 5)])
    def testRmUnknown(self, cmd, expected):
        cmd.addfiles()
        cmd.rmUnknown()
        assert expected == len(cmd.args)

    @pytest.mark.parametrize('name,dirname,file,parser,tmpl', [('MolBldr', 'empty', *THREE)])
    @pytest.mark.parametrize('word,expected', [('Ar', ['Ar']),
                                               ('[Ar]', ['"[Ar]"']),
                                               ('"[Ar]"', ['"[Ar]"']),
                                               ("'[Ar]'", ["'[Ar]'"])])
    def testAddQuot(self, cmd, word, expected):
        cmd.args = [word]
        cmd.addQuot()
        assert expected == cmd.args

    @pytest.mark.parametrize('name,dirname,file,parser,tmpl', [('MolBldr', 'empty', *THREE)])
    @pytest.mark.parametrize('word,expected', [('Ar', 'Ar'), ('@', '"@"'),
                                               ('[Ar]', '"[Ar]"'),
                                               ('"[Ar]"', '"[Ar]"'),
                                               ("'[Ar]'", "'[Ar]'")])
    def testQuote(self, cmd, word, expected):
        assert expected == cmd.quote(word)

    @pytest.mark.parametrize('name,dirname,file,parser,tmpl', [('MolBldr', 'empty', *THREE)])
    def testSetName(self, cmd,):
        cmd.args = []
        cmd.setName()
        assert ['-JOBNAME', 'mol_bldr'] == cmd.args

    @pytest.mark.parametrize('dirname', ['0045_test'])
    @pytest.mark.parametrize('name,file,parser,tmpl,expected', [
        ('AmorpBldr', 'amorp_bldr_driver.py', parserutils.AmorpBldr, None, 23),
        ('Lammps', 'lammps_driver.py', parserutils.Lammps, [None], 9)
    ])
    def testGetCmd(self, cmd, expected):
        cmd.run()
        assert expected == len(cmd.getCmd().split())
