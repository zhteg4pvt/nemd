import collections
import glob
import json
import os
import shutil
from unittest import mock

import flow
import pytest

from nemd import envutils
from nemd import parserutils
from nemd import taskbase

TEST_0001 = envutils.test_data('itest', '0001_test')
FAIL_0001 = envutils.test_data('itest', '0001_fail')
AB_LMP_TRAJ = envutils.test_data('itest', 'ab_lmp_traj')


@pytest.fixture
def jobs(dirname, tmp_dir):
    if dirname is None:
        return [None]
    test_dir = envutils.test_data('itest', dirname)
    shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
    jobs = flow.project.FlowProject.get_project(os.curdir).find_jobs()
    return list(jobs)


class TestJob:

    @pytest.fixture
    def raw(self, name, tmp_dir):
        return type(name, (taskbase.Job, ), {})

    @pytest.fixture
    def job(self, jobs, status, jobname, tmp_dir):
        return taskbase.Job(*jobs,
                            jobname=jobname,
                            status=status,
                            logger=mock.Mock())

    @pytest.mark.parametrize('name,expected', [('Job', 'job'),
                                               ('Check', 'check')])
    def testDefault(self, raw, expected):
        assert expected == raw.default

    @pytest.mark.parametrize("name", [('Job'), ('Check')])
    def testAgg(self, raw):
        assert False == raw.agg

    @pytest.mark.parametrize('name,jobname,expected',
                             [('Check', None, 'check'),
                              ('Check', 'myname', 'myname')])
    def testGetOpr(self, raw, jobname, expected, flow_opr):
        opr = raw.getOpr(jobname=jobname)
        assert expected == opr.jobname
        assert False == opr.opr._flow_cmd
        assert True == opr.opr._flow_with_job
        assert True == opr.opr._flow_aggregate._is_default_aggregator
        assert issubclass(opr.cls, taskbase.Job)

    @pytest.mark.parametrize("name", [('Job')])
    def testRunOpr(self, raw):
        with mock.patch('nemd.taskbase.Job.run') as mocked:
            assert raw.runOpr() is None
        assert mocked.called
        assert os.path.isfile('.job_document.json')

    @pytest.mark.parametrize('jobs,status,jobname', [([], None, None)])
    def testRun(self, job):
        job.run()
        assert job.out is False

    @pytest.mark.parametrize('jobs,status,jobname', [([], None, 'check')])
    def testOut(self, jobname, job):
        assert job.out is None
        job.out = False
        assert job.out is False
        with open(f'.{jobname}_document.json') as fh:
            data = json.load(fh)
        assert {'status': False} == data

    @pytest.mark.parametrize('jobs,status,jobname', [([], None, 'check')])
    def testGetCmd(self, job):
        assert job.getCmd() is None

    @pytest.mark.parametrize('dirname,status,jobname,expected',
                             [(TEST_0001, {}, 'check', True),
                              (TEST_0001, {}, 'tag', False)])
    def testPostOpr(self, jobname, jobs, job, expected):
        assert expected == job.postOpr(jobname=jobname, *jobs)

    @pytest.mark.parametrize('dirname,jobname,status,expected,logged',
                             [(None, None, None, False, False),
                              (None, None, dict(job=True), True, False),
                              (None, None, dict(job=False), False, False),
                              (TEST_0001, 'check', None, True, False),
                              (FAIL_0001, 'check', {}, True, True)])
    def testPost(self, jobname, job, status, expected, logged):
        assert expected == job.post()
        assert logged == job.logger.log.called
        if not logged:
            return
        job.logger.log.call_args_list[0][0][0].startswith(jobname)
        assert (jobname, '06b39c3b9b6541a2dc6e15baa6734cb2') in status

    @pytest.mark.parametrize('dirname,jobname,status,expected',
                             [(TEST_0001, None, None, 2),
                              (None, None, None, 0)])
    def testGetJobs(self, job, expected):
        assert expected == len(job.getJobs())

    @pytest.mark.parametrize('jobs,status,jobname', [([], None, 'check')])
    def testLog(self, job):
        job.log('msg')
        assert 'msg' == job.out == job.getData()['status']
        job.log('another')
        assert 'msg\nanother' == job.out == job.getData()['status']


class TestAgg:

    @pytest.fixture
    def raw(self, name, tmp_dir):
        return type(name, (taskbase.Agg, ), {})

    @pytest.fixture
    def agg(self, jobs, jobname, status):
        return taskbase.Agg(*jobs,
                            jobname=jobname,
                            status=status,
                            logger=mock.Mock())

    @pytest.mark.parametrize('name,expected', [('Agg', 'agg'),
                                               ('TimeAgg', 'time_agg')])
    def testDefault(self, raw, expected):
        assert expected == raw.default

    @pytest.mark.parametrize("name", [('Agg'), ('TimeAgg')])
    def testAgg(self, raw):
        assert True == raw.agg

    @pytest.mark.parametrize('name,jobname', [('TimeAgg', 'time_agg')])
    def testGetOpr(self, jobname, raw, flow_opr):
        opr = raw.getOpr(jobname=jobname)
        assert jobname == opr.opr.keywords['jobname']
        assert False == opr.opr._flow_aggregate._is_default_aggregator

    @pytest.mark.parametrize(
        'dirname,jobname,status,expected,logged',
        [(AB_LMP_TRAJ, 'time_agg', {}, True, True),
         (AB_LMP_TRAJ, 'time_agg', dict(time_agg=True), True, False),
         (AB_LMP_TRAJ, 'time_agg2', {}, False, False),
         (AB_LMP_TRAJ, 'time_agg2', dict(time_agg2=True), True, False)])
    def testPost(self, agg, jobname, status, expected, logged):
        assert expected == agg.post()
        assert logged == agg.logger.log.called
        if not logged:
            return
        assert not agg.logger.log.call_args_list[0][0][0].startswith(jobname)
        assert jobname in status


class TestCmd:

    @pytest.fixture
    def raw(self, name, tmp_dir):
        return type(name, (taskbase.Cmd, ), {})

    @pytest.fixture
    def cmd(self, name, file, parser, args_tmpl, jobs, tmp_dir):
        attrs = dict(FILE=file, ParserClass=parser, ARGS_TMPL=args_tmpl)
        Name = type(name, (taskbase.Cmd, ), attrs)
        return Name(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('name,expected', [('MolBldr', 'mol_bldr')])
    def testDefault(self, raw, expected):
        assert expected == raw.default

    @pytest.mark.parametrize("name", [('MolBldr')])
    def testAgg(self, raw):
        assert False == raw.agg

    @pytest.mark.parametrize('name,jobname,expected',
                             [('MolBldr', None, 'mol_bldr'),
                              ('TrajLmp', 'myname', 'myname')])
    def testGetOpr(self, raw, jobname, expected, flow_opr):
        opr = raw.getOpr(jobname=jobname)
        assert expected == opr.jobname
        assert True == opr.opr._flow_cmd
        assert True == opr.opr._flow_with_job
        assert True == opr.opr._flow_aggregate._is_default_aggregator
        assert issubclass(opr.cls, taskbase.Cmd)

    @pytest.mark.parametrize(
        "name,jobname,expected",
        [('Job', None, 'nemd_run -JOBNAME job'),
         ('Job', 'mol_bldr', 'nemd_run -JOBNAME mol_bldr')])
    def testRunOpr(self, raw, jobname, expected):
        assert expected == raw.runOpr(jobname=jobname)
        assert not glob.glob('.*_document.json')

    @pytest.mark.parametrize('dirname', [AB_LMP_TRAJ])
    @pytest.mark.parametrize('name,,jobname,expected',
                             [('AmorpBldr', 'amorp_bldr', True),
                              ('LmpTraj', 'lmp_traj', False)])
    def testPostOpr(self, jobs, name, jobname, raw, expected):
        assert expected == raw.postOpr(*jobs, jobname=jobname)

    @pytest.mark.parametrize('dirname', [AB_LMP_TRAJ])
    @pytest.mark.parametrize(
        'name,file,parser,args_tmpl,expected,status',
        [('AmorpBldr', None, None, None, True, 'amorp_bldr.in'),
         ('LmpTraj', None, None, None, False, None)])
    def testPost(self, cmd, expected, status):
        assert expected == cmd.post()
        assert status == next(iter(cmd.status.values()))
        assert not cmd.logger.log.called

    @pytest.mark.parametrize('dirname', [AB_LMP_TRAJ])
    @pytest.mark.parametrize('name,file,parser,args_tmpl,expected',
                             [('AmorpBldr', 'amorp_bldr_driver.py',
                               parserutils.AmorpBldr, None, '[Ar]'),
                              ('Lammps', 'lammps_driver.py',
                               parserutils.Lammps, [None], 'amorp_bldr.in')])
    def testAddfiles(self, cmd, expected):
        cmd.addfiles()
        assert cmd.args[0].endswith(expected)

    @pytest.mark.parametrize('dirname', [AB_LMP_TRAJ])
    @pytest.mark.parametrize(
        'name,file,parser,args_tmpl,expected',
        [('AmorpBldr', 'amorp_bldr_driver.py', parserutils.AmorpBldr, None, 9),
         ('Lammps', 'lammps_driver.py', parserutils.Lammps, [None], 5)])
    def testRmUnknown(self, cmd, expected):
        cmd.addfiles()
        cmd.rmUnknown()
        assert expected == len(cmd.args)

    @pytest.mark.parametrize('name,jobname,expected',
                             [('MolBldr', None, 'mol_bldr'),
                              ('MolBldr', 'myname', 'myname')])
    def testSetName(self, raw, jobname, expected):
        mol_bldr = raw(jobname=jobname)
        mol_bldr.args = []
        mol_bldr.setName()
        assert ['-JOBNAME', expected] == mol_bldr.args

    @pytest.mark.parametrize('dirname', [AB_LMP_TRAJ])
    @pytest.mark.parametrize('name,file,parser,args_tmpl,expected', [
        ('AmorpBldr', 'amorp_bldr_driver.py', parserutils.AmorpBldr, None, 11),
        ('Lammps', 'lammps_driver.py', parserutils.Lammps, [None], 7)
    ])
    def testGetCmd(self, cmd, expected):
        cmd.run()
        assert expected == len(cmd.getCmd().split())
