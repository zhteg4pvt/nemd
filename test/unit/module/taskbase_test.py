import collections
import datetime
import json
import os
import shutil
from unittest import mock

import flow
import pytest

from nemd import envutils
from nemd import jobutils
from nemd import parserutils
from nemd import taskbase

TEST_0001 = envutils.test_data('itest', '0001_test')
FAIL_0001 = envutils.test_data('itest', '0001_fail')
AB_LMP_TRAJ = envutils.test_data('itest', 'ab_lmp_traj')


@pytest.fixture
def flow_opr():
    project = flow.FlowProject
    functions = project._OPERATION_FUNCTIONS
    postconditions = project._OPERATION_POSTCONDITIONS
    project._OPERATION_FUNCTIONS = []
    project._OPERATION_POSTCONDITIONS = collections.defaultdict(list)
    yield
    project._OPERATION_FUNCTIONS = functions
    project._OPERATION_POSTCONDITIONS = postconditions


class TestJob:

    @pytest.fixture
    def job(self, name, tmp_dir):
        return taskbase.Job(name=name)

    @pytest.fixture
    def jobs(self, dirname, tmp_dir):
        if dirname is None:
            return [None]
        test_dir = envutils.test_data('itest', dirname)
        shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
        jobs = flow.project.FlowProject.get_project(os.curdir).find_jobs()
        return list(jobs)

    @pytest.mark.parametrize('name,expected', [('Job', 'job'),
                                               ('MolBldr', 'mol_bldr')])
    def testDefault(self, name, expected):
        job = type(name, (taskbase.Job, ), {})
        assert expected == job.default

    @pytest.mark.parametrize("name", [('Job'), ('MolBldr')])
    def testAgg(self, name):
        assert False == type(name, (taskbase.Job, ), {}).agg

    @pytest.mark.parametrize('cname,name,cmd,expected',
                             [('Check', None, None, ['check', False]),
                              ('Job', 'mol_bldr', True, ['mol_bldr', True])])
    def testGetOpr(self, cname, name, cmd, expected, flow_opr):
        job = type(cname, (taskbase.Job, ), {})
        opr = job.getOpr(name=name, cmd=cmd)
        assert issubclass(opr.cls, taskbase.Job)
        assert expected == [opr.name, opr.opr._flow_cmd]

    def testRunOpr(self, tmp_dir):
        with mock.patch('nemd.taskbase.Job.run') as mocked:
            assert taskbase.Job.runOpr() is None
        assert mocked.called
        assert os.path.isfile('.job_document.json')

    @pytest.mark.parametrize('name', ['mol_bldr'])
    def testRun(self, job):
        job.run()
        assert job.out is False

    @pytest.mark.parametrize('name', ['mol_bldr'])
    def testOut(self, name, job):
        assert job.out is None
        job.out = False
        assert job.out is False
        with open(f'.{name}_document.json') as fh:
            data = json.load(fh)
        assert {'status': False} == data

    @pytest.mark.parametrize('name', ['mol_bldr'])
    def testGetCmd(self, job):
        assert job.getCmd() is None

    @pytest.mark.parametrize('dirname,name,expected',
                             [(TEST_0001, 'check', True),
                              (TEST_0001, 'tag', False)])
    def testPostOpr(self, name, jobs, expected):
        assert expected == taskbase.Job.postOpr(name=name, *jobs)

    @pytest.mark.parametrize('dirname,name,status,expected,logged',
                             [(None, None, None, False, False),
                              (None, None, True, True, False),
                              (None, None, False, False, False),
                              (TEST_0001, 'check', None, True, False),
                              (FAIL_0001, 'check', None, True, True)])
    def testPost(self, jobs, name, status, expected, logged):
        if status is not None:
            status = {'job': status}
        logger = mock.Mock()
        job = taskbase.Job(name=name, *jobs, status=status, logger=logger)
        assert expected == job.post()
        assert logged == logger.log.called

    @pytest.mark.parametrize('dirname,expected', [(TEST_0001, 2), (None, 0)])
    def testGetJobs(self, jobs, expected):
        assert expected == len(taskbase.Job(*jobs).getJobs())

    @pytest.mark.parametrize('name', ['mol_bldr'])
    def testLog(self, job):
        job.log('msg')
        assert 'msg' == job.out == job.getData()['status']
        job.log('another')
        assert 'msg\nanother' == job.out == job.getData()['status']


class TestJob:

    @pytest.fixture
    def job(self, file, parser, args_tmpl, tmp_dir):
        jdir = envutils.test_data('itest', '0923fc12bdc6d40132eb65ed96588f52')
        shutil.copytree(jdir, os.curdir, dirs_exist_ok=True)
        with mock.patch('nemd.taskbase.Job.FILE', file):
            job = taskbase.Job(jobutils.Job(job_dir=os.curdir))
        job.FILE = file
        job.ParserClass = parser
        job.ARGS_TMPL = args_tmpl
        return job

    @pytest.mark.parametrize(
        'file,parser,args_tmpl,expected',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None, '[Ar]'),
         ('lammps_driver.py', parserutils.Lammps, [None], 'amorp_bldr.in')])
    def testAddfiles(self, job, expected):
        job.addfiles()
        assert expected == job.args[0]

    @pytest.mark.parametrize(
        'file,parser,args_tmpl,expected',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None, 9),
         ('lammps_driver.py', parserutils.Lammps, [None], 5)])
    def testRmUnknown(self, job, expected):
        job.addfiles()
        job.rmUnknown()
        assert expected == len(job.args)

    @pytest.mark.parametrize(
        'file,parser,args_tmpl,expected',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None, 'amorp_bldr'),
         ('lammps_driver.py', parserutils.Lammps, [None], 'lammps')])
    def testSetName(self, job, expected):
        job.args = []
        job.setName()
        assert ['-JOBNAME', expected] == job.args

    @pytest.mark.parametrize(
        'file,parser,args_tmpl,expected',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None, 89),
         ('lammps_driver.py', parserutils.Lammps, [None], 62)])
    def testGetCmd(self, job, expected):
        job.run()
        assert expected == len(job.getCmd())

    @pytest.mark.parametrize(
        'file,parser,args_tmpl,expected',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None, True),
         ('wrong_name_driver.py', parserutils.Lammps, [None], False)])
    def testPost(self, job, expected):
        assert expected == job.post()

    @pytest.mark.parametrize('file,parser,args_tmpl,expected', [
        ('amorp_bldr_driver.py', parserutils.AmorpBldr, None, 'amorp_bldr.in'),
        ('wrong_name_driver.py', parserutils.Lammps, [None], None)
    ])
    def testOutfile(self, job, expected):
        assert expected == job.outfile
        job.outfile = 'file'
        assert 'file' == job.outfile

    @pytest.mark.parametrize(
        'file,parser,args_tmpl',
        [('amorp_bldr_driver.py', parserutils.AmorpBldr, None)])
    def testClean(self, job):
        assert job.post() is True
        job.clean()
        assert job.post() is False


class TestAgg:

    CB_LMP_LOG = 'cb_lmp_log'
    TEST_MB_LMP_LOG = 'test_mb_lmp_log'

    @pytest.fixture
    def agg(self, dirname, name, tmp_dir):
        test_dir = envutils.test_data('itest', dirname)
        shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
        project = flow.project.FlowProject.get_project(os.curdir)
        jobs = list(project.find_jobs())
        agg = taskbase.Agg(*jobs, name=name)
        return agg

    # @pytest.mark.parametrize('dirname,name,expected',
    #                          [(CB_LMP_LOG, 'name', '00:00:06'),
    #                           (TEST_MB_LMP_LOG, 'name', '00:00:02')])
    # def testRun(self, agg, expected):
    #     with mock.patch.object(agg, 'log') as mocked:
    #         agg.run()
    #     assert expected == mocked.call_args_list[0][0][0].split()[-1]

    @pytest.mark.parametrize('dirname,name,expected',
                             [(CB_LMP_LOG, 'lmp_log_#_agg', False),
                              (TEST_MB_LMP_LOG, 'wrong_#_agg', None)])
    def testMessage(self, agg, expected):
        assert expected == agg.message
        agg.message = 'hi'
        assert 'hi' == agg.message

    # @pytest.mark.parametrize("delta, expected",
    #                          [(datetime.timedelta(hours=3), '59:59'),
    #                           (datetime.timedelta(minutes=3), '03:00')])
    # def testDelta2str(self, delta, expected):
    #     assert expected == taskbase.AggJob.delta2str(delta)

    @pytest.mark.parametrize('dirname,name', [(CB_LMP_LOG, 'lmp_log_#_agg')])
    def testClean(self, agg):
        agg.clean()
        assert not agg.post()
