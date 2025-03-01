import datetime
import os
import shutil
import sys
import types
from unittest import mock

import pytest
from flow.project import FlowProject

from nemd import envutils
from nemd import jobutils
from nemd import task

LMP_LOG = 'lmp_log'
LOG_ID = '1c57f0964168565049315565b1388af9'
LOG_DRIVER = task.LmpLog.DRIVER

TEST_DIR = envutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'data', 'itest')


def get_job(basename='6e4cfb3bcc2a689d42d099e51b9efe23'):
    job_dir = os.path.join(BASE_DIR, basename)
    shutil.copytree(job_dir, os.curdir, dirs_exist_ok=True)
    return jobutils.Job(job_dir=os.curdir)


def get_jobs(basename='e053136e2cd7374854430c868b3139e1'):
    proj_dir = os.path.join(BASE_DIR, basename)
    shutil.copytree(proj_dir, os.curdir, dirs_exist_ok=True)
    proj = FlowProject.get_project(os.curdir)
    return proj.find_jobs()


class TestLmpLogJob:

    @pytest.fixture
    def job(self, tmp_dir):
        job = get_job(basename=LOG_ID)
        return task.LmpLogJob(job, name='lmp_log', driver=task.LmpLog.DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args[1:3] == ['-data_file', 'crystal_builder.data']

    def testGetDatafile(self, job):
        job.args[0] = 'lammps_runner.log'
        data_file = job.getDataFile()
        assert data_file == ['-data_file', 'crystal_builder.data']


class TestTrajJob:

    @pytest.fixture
    def job(self, tmp_dir):
        basename = os.path.join('e053136e2cd7374854430c868b3139e1',
                                'workspace',
                                '6e4cfb3bcc2a689d42d099e51b9efe23')
        job = get_job(basename=basename)
        return task.TrajJob(job, name='lmp_traj', driver=task.LmpTraj.DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args[1:3] == ['-data_file', 'amorphous_builder.data']

    def testGetTrajfile(self, job):
        job.args[0] = 'lammps_runner.log'
        traj_file = job.getTrajFile()
        assert traj_file == 'dump.custom.gz'


class TestLmpLogAgg:

    @pytest.fixture
    def agg(self, tmp_dir):
        jobs = get_jobs(basename='c1f776be48922ec50a6607f75c34c78f')
        options = types.SimpleNamespace(jobname='cb_lmp_log',
                                        task=['toteng'],
                                        interactive=False)
        return task.LmpLogAgg(*jobs,
                              logger=mock.Mock(),
                              name='lmp_log_#_agg',
                              driver=task.LmpLog.DRIVER,
                              options=options)

    def testRun(self, agg):
        assert agg.post() is False
        agg.run()
        assert agg.post() is True


class TestMolBldr:

    @pytest.fixture
    def job(self):
        job = jobutils.Job()
        job.doc[jobutils.ARGS] = ['[Ar]']
        return job

    @pytest.fixture
    def task(self):
        return task.MolBldr()

    def testPre(self, task):
        assert task.pre() is True

    def testOperator(self, task, job):
        cmd = task.operator(job, write=False)
        assert 'run_nemd mol_bldr_driver.py [Ar] -JOBNAME mol_bldr' == cmd

    def testPost(self, task, job):
        assert task.post(job) is False
        job.doc[jobutils.OUTFILE] = {task.name: 'wa.log'}
        assert task.post(job) is True

    def testGetOpr(self, task):
        opr = task.getOpr(name='mol')
        assert 'mol' == opr.__name__

    def testGetAgg(self, task, tmp_dir):
        agg = task.getAgg(get_jobs())
        assert 'mol_bldr_#_agg' == agg.__name__


class TestLmpLog:

    @pytest.fixture
    def task(self):
        return task.LmpLog()

    def testAggPost(self, task):
        job = jobutils.Job()
        assert task.aggPost(job) is False
        job.project.doc['message'] = {task.agg_name: False}
        assert task.aggPost(job) is True

    def testGetAgg(self, task, tmp_dir):
        agg = task.getAgg(get_jobs())
        assert agg._flow_cmd is True
