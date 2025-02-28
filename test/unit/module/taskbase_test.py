import os
import shutil
from unittest import mock

import pytest

from nemd import envutils
from nemd import jobutils
from nemd import parserutils
from nemd import taskbase


class TestBase:

    @pytest.fixture
    def base(self, dirname, name, tmp_dir):
        job_dir = envutils.test_data('itest', dirname)
        shutil.copytree(job_dir, os.curdir, dirs_exist_ok=True)
        return taskbase.Base(jobutils.Job(job_dir=os.curdir), name=name)

    @pytest.mark.parametrize(
        'dirname,name,num',
        [('6e4cfb3bcc2a689d42d099e51b9efe23', 'amorphous_builder', 0),
         ('f76314cf050341b16309ea080081c830', 'ab_lmp_traj', 3)])
    def testInit(self, name, base, num):
        assert num == len(base.original)
        assert name == base.jobname

    @pytest.mark.parametrize('file,expected',
                             [(None, 'name'),
                              ('mol_bldr_driver.py', 'mol_bldr')])
    def testDefaultName(self, file, expected):
        with mock.patch('nemd.taskbase.Base.FILE', file):
            base = taskbase.Base(jobutils.Job(job_dir=os.curdir))
            assert expected == base.jobname

    @pytest.mark.parametrize(
        'dirname,name', [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj')])
    def testRun(self, base):
        assert not base.post()
        base.run()
        assert base.post()

    @pytest.mark.parametrize(
        'dirname,name,expected',
        [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj', False),
         ('f76314cf050341b16309ea080081c830', 'check', True)])
    def testPost(self, base, expected):
        assert expected == base.post()

    @pytest.mark.parametrize(
        'dirname,name,expected',
        [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj', None),
         ('f76314cf050341b16309ea080081c830', 'check', False)])
    def testMessage(self, base, expected):
        assert expected == base.message
        base.message = 'hi'
        assert 'hi' == base.message

    @pytest.mark.parametrize(
        'dirname,name', [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj'),
                         ('f76314cf050341b16309ea080081c830', 'check')])
    def testClean(self, base):
        base.clean()
        assert base.message is None


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
