import os
import shutil
from unittest import mock

import pytest

from nemd import envutils
from nemd import jobutils
from nemd import taskbase


class TestBase:

    @pytest.fixture
    def base(self, basename, jobname, tmp_dir):
        job_dir = envutils.test_data('itest', basename)
        shutil.copytree(job_dir, os.curdir, dirs_exist_ok=True)
        return taskbase.Base(jobutils.Job(job_dir=os.curdir), jobname=jobname)

    @pytest.mark.parametrize('file,expected',
                             [(None, 'name'),
                              ('mol_bldr_driver.py', 'mol_bldr')])
    def testName(self, file, expected):
        with mock.patch('nemd.taskbase.Base.FILE', file):
            base = taskbase.Base(jobutils.Job(job_dir=os.curdir))
            assert expected == base.name

    @pytest.mark.parametrize(
        'basename,jobname,expected',
        [('6e4cfb3bcc2a689d42d099e51b9efe23', 'amorphous_builder', 0),
         ('f76314cf050341b16309ea080081c830', 'ab_lmp_traj', 3)])
    def testArgs(self, base, expected):
        assert expected == len(base.args)

    @pytest.mark.parametrize(
        'basename,jobname,expected',
        [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj', False),
         ('f76314cf050341b16309ea080081c830', 'check', True)])
    def testPost(self, base, expected):
        assert expected == base.post()

    @pytest.mark.parametrize(
        'basename,jobname,expected',
        [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj', None),
         ('f76314cf050341b16309ea080081c830', 'check', False)])
    def testMessage(self, base, expected):
        assert expected == base.message
        base.message = 'hi'
        assert 'hi' == base.message

    @pytest.mark.parametrize(
        'basename,jobname,expected',
        [('0923fc12bdc6d40132eb65ed96588f52', 'ab_lmp_traj', None),
         ('f76314cf050341b16309ea080081c830', 'check', False)])
    def testClean(self, base, expected):
        base.clean()
        assert base.message is None
