import os
import shutil
from unittest import mock

import pytest

from nemd import envutils
from nemd import jobutils
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
