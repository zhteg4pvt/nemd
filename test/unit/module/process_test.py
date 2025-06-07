import os.path

import pytest

from nemd import process


@pytest.mark.parametrize('ekey,evalue', [('JOBNAME', None)])
class TestBase:

    @pytest.fixture
    def base(self, dirname, jobname, tmp_dir, env):
        return process.Base(dirname=dirname, jobname=jobname)

    @pytest.mark.parametrize('dirname,jobname,expected',
                             [(None, None, [os.curdir, 'base']),
                              ('mydir', 'myname', ['mydir', 'myname'])])
    def testRun(self, base, expected):
        assert 0 == base.run().returncode
        assert os.path.exists(f'{os.path.join(*expected)}_cmd')
        assert os.path.exists(f'{os.path.join(*expected)}.log')

    @pytest.mark.parametrize('dirname,jobname', [(os.curdir, 'myname')])
    def testGetCmd(self, base, jobname):
        assert 'echo hi' == base.getCmd()
        assert os.path.exists(f'{jobname}_cmd')

    @pytest.mark.parametrize('dirname,jobname', [(None, None)])
    def testArgs(self, base):
        assert ['echo', 'hi'] == base.args
