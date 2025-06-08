import os.path

import pytest

from nemd import alamode
from nemd import envutils
from nemd import osutils
from nemd import parserutils
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


class TestProcess:

    @pytest.mark.parametrize('tokens', [['echo', 'hello']])
    def testArgs(self, tokens):
        assert tokens == process.Process(tokens).args


class TestCheck:

    @pytest.mark.parametrize('tokens,expected',
                             [(['echo', 'hello'], 'echo;hello')])
    def testGetCmd(self, tokens, expected):
        assert expected == process.Check(tokens).getCmd(write_cmd=False)


@pytest.mark.parametrize('ekey,evalue', [('JOBNAME', None)])
class TestSubmodule:

    @pytest.fixture
    def submodule(self, mode, files, env, tmp_dir):
        return process.Submodule(mode, files=files)

    @pytest.mark.parametrize('mode,files,expected',
                             [('suggest', None, 'nemd_module echo hi')])
    def testGetCmd(self, submodule, expected):
        assert expected == submodule.getCmd(write_cmd=False)

    @pytest.mark.parametrize('mode,files,expected',
                             [('suggest', None, FileNotFoundError)])
    def testOutfiles(self, submodule, expected, raises):
        with raises:
            submodule.outfiles
        with osutils.chdir('suggest'):
            with open('submodule.log', 'w'):
                pass
        assert submodule.outfiles

    @pytest.mark.parametrize('mode,files,expected',
                             [('suggest', ['dispersion.data'], 'suggest')])
    def testFiles(self, submodule, files, expected):
        for file in files:
            with open(file, 'w'):
                pass
        assert all([os.path.exists(x) for x in submodule.files])
        with osutils.chdir('new_dir'):
            assert all([os.path.exists(x) for x in submodule.files])


@pytest.mark.parametrize('jobname,files', [('dispersion', [
    envutils.test_data('0044', 'displace', 'dispersion1.lammps')
])])
class TestLmp:

    @pytest.fixture
    def lmp(self, jobname, files, tmp_dir):
        options = parserutils.XtalBldr().parse_args(['-JOBNAME', 'dispersion'])
        mols = [alamode.Crystal.fromDatabase(options).mol]
        struct = alamode.Struct.fromMols(mols, options=options)
        return process.Lmp(struct, jobname=jobname, files=files)

    def testSetUp(self, lmp, tmp_dir):
        lmp.setUp()
        assert os.path.isfile(lmp.struct.datafile)
        assert os.path.isfile(lmp.struct.inscript)

    def testArg(self, lmp, tmp_dir):
        lmp.setUp()
        assert 7 == len(lmp.args)
