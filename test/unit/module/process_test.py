import os

import conftest
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
        base.run()
        assert 0 == base.proc.returncode
        assert os.path.exists(f'{os.path.join(*expected)}_cmd')
        assert os.path.exists(f'{os.path.join(*expected)}.std')

    @pytest.mark.parametrize('dirname,jobname', [(os.curdir, 'myname')])
    def testGetCmd(self, base, jobname):
        assert 'echo hi' == base.getCmd()
        assert os.path.exists(f'{jobname}_cmd')

    @pytest.mark.parametrize('dirname,jobname', [(None, None)])
    def testArgs(self, base):
        assert ['echo', 'hi'] == base.args

    @pytest.mark.parametrize('dirname,jobname,expected',
                             [(None, None, 'hi\n'), ('mydir', 'myname', '')])
    def testMsg(self, base, expected, tmp_dir):
        base.run()
        assert expected == base.msg


class TestProcess:

    @pytest.mark.parametrize('tokens', [['echo', 'hello']])
    def testArgs(self, tokens):
        assert tokens == process.Process(tokens).args

    @pytest.mark.parametrize('tokens,expected',
                             [(['echo', 'hello'], ''),
                              (['echo', 'hello', '>&2'], 'hello\n'),
                              (['exit', '1'], 'non-zero return code')])
    def testErr(self, tokens, expected, tmp_dir):
        prc = process.Process(tokens)
        prc.run()
        assert expected == prc.err


class TestCheck:

    @pytest.mark.parametrize('tokens,expected',
                             [(['echo', 'hello'], 'echo && hello')])
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

    @pytest.mark.parametrize('files', [None])
    @pytest.mark.parametrize('mode,file,expected',
                             [('suggest', 'wa', FileNotFoundError),
                              ('suggest', 'submodule.std', 1)])
    def testOutfiles(self, mode, file, submodule, expected, raises):
        with osutils.chdir(mode), open(file, 'w'):
            pass
        with raises:
            assert expected == len(submodule.outfiles)

    @pytest.mark.parametrize('mode,files,expected',
                             [('suggest', ['dispersion.data'], 'suggest')])
    def testFiles(self, submodule, files, expected):
        for file in files:
            with open(file, 'w'):
                pass
        assert all([os.path.exists(x) for x in submodule.files])
        with osutils.chdir('new_dir'):
            assert all([os.path.exists(x) for x in submodule.files])


class TestLmp:

    @pytest.fixture
    def lmp(self, infile):
        return process.Lmp(infile=infile)

    @pytest.mark.parametrize('infile', [None])
    def testArg(self, lmp):
        assert 4 == len(lmp.args)

    @conftest.require_src
    @pytest.mark.parametrize('infile,expected', [(envutils.Src().test(
        'ar', 'error.in'), 'cannot open file ar100.data'),
                                                 ('not_exist', 'errorcode')])
    def testErr(self, lmp, expected, tmp_dir):
        lmp.run()
        assert expected in lmp.err.lower()


class TestSubmodules:

    @pytest.mark.parametrize('mode,expected', [('mode', '.xyz')])
    def testExt(self, mode, expected):
        assert '.std' == process.Submodules(mode).ext
        Sub = type('', (process.Submodules, ), dict(EXTS={mode: expected}))
        assert expected == Sub(mode).ext


@conftest.require_src
class TestTools:
    DATA = envutils.Src().test('0044', 'dispersion.data')
    PATTERN = envutils.Src().test('0044', 'suggest',
                                  'dispersion.pattern_HARMONIC')
    CUSTOM = envutils.Src().test('0044', 'lammps1', 'dispersion.custom')

    @pytest.mark.parametrize('files,mode,expected',
                             [([DATA, PATTERN], 'displace', 9),
                              ([DATA, CUSTOM], 'extract', 4)])
    def testArgs(self, files, mode, expected):
        tools = process.Tools(files=files, mode=mode)
        assert expected == len(tools.args)


@pytest.mark.parametrize('jobname', ['dispersion'])
class TestAlamode:

    @pytest.fixture
    def ala(self, jobname, mode, file, tmp_dir):
        options = parserutils.XtalBldr().parse_args(['-JOBNAME', 'dispersion'])
        crystal = alamode.Crystal.fromDatabase(options, mode=mode)
        return process.Alamode(crystal, jobname=jobname, files=[file])

    @conftest.require_src
    @pytest.mark.parametrize(
        'file,mode,expected',
        [(None, 'suggest', ['dispersion.in']),
         (envutils.Src().test('0044', 'extract', 'dispersion.log'), 'optimize',
          ['dispersion.in', 'dispersion.dfset'])])
    def testSetUp(self, ala, expected, tmp_dir):
        ala.setUp()
        for file in expected:
            assert os.path.isfile(file)

    @pytest.mark.parametrize('file,mode,expected', [(None, 'suggest', 2)])
    def testArg(self, ala, expected):
        assert expected == len(ala.args)
