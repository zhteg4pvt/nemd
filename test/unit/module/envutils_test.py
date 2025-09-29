import os

import conftest
import pytest

from nemd import envutils
from nemd import process


class TestEnv:

    @pytest.mark.parametrize("ekey", ['INTERAC'])
    @pytest.mark.parametrize("evalue,expected", [(None, False), ('', False),
                                                 ('1', True)])
    def testInterac(self, expected, env):
        assert expected == envutils.Env().interac

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("evalue,expected", [('myname', 'myname'),
                                                 (None, None)])
    def testJobname(self, expected, env):
        assert expected == envutils.Env().jobname

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [('-1', -1), (None, 2)])
    def testMode(self, expected, env):
        assert expected == envutils.Env().mode

    @pytest.mark.parametrize("ekey", ['INTVL'])
    @pytest.mark.parametrize("evalue,expected", [(None, None), ('-1.1', None),
                                                 ('4.2', 4.2)])
    def testIntvl(self, expected, env):
        assert expected == envutils.Env().intvl

    @pytest.mark.parametrize("ekey", ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,expected", [(None, ''), ('', ''),
                                                 ('my_path', 'my_path')])
    def testSrc(self, expected, env):
        assert expected == envutils.Src()


class TestMode:

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, 2), ('0', 0)])
    def testInit(self, expected, env):
        assert expected == envutils.Mode()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, False), ('-1', False),
                                                 ('0', True)])
    def testOrig(self, expected, env):
        assert expected == envutils.Mode().orig

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, [True, True]),
                                                 ('-1', [False, False]),
                                                 ('0', [False, False]),
                                                 ('1', [True, False]),
                                                 ('2', [True, True])])
    def testKwargs(self, expected, env):
        kwargs = envutils.Mode().kwargs
        assert expected == list(kwargs.values())

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, True), ('-1', False),
                                                 ('0', False), ('1', True),
                                                 ('2', True)])
    def testNo(self, expected, env):
        assert expected == envutils.Mode().no


class TestSrc:
    SRC = envutils.Src()

    @conftest.require_src
    @pytest.mark.parametrize('ekey', ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,args,expected",
                             [(SRC, ('water', 'xyzl.data'), 'data'),
                              (SRC, ('water', 'defm_39'), 'data'),
                              ('', ('water', 'defm_39'), ''),
                              (None, ('water', 'defm_39'), '')])
    def testTest(self, args, expected, env):
        data = envutils.Src().test(*args)
        assert data.endswith(os.path.join(expected, *args))

    @pytest.mark.parametrize("args,module,base,expected",
                             [(('ff', ), 'nemd', None, True),
                              (('envutils.py', ), 'nemd', '', True),
                              (('tools', ), 'alamode', None, True),
                              (('tools', ), 'alamode', 'alamode', True)])
    def testGet(self, module, base, args, expected, tmp_dir, quot="'{}'"):
        args = list(map(quot.format, args))
        args.append(f"module={quot.format(module)}")
        args.append(f"base={base if base is None else quot.format(base)}")
        path = f"envutils.Src().get({', '.join(args)})"
        cmd = f'"from nemd import envutils; print({path})"'
        proc = process.Process(['nemd_run', '-c', cmd])
        proc.run()
        assert expected == os.path.exists(proc.msg.strip())
