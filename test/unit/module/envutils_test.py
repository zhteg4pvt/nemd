import os
from unittest import mock

import pytest

from nemd import envutils


class TestFunc:

    @pytest.mark.parametrize("ekey", ['INTERACTIVE'])
    @pytest.mark.parametrize("evalue", ['1', None])
    def testIsInteractive(self, evalue, env):
        assert evalue == envutils.is_interactive()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [('-1', '-1'), ('0', '0'),
                                                 (None, '2')])
    def testGetPythonMode(self, expected, env):
        assert expected == envutils.get_python_mode()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [('-1', False), ('0', True),
                                                 (None, False)])
    def testIsOriginal(self, expected, env):
        assert expected == envutils.is_original()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [('-1', False), ('0', False),
                                                 ('1', True), ('2', True)])
    def testIsNopython(self, expected, env):
        assert expected == envutils.is_nopython()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,nopython,cache", [('-1', False, False),
                                                       ('0', False, False),
                                                       ('1', True, False),
                                                       ('2', True, True)])
    def testGetJitKwargs(self, nopython, cache, env):
        kwargs = envutils.get_jit_kwargs()
        assert nopython == kwargs['nopython']
        assert cache == kwargs['cache']

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("evalue", [('myname'), (None)])
    def testSetJobnameDefault(self, env):
        pre = os.environ.get('JOBNAME', None)
        envutils.set_jobname_default('new')
        assert (pre if pre else 'new') == envutils.get_jobname()

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("evalue,default,expected",
                             [('myname', None, 'myname'),
                              (None, 'default', 'default'),
                              ('myname', 'default', 'myname')])
    def testGetJobname(self, default, expected, env):
        assert expected == envutils.get_jobname(default)

    @pytest.mark.parametrize("ekey", ['MEM_INTVL'])
    @pytest.mark.parametrize("evalue,expected", [('-1.1', None), ('4.2', 4.2),
                                                 (None, None)])
    def testGetMemIntvl(self, expected, env):
        assert expected == envutils.get_mem_intvl()

    @pytest.mark.parametrize("ekey", ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,args,expected",
                             [('/path/to/nemd', (), '/path/to/nemd'),
                              (None, ('data', 'test'), None),
                              ('/path/to/nemd', ('data', 'test'),
                               ('/path/to/nemd/data/test'))])
    def testGetNemdSrc(self, args, expected, env):
        assert expected == envutils.get_nemd_src(*args)

    @pytest.mark.parametrize("ekey", ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,name", [(None, 'nemd'),
                                             ('fake/path', 'nemd'),
                                             (None, 'alamode'),
                                             ('fake/path', 'alamode')])
    def testGetModuleDir(self, name, evalue, env):
        module_dir = envutils.get_module_dir(name)
        assert module_dir.endswith(name)
        if not evalue:
            return
        assert module_dir.startswith(evalue)

    @pytest.mark.parametrize("args", [('dirname', 'filename')])
    def testGetModuleDir(self, args):
        assert envutils.test_data(*args).endswith(os.path.join(*args))

    @pytest.mark.parametrize("module,base,args,endswith",
                             [('nemd', None, ('ff', ), 'nemd/data/ff'),
                              ('alamode', None,
                               ('tools', ), 'alamode/alamode/tools'),
                              ('alamode', '', ('tools', ), 'alamode/tools')])
    def testGetData(self, module, base, args, endswith):
        data = envutils.get_data(*args, module=module, base=base)
        assert data.endswith(endswith)
