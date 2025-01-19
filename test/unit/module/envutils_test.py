import os
from unittest import mock

import pytest

from nemd import envutils


class TestFunction:

    @pytest.mark.parametrize("debug", ['1', '-1', '', None])
    def testIsDebug(self, debug):
        environ = {} if debug is None else {'DEBUG': debug}
        with mock.patch.dict('os.environ', environ, clear=True):
            assert debug == envutils.is_debug()

    @pytest.mark.parametrize("interactive", ['1', None])
    def testIsIteractive(self, interactive):
        environ = {} if interactive is None else {'INTERACTIVE': interactive}
        with mock.patch.dict('os.environ', environ, clear=True):
            assert interactive == envutils.is_interactive()

    @pytest.mark.parametrize("jobname,default,expected",
                             [('jobname', None, 'jobname'),
                              (None, 'default', 'default'),
                              ('jobname', 'default', 'jobname')])
    def testGetJobname(self, jobname, default, expected):
        environ = {} if jobname is None else {'JOBNAME': jobname}
        with mock.patch.dict('os.environ', environ, clear=True):
            assert expected == envutils.get_jobname(default)

    @pytest.mark.parametrize("python,expected", [('-1', '-1'), ('0', '0'),
                                                 (None, '2')])
    def testGetPythonMode(self, python, expected):
        environ = {} if python is None else {'PYTHON': python}
        with mock.patch.dict('os.environ', environ):
            assert expected == envutils.get_python_mode()

    @pytest.mark.parametrize("nemd_scr,args,expected",
                             [('/path/to/nemd', (), '/path/to/nemd'),
                              (None, ('data', 'test'), None),
                              ('/path/to/nemd', ('data', 'test'),
                               ('/path/to/nemd/data/test'))])
    def testGetNemdSrc(self, nemd_scr, args, expected):
        environ = {} if nemd_scr is None else {'NEMD_SRC': nemd_scr}
        with mock.patch.dict('os.environ', environ, clear=True):
            assert expected == envutils.get_nemd_src(*args)

    @pytest.mark.parametrize("name", [('nemd'), ('alamode')])
    def testGetModuleDir(self, name):
        if envutils.get_nemd_src() is None:
            assert name == os.path.basename(envutils.get_module_dir(name))
            return
        with mock.patch.dict('os.environ', {'NEMD_SRC': ''}, clear=True):
            from_pkgutil = envutils.get_module_dir(name)
        assert from_pkgutil == envutils.get_module_dir(name)

    @pytest.mark.parametrize("module,base,args,endswith",
                             [('nemd', None, ('ff', ), 'nemd/data/ff'),
                              ('alamode', None,
                               ('tools', ), 'alamode/alamode/tools'),
                              ('alamode', '', ('tools', ), 'alamode/tools')])
    def testGetData(self, module, base, args, endswith):
        data = envutils.get_data(*args, module=module, base=base)
        assert data.endswith(endswith)
