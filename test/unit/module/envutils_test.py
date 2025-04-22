import os

import pytest

from nemd import envutils


class TestFunc:
    NEMD_SRC = envutils.get_src()

    @pytest.mark.parametrize("ekey", ['INTERAC'])
    @pytest.mark.parametrize("evalue,expected", [(None, False), ('', False),
                                                 ('1', True)])
    def testIsInterac(self, expected, env):
        assert expected == envutils.is_interac()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, '2'), ('0', '0')])
    def testGetPythonMode(self, expected, env):
        assert expected == envutils.get_python_mode()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, False), ('-1', False),
                                                 ('0', True)])
    def testIsOriginal(self, expected, env):
        assert expected == envutils.is_original()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, True), ('-1', False),
                                                 ('0', False), ('1', True),
                                                 ('2', True)])
    def testIsNopython(self, expected, env):
        assert expected == envutils.nopython()

    @pytest.mark.parametrize("ekey", ['PYTHON'])
    @pytest.mark.parametrize("evalue,expected", [(None, [True, True]),
                                                 ('-1', [False, False]),
                                                 ('0', [False, False]),
                                                 ('1', [True, False]),
                                                 ('2', [True, True])])
    def testGetJitKwargs(self, expected, env):
        kwargs = envutils.jit_kwargs()
        assert expected == list(kwargs.values())

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("evalue,expected", [('myname', 'myname'),
                                                 (None, None)])
    def testGetJobname(self, expected, env):
        assert expected == envutils.get_jobname()

    @pytest.mark.parametrize("ekey", ['MEM_INTVL'])
    @pytest.mark.parametrize("evalue,expected", [(None, None), ('-1.1', None),
                                                 ('4.2', 4.2)])
    def testGetMemIntvl(self, expected, env):
        assert expected == envutils.get_mem_intvl()

    @pytest.mark.skipif(NEMD_SRC is None, reason="NEMD_SRC not found")
    @pytest.mark.parametrize("ekey", ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,args,expected",
                             [(NEMD_SRC, (), NEMD_SRC),
                              (None, ('data', 'test'), None),
                              (NEMD_SRC, ('test', ), 'test')])
    def testGetSrc(self, args, evalue, expected, env):
        if all([evalue, expected]):
            expected = os.path.join(evalue, expected)
        assert expected == envutils.get_src(*args)

    @pytest.mark.skipif(NEMD_SRC is None, reason="NEMD_SRC not found")
    @pytest.mark.parametrize('ekey', ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,args,expected",
                             [(NEMD_SRC, ('water', 'xyzl.data'), 'data'),
                              (NEMD_SRC, ('water', 'defm_39'), 'data')])
    def testTestData(self, args, expected, env):
        data = envutils.test_data(*args)
        assert data.endswith(os.path.join(expected, *args))

    @pytest.mark.parametrize("ekey", ['NEMD_SRC'])
    @pytest.mark.parametrize("evalue,name,expected",
                             [(None, 'nemd', 'nemd'),
                              (NEMD_SRC, 'nemd', 'nemd'),
                              (None, 'alamode', 'alamode'),
                              (NEMD_SRC, 'alamode', 'alamode')])
    def testGetModuleDir(self, name, expected, env):
        assert envutils.get_module_dir(name).endswith(expected)

    @pytest.mark.parametrize(
        "args,module,base,expected",
        [(('ff', ), 'nemd', None, ['data', 'ff']),
         (('tools', ), 'alamode', None, ['alamode', 'tools']),
         (('tools', ), 'alamode', '', ['tools'])])
    def testGetData(self, module, base, args, expected):
        data = envutils.get_data(*args, module=module, base=base)
        assert data.endswith(os.path.join(module, *expected))
