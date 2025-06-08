import pytest

from nemd import alamode
from nemd import envutils
from nemd import parserutils

OPTIONS = parserutils.XtalBldr().parse_args(['-JOBNAME', 'dispersion'])
STRUCT = alamode.Struct.fromMols([alamode.Crystal.fromDatabase(OPTIONS).mol],
                                 options=OPTIONS)


class TestStruct:

    def testTraj(self, tmp_dir):
        alamode.Struct().writeIn()
        with open('struct.in', 'r') as fh:
            lines = fh.read()
        assert 'id xu yu zu fx fy fz' in lines
        assert "format float '%20.15f'\n" in lines


class TestLmp:

    @pytest.mark.parametrize('struct,expected', [(STRUCT, '.custom')])
    def testExt(self, struct, expected):
        assert expected == alamode.Lmp(struct).ext


class TestFunc:
    DAT = envutils.test_data('0044', 'dispersion.data')
    PATT = envutils.test_data('0044', 'suggest', 'dispersion.pattern_HARMONIC')

    @pytest.mark.parametrize(
        'obj,kwargs,expected',
        [(alamode.Crystal.fromDatabase(OPTIONS), {}, 1), (STRUCT, {}, 1),
         ('displace', dict(files=[DAT, PATT], jobname='dispersion'), 1)])
    def testExe(self, obj, kwargs, expected, tmp_dir):
        assert expected == len(alamode.exe(obj, **kwargs))
