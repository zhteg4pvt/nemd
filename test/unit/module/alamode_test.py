import pytest

from nemd import alamode
from nemd import parserutils


class TestStruct:

    def testTraj(self, tmp_dir):
        alamode.Struct().writeIn()
        with open('struct.in', 'r') as fh:
            lines = fh.read()
        assert 'id xu yu zu fx fy fz' in lines
        assert "format float '%20.15f'\n" in lines


class TestFunc:

    OPTIONS = parserutils.XtalBldr().parse_args(['-JOBNAME', 'dispersion'])
    STRUCT = alamode.Struct.fromMols(
        [alamode.Crystal.fromDatabase(OPTIONS).mol], options=OPTIONS)

    @pytest.mark.parametrize('obj,kwargs,expected',
                             [(alamode.Crystal.fromDatabase(OPTIONS), {}, 1),
                              (STRUCT, {}, 1)])
    def testExe(self, obj, kwargs, expected, tmp_dir):
        assert expected == len(alamode.exe(obj, **kwargs))
