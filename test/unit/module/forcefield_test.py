import pytest

from nemd import forcefield
from nemd import oplsua


class TestFunc:

    @pytest.mark.parametrize('name,args,expected',
                             [('SW', ('Si', ), str),
                              ('OPLSUA', 'SPC', oplsua.Parser)])
    def testGet(self, name, args, expected):
        assert isinstance(forcefield.get(name, args), expected)
