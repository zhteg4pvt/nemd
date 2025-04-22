import pytest

from nemd import objectutils


class TestObject:

    @pytest.mark.parametrize('name,expected', [('Job', 'job'),
                                               ('MolBldr', 'mol_bldr')])
    def testName(self, name, expected):
        assert expected == type(name, (objectutils.Object, ), {}).name
