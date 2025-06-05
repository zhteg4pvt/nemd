import pytest

from nemd import builtinsutils


class TestObject:

    @pytest.mark.parametrize('name,expected', [('Job', 'job'),
                                               ('MolBldr', 'mol_bldr')])
    def testName(self, name, expected):
        assert expected == type(name, (builtinsutils.Object, ), {}).name


class TestDict:

    def test__setattr__(self):
        mydict = builtinsutils.Dict()
        mydict.key = 1
        assert {'key': 1} == mydict

    def testSetattr(self):
        mydict = builtinsutils.Dict()
        mydict.setattr('key', 1)
        assert 1 == mydict.key
        assert {} == mydict

    def testGetattr(self):
        mydict = builtinsutils.Dict({'key': 1})
        mydict.value = 2
        assert 1 == mydict.key
        assert 2 == mydict.value


class TestFloat:

    @pytest.mark.parametrize('name,number', [('hi', 1.2)])
    def testStr(self, name, number):
        assert 'hi: 1.20' == str(builtinsutils.Float(number, name=name))
