from nemd import dictutils


class TestDict:

    def test__setattr__(self):
        mydict = dictutils.Dict()
        mydict.key = 1
        assert {'key': 1} == mydict

    def testSetattr(self):
        mydict = dictutils.Dict()
        mydict.setattr('key', 1)
        assert 1 == mydict.key
        assert {} == mydict

    def testGetattr(self):
        mydict = dictutils.Dict({'key': 1})
        mydict.value = 2
        assert 1 == mydict.key
        assert 2 == mydict.value
