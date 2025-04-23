from unittest import mock

import pytest

from nemd import check


class TestExist:

    @pytest.fixture
    def exist(self, args, copied):
        exist = check.Exist(*args)
        exist.error = mock.Mock()
        return exist

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0000', ('cmd', 'wa'), 'wa not found.'),
                              ('0000', ('cmd', 'check'), None)])
    def testRun(self, exist, expected, tmp_dir):
        exist.run()
        exist.error.assert_has_calls([mock.call(expected)] if expected else [])


class TestGlob:

    @pytest.fixture
    def glob(self, args, kwargs, copied):
        glob = check.Glob(*args, **kwargs)
        glob.error = mock.Mock()
        return glob

    @pytest.mark.parametrize('dirname,args,kwargs,expected',
                             [(None, ('cmd*', ), {}, "0 files found. (None)"),
                              ('0000', ('cmd*', ), {}, None),
                              ('0000',
                               ('cmd*', ), dict(num=1), "2 files found. (1)"),
                              ('0000', ('cmd*', ), dict(num=2), None),
                              ('0000', ('cmd*', 'check'), dict(num=3), None)])
    def testRun(self, glob, expected, tmp_dir):
        glob.run()
        glob.error.assert_has_calls([mock.call(expected)] if expected else [])


class TestHas:

    @pytest.fixture
    def has(self, args, copied):
        has = check.Has(*args)
        has.error = mock.Mock()
        return has

    @pytest.mark.parametrize(
        'dirname,args,expected',
        [(None, ('cmd', ), "cmd not found."), ('0000', ('cmd', 'hi'), None),
         ('0000', ('cmd', 'hi', 'wa'), None),
         ('0000', ('cmd', 'hi', 'wa', 'hello'), "hello not found in cmd.")])
    def testRun(self, has, expected, tmp_dir):
        try:
            has.run()
        except FileNotFoundError:
            pass
        has.error.assert_has_calls([mock.call(expected)] if expected else [])
