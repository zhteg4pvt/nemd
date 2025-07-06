import os.path
from unittest import mock

import pytest

from nemd import check


class TestExist:

    @pytest.fixture
    def exist(self, args, called, copied):
        exist = check.Exist(*args)
        exist.error = called
        return exist

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0000', ('cmd', 'wa'), 'wa not found.'),
                              ('0000', ('cmd', 'check'), None)])
    def testRun(self, exist, expected):
        exist.run()


@pytest.mark.parametrize('dirname', ['0000'])
class TestGlob:

    @pytest.fixture
    def glob(self, args, kwargs, called, copied):
        glob = check.Glob(*args, **kwargs)
        glob.error = called
        return glob

    @pytest.mark.parametrize('args,kwargs,expected',
                             [(('cmd4*', ), {}, "0 files found. (None)"),
                              (('cmd*', ), {}, None),
                              (('cmd*', ), dict(num=1), "2 files found. (1)"),
                              (('cmd*', ), dict(num=2), None),
                              (('cmd*', 'check'), dict(num=3), None)])
    def testRun(self, glob, expected):
        glob.run()


@pytest.mark.parametrize('dirname', ['0000'])
class TestHas:

    @pytest.fixture
    def has(self, args, called, copied):
        has = check.Has(*args)
        has.error = called
        return has

    @pytest.mark.parametrize(
        'args,expected',
        [(('cmd3', ), "cmd3 not found."), (('cmd', 'hi'), None),
         (('cmd', 'hi', 'wa'), None),
         (('cmd', 'hi', 'wa', 'hello'), "hello not found in cmd.")])
    def testRun(self, has, expected, tmp_dir):
        try:
            has.run()
        except FileNotFoundError:
            pass


@pytest.mark.parametrize('dirname', ['0000'])
class TestCmp:

    @pytest.fixture
    def cmp(self, args, kwargs, called, copied):
        cmp = check.Cmp(*args, **kwargs)
        cmp.errorDiff = called
        return cmp

    @pytest.mark.parametrize(
        'args,kwargs,expected',
        [(['cmd', 'cmd_same'], {}, None), (['cmd', 'check'], {}, 'check'),
         (['cmd', 'cmd_same', 'check'], {}, 'check'),
         (['cmd', 'check'], dict(equal_nan='True'), None)])
    def testFile(self, cmp, expected):
        cmp.file()

    @pytest.mark.parametrize(
        'args,kwargs,expected',
        [(['original.csv', 'same.csv'], dict(atol='1e-08'), None),
         (['original.csv', 'close.csv'], dict(atol='1e-08'), 'close.csv'),
         (['original.csv', 'close.csv'], dict(atol='1e-06'), None),
         (['original.csv', 'same.csv', 'close.csv'
           ], dict(atol='1e-08'), 'close.csv'),
         (['original.csv', 'different.csv'
           ], dict(atol='1e-06'), 'different.csv')])
    def testCsv(self, cmp, expected):
        try:
            cmp.csv()
        except ValueError:
            pass

    @pytest.mark.parametrize(
        'args,kwargs,expected',
        [(['original.data', 'same.data'], dict(atol='1e-08'), None),
         (['original.data', 'close.data'], dict(atol='1e-08'), 'close.data'),
         (['original.data', 'close.data'], dict(atol='1e-03'), None),
         (['original.data', 'same.data', 'close.data'
           ], dict(atol='1e-08'), 'close.data'),
         (['original.data', 'different.data'
           ], dict(atol='1e-03'), 'different.data')])
    def testData(self, cmp, expected):
        try:
            cmp.data()
        except ValueError:
            pass


class TestCollect:
    TEST0049 = os.path.join('0049_test', 'workspace',
                            '3ec5394f589c9363bd15af35d45a7c44')

    @pytest.fixture
    def collect(self, args, copied):
        collect = check.Collect(*args)
        collect.error = mock.Mock()
        return collect

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0049', ['task_time'], (0, 1)),
                              ('0049_ubuntu', ['task_time', 'memory'], (2, 2))])
    def testSet(self, collect, expected):
        collect.set()
        assert expected == collect.data.shape
        assert bool(expected[0]) == os.path.isfile('collect.csv')

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0049', ['task_time'], 0),
                              (TEST0049, ['task_time'], 1),
                              (TEST0049, ['task_time', 'memory'], 1),
                              ('0049_ubuntu', ['task_time', 'memory'], 2),
                              ('0049_ubuntu', ['memory'], 1)])
    def testPlot(self, collect, expected):
        collect.set()
        collect.plot()
        assert expected == (len(collect.fig.axes) if collect.fig else 0)
        assert bool(expected) == os.path.exists('collect.png')
