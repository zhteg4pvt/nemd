import os.path
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
    def testRun(self, exist, expected):
        exist.run()
        exist.error.assert_called_with(
            expected) if expected else exist.error.assert_not_called()


@pytest.mark.parametrize('dirname', ['0000'])
class TestGlob:

    @pytest.fixture
    def glob(self, args, kwargs, copied):
        glob = check.Glob(*args, **kwargs)
        glob.error = mock.Mock()
        return glob

    @pytest.mark.parametrize('args,kwargs,expected',
                             [(('cmd4*', ), {}, "0 files found. (None)"),
                              (('cmd*', ), {}, None),
                              (('cmd*', ), dict(num=1), "2 files found. (1)"),
                              (('cmd*', ), dict(num=2), None),
                              (('cmd*', 'check'), dict(num=3), None)])
    def testRun(self, glob, expected):
        glob.run()
        glob.error.assert_called_with(
            expected) if expected else glob.error.assert_not_called()


@pytest.mark.parametrize('dirname', ['0000'])
class TestHas:

    @pytest.fixture
    def has(self, args, copied):
        has = check.Has(*args)
        has.error = mock.Mock()
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
        has.error.assert_called_with(
            expected) if expected else has.error.assert_not_called()


@pytest.mark.parametrize('dirname', ['0000'])
class TestCmp:

    @pytest.fixture
    def cmp(self, args, kwargs, copied):
        cmp = check.Cmp(*args, **kwargs)
        cmp.error = mock.Mock()
        return cmp

    @pytest.mark.parametrize(
        'args,kwargs,expected',
        [(['cmd', 'cmd_same'], {}, None), (['cmd', 'check'], {}, 'check'),
         (['cmd', 'cmd_same', 'check'], {}, 'check'),
         (['cmd', 'check'], dict(equal_nan='True'), None)])
    def testFile(self, cmp, expected):
        cmp.file()
        cmp.error.assert_called_with(
            expected) if expected else cmp.error.assert_not_called()

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
        cmp.error.assert_called_with(
            expected) if expected else cmp.error.assert_not_called()

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
        cmp.error.assert_called_with(
            expected) if expected else cmp.error.assert_not_called()


class TestCollect:

    @pytest.fixture
    def collect(self, args, copied):
        collect = check.Collect(*args)
        collect.error = mock.Mock()
        return collect

    @pytest.mark.parametrize(
        'dirname,args,expected',
        [
            #    ('0049', ['task_time'], None),
            # ('0049_test', ['task_time'], (2, 1)),
            ('0049_test', ['task_time', 'memory'], (2, 1)),
            # ('0049_ubuntu', ['task_time', 'memory'], (2, 2)),
            # ('0049_ubuntu', ['memory'], (2, 1))
        ])
    def testSet(self, collect, expected):
        collect.set()
        assert expected == (collect.data.shape if expected else collect.data)
        assert bool(expected) == os.path.isfile('collect.csv')

    @pytest.mark.parametrize('dirname,args,expected',
                             [('0049', ['task_time'], 0),
                              ('0049_test', ['task_time'], 1),
                              ('0049_test', ['task_time', 'memory'], 1),
                              ('0049_ubuntu', ['task_time', 'memory'], 2),
                              ('0049_ubuntu', ['memory'], 1)])
    def testPlot(self, collect, expected):
        collect.set()
        collect.plot()
        assert expected == (len(collect.fig.axes) if collect.fig else 0)
        assert bool(expected) == os.path.exists('collect.png')
