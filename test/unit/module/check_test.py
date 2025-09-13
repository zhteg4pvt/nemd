import os
from unittest import mock

import conftest
import pytest

from nemd import check


@conftest.require_src
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


@conftest.require_src
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


@conftest.require_src
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


@conftest.require_src
@pytest.mark.parametrize('dirname', ['0000'])
class TestCmp:

    @pytest.fixture
    def cmp(self, args, kwargs, called, copied):
        cmp = check.Cmp(*args, **kwargs)
        cmp.errorDiff = called
        return cmp

    @pytest.mark.parametrize('kwargs', [{}])
    @pytest.mark.parametrize('args,expected',
                             [(['cmd', 'check'], 'check'),
                              (['original.csv', 'close.csv'], 'close.csv'),
                              (['original.data', 'close.data'], 'close.data')])
    def testRun(self, cmp, expected):
        cmp.run()

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
        cmp.csv()

    @pytest.mark.parametrize(
        'args,target,expected',
        [(['cmd', 'check'], 'check', 'cmd is different from check.')])
    def testErrorDiff(self, args, target, called, dirname):
        cmp = check.Cmp(*args)
        cmp.error = called
        cmp.errorDiff(target)

    @pytest.mark.parametrize(
        'args,kwargs,expected',
        [(['original.csv'], {}, (1, 3)),
         (['original.csv'], dict(selected='Clash (count) (num=2)'), (1, 1))])
    def testReadCsv(self, args, kwargs, expected, copied):
        cmp = check.Cmp(*args, **kwargs)
        _, non = cmp.readCsv(cmp.args[0])
        assert expected == non.shape

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
        cmp.data()


@conftest.require_src
class TestCollect:
    TEST0049 = os.path.join('0049_test', 'workspace',
                            '3ec5394f589c9363bd15af35d45a7c44')

    @pytest.fixture
    def collect(self, args, kwargs, error, copied):
        collect = check.Collect(*args, **kwargs)
        collect.error = error
        return collect

    @pytest.mark.parametrize('dirname,args,kwargs,expected',
                             [('empty', [], {}, ['task_time'])])
    def testInit(self, collect, expected):
        assert expected == collect.args

    @pytest.mark.parametrize('kwargs', [{}])
    @pytest.mark.parametrize('dirname,args,expected',
                             [('0000', ['memory', 'finished'], SystemExit),
                              ('0049_ubuntu', ['task_time'], (2, 1))])
    def testRun(self, collect, expected, raises):
        with raises:
            collect.run()
            assert expected == collect.data.shape
            assert os.path.isfile('collect.csv')

    @pytest.mark.parametrize(
        'dirname,args,kwargs,expected',
        [('0049', ['task_time'], {}, SystemExit),
         ('0049', ['task_time'], dict(dropna='False'), SystemExit),
         ('0049_ubuntu', ['task_time', 'memory'], {}, (2, 2)),
         ('0049_ubuntu', ['task_time', 'memory', 'finished'], {}, (2, 3)),
         ('0000', ['task_time', 'memory', 'finished'], {}, SystemExit),
         ('0001_fail', ['task_time', 'finished'], {}, (1, 1)),
         ('0001_fail', ['finished'], dict(dropna='False'), SystemExit)])
    def testSet(self, collect, expected, raises):
        with raises:
            collect.set()
            assert expected == collect.data.shape
            assert os.path.isfile('collect.csv')

    @pytest.mark.parametrize('kwargs', [{}])
    @pytest.mark.parametrize(
        'dirname,args,expected,outfiles',
        [(TEST0049, ['task_time'], [1], ['collect_task.svg']),
         (TEST0049, ['task_time', 'memory'], [1], ['collect_task.svg']),
         ('0049_ubuntu', ['task_time', 'memory'], [2], ['collect.svg']),
         ('0049_ubuntu', ['task_time', 'finished'], [2], ['collect.svg']),
         ('0049_ubuntu', ['task_time', 'memory', 'finished'], [1, 1, 1],
          ['collect_task.svg', 'collect_memory.svg', 'collect_finished.svg']),
         ('0049_ubuntu', ['memory'], [1], ['collect_memory.svg'])])
    def testPlot(self, collect, expected, outfiles):
        collect.set()
        collect.plot()
        assert expected == [len(x.axes) for x in collect.figs]
        for outfile in outfiles:
            assert os.path.exists(outfile)


class TestMerge:

    @pytest.fixture
    def merge(self, args, kwargs, error, copied):
        merge = check.Merge(*args, **kwargs)
        merge.error = error
        return merge

    @pytest.mark.parametrize('dirname,args,kwargs,expected',
                             [('empty', [], {}, SystemExit),
                              ('p0001', [], {}, (11, 3))])
    def testSet(self, merge, expected, raises):
        with raises:
            merge.set()
            assert expected == merge.data.shape

    @pytest.mark.parametrize(
        'dirname,args,kwargs,expected',
        [('p0001', [], {}, ['Task Time (min)', 'Memory (MB)'])])
    def testCols(self, merge, expected):
        merge.set()
        assert expected == merge.cols

    @pytest.mark.parametrize('dirname,args,kwargs', [('p0001', [], {})])
    @pytest.mark.parametrize('col,expected', [('Task Time (min)', 2),
                                              ('Memory (MB)', 1)])
    def testAxPlot(self, merge, col, expected, ax):
        merge.set()
        merge.axPlot(ax, col)
        assert expected == len(ax.lines)

    @pytest.mark.parametrize(
        'dirname,args,kwargs,col,expected',
        [('p0001', [], {}, 'Task Time (min)', 'merge.svg')])
    def testSave(self, merge, col, expected, fig, tmp_dir):
        merge.set()
        ax = fig.add_subplot(1, 1, 1)
        merge.axPlot(ax, col)
        merge.save(fig)
        assert os.path.exists(expected)


class TestMain:

    @pytest.mark.parametrize('args,expected', [(['cmp'], check.Cmp),
                                               (['cmp2'], None)])
    def testInit(self, args, expected, tmp_dir):
        assert expected == check.Main(args).Class

    @conftest.require_src
    @pytest.mark.parametrize('dirname,args,expected,msg', [
        ('0000', ('exist', 'wa'), SystemExit, 'wa not found.'),
        ('0000', ('exist2', 'cmd'), SystemExit,
         'exist2 found. Please select from exist, glob, has, cmp, collect, merge.'
         ), ('0000', ('exist', 'cmd'), None, None)
    ])
    def testRun(self, args, expected, copied, msg, raises):
        with mock.patch('nemd.logutils.Base.log') as mocked:
            main = check.Main(args)
            with raises:
                main.run()
            if msg is not None:
                mocked.assert_called_once_with(msg)
