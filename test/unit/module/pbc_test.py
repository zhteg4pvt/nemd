import math

import numpy as np
import pandas as pd
import pytest

from nemd import numpyutils
from nemd import pbc


class TestBase:

    @pytest.fixture
    def base(self, TYPE_COL, ID_COLS, FMT):
        attrs = dict(COLUMNS=['a', 'b', 'c'],
                     TYPE_COL=TYPE_COL,
                     ID_COLS=ID_COLS,
                     FMT=FMT)
        Base = type('Raw', (pbc.Base, ), attrs)
        return Base(np.zeros((3, 3), dtype=int))

    @pytest.mark.parametrize('data,shape,columns',
                             [(2, (2, 1), ['label']),
                              (pd.DataFrame([[2, 1]]), (1, 2), [0, 1])])
    def testInit(self, data, shape, columns):
        base = pbc.Base(data=data)
        assert shape == base.shape
        assert columns == base.columns.to_list()

    def testConstructor(self):
        assert isinstance(pbc.Base().iloc[:], pbc.Base)

    @pytest.mark.parametrize('TYPE_COL,ID_COLS,FMT', [('b', ['c'], None)])
    @pytest.mark.parametrize('lines', [(['1 2 3\n', '3 4 5\n'])])
    @pytest.mark.parametrize('index_col,expected', [(None, [0, 1, 1, 2]),
                                                    (0, [0, 1, 2])])
    def testFromLines(self, base, lines, index_col, expected):
        data = base.fromLines(lines, index_col=index_col)
        assert ['b', 'c'] == data.columns[-2:].to_list()
        assert expected == data.reset_index().iloc[0, :].to_list()
        assert base.fromLines([]).empty

    @pytest.mark.parametrize('FMT', [None])
    @pytest.mark.parametrize('TYPE_COL,ID_COLS,delta,index',
                             [(None, None, 1, True), ('a', ['b'], 1, True),
                              ('c', ['a'], -1, False)])
    def testShift(self, base, TYPE_COL, ID_COLS, delta, index):
        base.shift(base, delta=delta, index=index)
        if TYPE_COL:
            assert (delta == base[TYPE_COL]).all()
        if ID_COLS:
            assert (delta == base[ID_COLS]).all().all()
        assert int(index) == base.index[0]

    @pytest.mark.parametrize('TYPE_COL,ID_COLS', [('b', ['c'])])
    @pytest.mark.parametrize('index_column,as_block,FMT,shape',
                             [(None, True, None, (3, 3)),
                              (None, True, ('%i %i %i %i'), (3, 3)),
                              ('a', False, None, (3, 2)),
                              ('a', False, ('%i %i %i'), (3, 2))])
    def testWrite(self, base, index_column, as_block, shape, tmp_dir):
        with open('filename', 'w') as fh:
            base.write(fh, index_column=index_column, as_block=as_block)
        with open('filename', 'r') as fh:
            data = fh.readlines()
        if as_block:
            assert 'Block\n' == data[0]
            data = data[2:]
        data = base.fromLines(data, index_col=index_column)
        assert (0 == data).all().all()
        assert shape == data.shape

    @pytest.mark.parametrize('TYPE_COL,ID_COLS,FMT', [('b', ['c'], None)])
    @pytest.mark.parametrize('dtype,floats,expected',
                             [(int, ('float', 'int'), True),
                              (float, ('float', 'int'), True),
                              (float, ('float', ), False)])
    def testAllClose(self, base, dtype, floats, expected):
        other = np.zeros((3, 3), dtype=dtype)
        other = pd.DataFrame(other, columns=['a', 'b', 'c'])
        assert expected == base.allClose(other, floats=floats)


class TestBoxNumba:

    @pytest.fixture
    def box(self, params, tilted):
        return pbc.BoxOrig.fromParams(*params, tilted=tilted)

    @pytest.mark.parametrize('params,tilted', [([2, 2, 2, 90, 90, 90], False),
                                               ([1, 2, 3, 90, 90, 90], False),
                                               ([1, 2, 3, 90, 90, 90], True),
                                               ([2, 2, 2, 60, 70, 80], True)])
    def testVolume(self, params, box):
        cos_values = [math.cos(math.radians(x)) for x in params[3:]]
        squared = np.square(cos_values)
        product = np.prod(cos_values)
        volume = np.prod(params[:3]) * np.sqrt(1 - sum(squared) + 2 * product)
        np.testing.assert_almost_equal(box.volume, volume, decimal=2)

    @pytest.mark.parametrize('tilted', [True, False])
    @pytest.mark.parametrize('params', [([1, 2, 3, 90, 90, 90])])
    def testSpan(self, params, box):
        assert params[:3] == box.span.tolist()

    @pytest.mark.parametrize('tilted', [True, False])
    @pytest.mark.parametrize('params,expected',
                             [([1, 2, 3, 90, 90, 90], [0.5, 1, 1.5])])
    def testCenter(self, box, expected):
        assert expected == box.center.tolist()

    @pytest.mark.parametrize(
        'params,tilted,his,expected',
        [([2, None, None, 90, 90, 90], False, [2, 2, 2], None),
         ([1, 2, None, 90, 90, 90], False, [1, 2, 2], None),
         ([2, 2, 2, 90, 90, 90], False, [2, 2, 2], None),
         ([1, 2, 3, 90, 90, 90], False, [1, 2, 3], None),
         ([1, 2, 3, 90, 90, 90], True, [1, 2, 3], [0, 0, 0]),
         ([2, 2, 2, 60, 70, 80], True, [2, 1.97, 1.65], [0.34, 0.68, 0.89]),
         ([2, 2, 2, 60, 70, 80], False, None, AssertionError)])
    def testFromParams(self, params, tilted, his, expected, raises):
        with raises:
            box = pbc.BoxOrig.fromParams(*params, tilted=tilted)
        if his is None:
            return
        np.testing.assert_almost_equal(his, box['hi'], decimal=2)
        numpyutils.assert_almost_equal(box.tilt, expected, decimal=2)

    def testGetLables(self):
        labels = pbc.BoxOrig.getLabels()
        assert ['xlo', 'ylo', 'zlo'] == labels['lo_cmt']
        assert ['xhi', 'yhi', 'zhi'] == labels['hi_cmt']

    @pytest.mark.parametrize('tilted', [False])
    @pytest.mark.parametrize('params', [([1, 2, 3, 90, 90, 90]),
                                        ([3, 2, 5, 90, 90, 90])])
    def testEdges(self, params, box):
        assert sum(params[:3]) * 12 == box.edges.sum()

    @pytest.mark.parametrize('tilted,vec', [(False, [1, 1, 1])])
    @pytest.mark.parametrize('params,expected',
                             [([1, 2, 3, 90, 90, 90], 1.41421356),
                              ([3, 3, 5, 90, 90, 90], 1.73205081)])
    def testNorms(self, box, vec, expected):
        vecs = np.array([vec])
        np.testing.assert_almost_equal(box.norms(vecs), expected)

    @pytest.mark.parametrize('tilted', [False])
    @pytest.mark.parametrize('params,size', [([10], 1000), ([20], 100)])
    def testGetPoint(self, box, size):
        points = box.getPoints(size=size)
        assert size == points.shape[0] == np.unique(points, axis=0).shape[0]
        assert (box.lo.min() <= points.min()).all()
        assert (box.hi.max() >= points.max()).all()


class TestBoxOrig:

    @pytest.fixture
    def box(self, params, tilted):
        return pbc.BoxNumba.fromParams(*params, tilted=tilted)

    @pytest.mark.parametrize('tilted,vec', [(False, [1, 1, 1])])
    @pytest.mark.parametrize('params,expected',
                             [([1, 2, 3, 90, 90, 90], 1.41421356),
                              ([3, 3, 5, 90, 90, 90], 1.73205081)])
    def testNorms(self, box, vec, expected):
        vecs = np.array([vec])
        np.testing.assert_almost_equal(box.norms(vecs), expected)
