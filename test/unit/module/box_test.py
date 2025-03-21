import math

import numpy as np
import pandas as pd
import pytest

from nemd import box


class TestBase:

    @pytest.fixture
    def base(self, TYPE_COL, ID_COLS, FMT):
        attrs = dict(COLUMNS=['a', 'b', 'c'],
                     TYPE_COL=TYPE_COL,
                     ID_COLS=ID_COLS,
                     FMT=FMT)
        Base = type('Raw', (box.Base, ), attrs)
        return Base(np.zeros((3, 3), dtype=int))

    @pytest.mark.parametrize('data,shape,columns',
                             [(2, (2, 1), ['label']),
                              (pd.DataFrame([[2, 1]]), (1, 2), [0, 1])])
    def testInit(self, data, shape, columns):
        base = box.Base(data=data)
        assert shape == base.shape
        assert columns == base.columns.to_list()

    def testConstructor(self):
        assert isinstance(box.Base().iloc[:], box.Base)

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
    @pytest.mark.parametrize('dtype,expected', [(int, True), (float, False)])
    def testAllClose(self, base, dtype, expected):
        other = np.zeros((3, 3), dtype=dtype)
        other = pd.DataFrame(other, columns=['a', 'b', 'c'])
        assert expected == base.allClose(other)


class TestBox:

    @pytest.fixture
    def abox(self, al, bl, cl, alpha, beta, gamma, tilted):
        return box.Box.fromVecs(al, bl, cl, alpha, beta, gamma, tilted=tilted)

    @pytest.mark.parametrize('al,bl,cl,alpha,beta,gamma,tilted',
                             [(2, 2, 2, 90, 90, 90, False),
                              (1, 2, 3, 90, 90, 90, False),
                              (1, 2, 3, 90, 90, 90, True),
                              (2, 2, 2, 60, 70, 80, True)])
    def testVolume(self, al, bl, cl, alpha, beta, gamma, tilted, abox):
        cos_values = [math.cos(math.radians(x)) for x in [alpha, beta, gamma]]
        squared = np.square(cos_values)
        product = np.prod(cos_values)
        volume = al * bl * cl * np.sqrt(1 - sum(squared) + 2 * product)
        np.testing.assert_almost_equal(abox.volume, volume, decimal=2)

    @pytest.mark.parametrize('al,bl,cl,alpha,beta,gamma,tilted',
                             [(2, 2, 2, 90, 90, 90, False),
                              (1, 2, 3, 90, 90, 90, False),
                              (1, 2, 3, 90, 90, 90, True)])
    def testSpan(self, al, bl, cl, alpha, beta, gamma, tilted, abox):
        assert [al, bl, cl] == abox.span.tolist()

    @pytest.mark.parametrize(
        'al,bl,cl,alpha,beta,gamma,tilted,his,expected',
        [(2, None, None, 90, 90, 90, False, [2, 2, 2], None),
         (1, 2, None, 90, 90, 90, False, [1, 2, 2], None),
         (2, 2, 2, 90, 90, 90, False, [2, 2, 2], None),
         (1, 2, 3, 90, 90, 90, False, [1, 2, 3], None),
         (1, 2, 3, 90, 90, 90, True, [1, 2, 3], [0, 0, 0]),
         (2, 2, 2, 60, 70, 80, True, [2, 1.97, 1.65], [0.34, 0.68, 0.89]),
         (2, 2, 2, 60, 70, 80, False, None, AssertionError)])
    def testFromVecs(self, al, bl, cl, alpha, beta, gamma, tilted, his,
                     expected, raises):
        with raises:
            abox = box.Box.fromVecs(al,
                                    bl,
                                    cl,
                                    alpha,
                                    beta,
                                    gamma,
                                    tilted=tilted)
        if his is None:
            return
        np.testing.assert_almost_equal(his, abox['hi'], decimal=2)
        if expected:
            np.testing.assert_almost_equal(abox.tilt, expected, decimal=2)
        else:
            assert abox.tilt is None

    def testGetLables(self):
        labels = box.Box.getLabels()
        assert ['xlo', 'ylo', 'zlo'] == labels['lo_cmt']
        assert ['xhi', 'yhi', 'zhi'] == labels['hi_cmt']

    @pytest.mark.parametrize('al,bl,cl,alpha,beta,gamma,tilted',
                             [(1, 2, 3, 90, 90, 90, False),
                              (3, 2, 5, 90, 90, 90, True)])
    def testEdges(self, al, bl, cl, alpha, beta, gamma, tilted, abox):
        assert sum([al, bl, cl]) * 12 == abox.edges.sum()
