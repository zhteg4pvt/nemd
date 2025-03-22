import numpy as np
import pytest

from nemd import box
from nemd import frame


class TestBase:

    @pytest.fixture
    def base(self, data, box, shape):
        return frame.Base(data=data, box=box, shape=shape)

    @pytest.mark.parametrize('data,box,shape',
                             [(None, None, (0, )),
                              (np.array([[1, 2], [3, 4]]), box.Box(), None)])
    def testNew(self, data, box, shape, base):
        assert base.box is box
        expected = np.zeros(shape) == data if data is None else data
        np.testing.assert_almost_equal(base, expected)

    @pytest.mark.parametrize('data,box,shape,expected',
                             [([[0.1, 0.2, 0.4], [0.9, 0.8, 0.6]],
                               box.Box.fromParams(1.), None, [0.48989795])])
    @pytest.mark.parametrize('grp,grps', [(None, None), ([0, 1], None),
                                          ([0], [[1]])])
    def testPairDists(self, base, grp, grps, expected):
        dists = base.pairDists(grp=grp, grps=grps)
        np.testing.assert_almost_equal(dists, expected)
