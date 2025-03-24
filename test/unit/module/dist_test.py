import numpy as np
import pytest

from nemd import dist
from nemd import envutils
from nemd import lmpfull

NACL_DATA = envutils.test_data('itest', '0027_test', 'workspace',
                               '062200efd143bd63bc59842f7ffb56d5',
                               'amorp_bldr.data')
NACL_READER = lmpfull.Reader(NACL_DATA)


class TestRadius:

    @pytest.mark.parametrize('struct,num,expected',
                             [(None, 3, [1, 3]), (NACL_READER, 1, [2, 20])])
    def testNew(self, struct, num, expected):
        radii = dist.Radius(struct=struct, num=num)
        assert expected == [radii.shape[0], radii.map.shape[0]]

    @pytest.fixture
    def radii(self):
        return dist.Radius(struct=NACL_READER)

    @pytest.mark.parametrize('args,expected',
                             [((0, 0), 2.2311627), ((10, 10), 1.6822114),
                              ((9, 10), 1.9373403),
                              ((0, [1, 11]), [2.2311627, 1.9373403]),
                              (([12, 19], [1, 11]), [1.9373403, 1.6822114])])
    def testGet(self, radii, args, expected):
        np.testing.assert_almost_equal(radii.get(*args), expected)
