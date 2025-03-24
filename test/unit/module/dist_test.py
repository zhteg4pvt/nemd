import numpy as np
import pytest

from nemd import dist
from nemd import envutils
from nemd import lmpfull

NACL_DATA = envutils.test_data('itest', '0027_test', 'workspace',
                               '062200efd143bd63bc59842f7ffb56d5',
                               'amorp_bldr.data')
NACL_RDR = lmpfull.Reader(NACL_DATA)
HE_RDR = lmpfull.Reader(envutils.test_data('he', 'mol_bldr.data'))


class TestRadius:

    @pytest.mark.parametrize('struct,num,expected', [(None, 3, [1, 3]),
                                                     (NACL_RDR, 1, [2, 20]),
                                                     (HE_RDR, 1, [1, 1])])
    def testNew(self, struct, num, expected):
        radii = dist.Radius(struct=struct, num=num)
        assert expected == [radii.shape[0], radii.map.shape[0]]

    @pytest.fixture
    def radii(self, struct):
        return dist.Radius(struct=struct)

    @pytest.mark.parametrize('struct,args,expected',
                             [(HE_RDR, (0, 0), 1.4), (NACL_RDR, (0, 0), 2.231),
                              (NACL_RDR, (10, 10), 1.682),
                              (NACL_RDR, (9, 10), 1.937),
                              (NACL_RDR, (0, [1, 11]), [2.231, 1.937]),
                              (NACL_RDR, ([12, 19], [1, 11]), [1.937, 1.682])])
    def testGet(self, radii, args, expected):
        np.testing.assert_almost_equal(radii.get(*args), expected, decimal=3)
