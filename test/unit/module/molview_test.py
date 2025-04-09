import numpy as np
import pytest

from nemd import envutils
from nemd import lmpfull
from nemd import molview
from nemd import traj

FRM = envutils.test_data('water', 'three.custom')
RDR = lmpfull.Reader(envutils.test_data('water', 'polymer_builder.data'))


@pytest.mark.parametrize('file', [FRM])
class TestFrame:

    @pytest.fixture()
    def frm(self, file, rdf):
        return molview.Frame(traj.Traj(file), rdf=rdf)

    @pytest.mark.parametrize('rdf,expected', [(None, [9, 9, 9, 1, 1, 1]),
                                              (RDR, [9, 9, 9, 2, 2, 2])])
    def testSetUp(self, frm, expected):
        np.testing.assert_almost_equal(frm.nunique(), expected)

    @pytest.mark.parametrize('rdf,expected', [(None, (1000, 56.409))])
    def testUpdate(self, frm, expected):
        copied = frm.copy()
        frm.update(frm.trj[-1])
        assert not (copied == frm)[['xu', 'yu', 'zu']].any().any()
        assert expected == (frm.step, frm.box.max().max())

    @pytest.mark.parametrize('rdf,expected',
                             [(None, ['X', 20, '#FF1493', 9, 3]),
                              (RDR, ['O', 3.1507, '#f00000', 3, 3])])
    def testGetCoords(self, frm, expected):
        info, coords = next(frm.getCoords())
        assert expected == [*info, *coords.shape]

    @pytest.mark.parametrize('rdf,expected', [(None, ['X']),
                                              (RDR, ['O', 'H'])])
    def testElements(self, frm, expected):
        assert (expected == frm.elements).all()

    @pytest.mark.parametrize('rdf,expected', [(None, 0), (RDR, 12)])
    def testBonds(self, frm, expected):
        assert expected == len(list(frm.getBonds()))

    @pytest.mark.parametrize(
        'rdf,expected', [(None, [[-7.670475, 6.367965], [-3.896735, 10.141705],
                                 [-6.89382, 7.14462]])])
    def testGetRanges(self, frm, expected):
        np.testing.assert_almost_equal(frm.getRanges(), expected)
