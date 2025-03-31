import os

import numpy as np
import pytest

from nemd import envutils
from nemd import traj

TRAJS = 'trajs'
BASE_DIR = envutils.test_data(TRAJS)
CC3COOH = os.path.join(BASE_DIR, 'CC3COOH.custom')
CC3COOH_RANDOMIZED = os.path.join(BASE_DIR, 'CC3COOH_randomized.custom')


class TestTraj:

    @pytest.fixture
    def raw_frms(self, filename):
        return traj.Frame.read(filename)

    @pytest.mark.parametrize(('filename', 'same'),
                             [(CC3COOH, False), (CC3COOH_RANDOMIZED, True)])
    def testRead(self, raw_frms, same):
        frms = list(raw_frms)
        assert 2 == len(frms)
        frm1, frm2 = frms
        assert same == all((frm1 == frm2).all())

    @pytest.fixture
    def frm(self, filename):
        return next(traj.Frame.read(filename))

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testInit(self, frm):
        array = frm.values
        nfrm = traj.Frame(array)
        assert all((frm == nfrm).all())


class TestCell:

    @pytest.fixture
    def frm(self, filename):
        frm = next(traj.Frame.read(filename))
        return traj.Cell(frm, cut=3., res=1.)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetSpan(self, frm):
        frm.setSpan()
        assert (frm.span == 48).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetgrids(self, frm):
        frm.setSpan()
        frm.setgrids()
        assert (frm.grids == 1).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetNeighborIds(self, frm):
        frm.setSpan()
        frm.setgrids()
        frm.setNeighborIds()
        assert 311 == len(frm.neigh_ids)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetAtomCell(self, frm):
        frm.setSpan()
        frm.setgrids()
        frm.setNeighborIds()
        frm.setAtomCell()
        assert (48, 48, 48, 19) == frm.atom_cell.shape

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testGetNeighbors(self, frm):
        frm.cut = 3
        frm.res = 1
        frm.setSpan()
        frm.setgrids()
        frm.setNeighborIds()
        frm.setNeighborMap()
        frm.setAtomCell()
        xyzs = [frm.frm.getXYZ(x) for x in frm.getNeighbors((0, 0, 0))]
        dists = [np.linalg.norm(x) for x in xyzs]
        assert 16 == len(dists)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testGetNeighbors(self, frm):
        frm.cut = 3
        frm.res = 1
        frm.setSpan()
        frm.setgrids()
        frm.setNeighborIds()
        frm.setNeighborMap()
        frm.setAtomCell()
        row = frm.frm.getXYZ(1)
        assert not frm.getClashes(row)
