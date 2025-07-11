# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Distance cell module to search neighbors and check clashes.
"""
import functools
import itertools
import math

import methodtools
import numba
import numpy as np

from nemd import envutils
from nemd import frame
from nemd import numpyutils


class Radius(np.ndarray):
    """
    Van der Waals radius.

    He: 1.4345 OPLSUA vs 1.40 https://en.wikipedia.org/wiki/Van_der_Waals_radius
    """
    MIN_DIST = 1.4

    @methodtools.lru_cache()
    def __new__(cls, *args, struct=None, num=1, scale=0.9, min_dist=MIN_DIST):
        """
        :param struct `lmpatomic.Struct`: the structure
        :param num int: the total number of atoms.
        :param scale float: scale the mean radii by this factor
        :param min_dist float: the minimum distance.
        :return `Radius`: the vdw radius for each atom pair.
        """
        radii = struct.pair_coeffs.dist.values if struct else np.zeros(1)
        # Mix geometric https://docs.lammps.org/pair_modify.html
        radii = np.full(radii.shape * 2, radii)
        radii *= radii.transpose()
        # Energy minimum https://en.wikipedia.org/wiki/Lennard-Jones_potential
        radii = scale * pow(2, 1 / 6) * np.sqrt(radii) / 2
        radii[radii < min_dist] = min_dist
        obj = np.asarray(radii).view(cls)
        obj.map = struct.atoms.type_id.values if struct else np.zeros(
            num, dtype=int)
        return obj

    @classmethod
    def __array_ufunc__(cls, obj, ufunc, *args, **kwargs):
        """
        :param obj: the ufunc object that was called
        :param ufunc 'numpy.ufunc': the function
        :param args: a tuple of the input arguments to the ufunc
        :param kwargs: any optional or keyword arguments passed to the function
        :return scalar or array: ufunc return
        """
        args = [x.view(np.ndarray) if isinstance(x, cls) else x for x in args]
        return super().__array_ufunc__(obj, ufunc, *args, **kwargs)

    def get(self, *args):
        """
        Get radius of atom pairs.

        :param args iterable of int: the global atom id pair
        :return np.ndarray: the radii of the atom pair(s)
        """
        return self[tuple(self.map[x] for x in args)]


class CellOrig:
    """
    Grid box and track atoms.

    FIXME: triclinic support
    """

    def __init__(self, frm, span, cut, upper=20):
        """
        :param frm 'Frame': the coordinate frame
        :param span 'ndarray': the pbc span
        :param cut float: the cut-off
        :param upper int: the upper limit of the shape
        """
        self.frm = frm
        self.span = span
        self.cut = cut
        self.dims = np.floor(self.span / self.cut).astype(np.int64)
        self.dims[self.dims >= upper] = upper
        self.grids = self.span / self.dims
        shape = (self.dims[0], self.dims[1], self.dims[2], self.frm.shape[0])
        self.cell = np.zeros(shape, dtype=np.bool_)
        self.nbrs = self.getNbrs(*shape[:3])
        self.empty = np.ones(shape[:3], dtype=np.bool_)
        self.uniq = np.zeros(shape[:3], dtype=np.bool_)

    def set(self, gids, state=True):
        """
        Set the cell state.

        :param gids list of int: the global atom ids
        :param state bool: the state to set
        """
        ixs, iys, izs = self.getCid(gids).transpose()
        self.cell[ixs, iys, izs, gids] = state
        self.empty[ixs, iys, izs] = \
            not (state or self.cell[ixs, iys, izs].any())

    def getCid(self, gids):
        """
        Get the cell id(s).

        :param gids int or list of int: the global atom id(s)
        :return `np.ndarray`: the cell id(s)
        """
        cid = np.round(self.frm[gids] / self.grids).astype(np.int32)
        return cid % self.dims

    def get(self, gid, less=False):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself).

        :param gid int: the global atom ids
        :param gt bool: only include the global atom ids greater than the gid
        :return list of int: the neighbor atom ids around the coordinates
        """
        cids = map(tuple, self.nbrs[tuple(self.getCid(gid))])
        gids = [self.cell[x].nonzero()[0] for x in cids if not self.empty[x]]
        if less:
            gids = [x[x < gid] for x in gids]
        return [y for x in gids for y in x]

    def gidsGet(self, gids):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself).

        :param gids list of int: the global atom ids
        :return list of ints: the neighbor atom ids around the coordinates
        """
        cids = tuple(self.getCid(gids).transpose())
        self.uniq[cids] = True
        cids = self.uniq.nonzero()
        self.uniq[cids] = False
        for nbr in self.nbrs[cids]:
            self.uniq[tuple(nbr.transpose())] = True
        nbrs = self.uniq.nonzero()
        self.uniq[nbrs] = False
        nbrs = np.array(nbrs)[:, ~self.empty[nbrs]]
        return self.cell[tuple(np.array(nbrs))].nonzero()[1]

    @classmethod
    @functools.cache
    def getNbrs(cls, *dims):
        """
        The neighbor cells ids of all cells. By definition, neighbors are cells
        with separation distance <= the cutoff, and adjacent cells are 0 distance
        separated. In addition, one cell may contain multiple atoms.

        :param dims tuple: the dimensions (span over grid in x, y, and z)
        :return 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
            the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        # As grid >= cut, neighbor cells are one grid away around the self cell
        wrapped = [[int(math.remainder(y, x)) for y in range(-1, 2)]
                   for x in dims]
        orig = np.array((list(itertools.product(*wrapped))))
        nbrs = np.zeros((*dims, *orig.shape), dtype=int)
        for cid in itertools.product(*[range(x) for x in dims]):
            nbrs[cid] = (orig + cid) % dims
        return nbrs


@numba.experimental.jitclass([('frm', numba.float64[:, :]),
                              ('span', numba.float64[:]),
                              ('cut', numba.float64), ('dims', numba.int64[:]),
                              ('grids', numba.float64[:]),
                              ('nbrs', numba.int64[:, :, :, :, :]),
                              ('cell', numba.boolean[:, :, :, :]),
                              ('empty', numba.boolean[:, :, :]),
                              ('uniq', numba.boolean[:, :, :])])
class CellNumba(CellOrig):
    """
    See the parent. (accelerated by numba)
    """

    def set(self, gids, state=True):
        """
        See parent.
        """
        for gid in gids:
            cid = self.getCid(gid)
            self.cell[cid[0], cid[1], cid[2], gid] = state
            self.empty[cid[0], cid[1], cid[2]] = \
                not (state or self.cell[cid[0], cid[1], cid[2]].any())

    def get(self, gid, less=False):
        """
        See parent.
        """
        cid = self.getCid(gid)
        cids = self.nbrs[cid[0], cid[1], cid[2], :]
        # The atom ids from all neighbor cells
        gids = [
            self.cell[x[0], x[1], x[2], :].nonzero()[0] for x in cids
            if not self.empty[x[0], x[1], x[2]]
        ]
        if less:
            gids = [x[x < gid] for x in gids]
        return [y for x in gids for y in x]

    def gidsGet(self, gids):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself).

        :param gids list of int: the global atom ids
        :return list of ints: the neighbor atom ids around the coordinates
        """
        cids = [self.getCid(x) for x in gids]
        for cid in cids:
            self.uniq[cid[0], cid[1], cid[2]] = True
        cids = np.stack(self.uniq.nonzero()).transpose()
        for cid in cids:
            self.uniq[cid[0], cid[1], cid[2]] = False
        for cid in cids:
            for nbr in self.nbrs[cid[0], cid[1], cid[2]]:
                if self.empty[nbr[0], nbr[1], nbr[2]]:
                    continue
                self.uniq[nbr[0], nbr[1], nbr[2]] = True
        nbrs = np.stack(self.uniq.nonzero()).transpose()
        for nbr in nbrs:
            self.uniq[nbr[0], nbr[1], nbr[2]] = False
        gids = [self.cell[x[0], x[1], x[2]].nonzero()[0] for x in nbrs]
        return [y for x in gids for y in x]

    @staticmethod
    def getNbrs(nx, ny, nz):
        """
        See parent.
        """
        dims = np.array([nx, ny, nz], dtype=np.int64)
        vecs = []
        for dim in dims:
            vec = np.arange(-1, 2, dtype=np.int64)
            vecs.append(vec - np.round(vec / dim).astype(np.int64) * dim)

        orig_shape = (np.prod(np.array([x.size for x in vecs])), len(vecs))
        orig = np.zeros(orig_shape, dtype=np.int64)
        idx = 0
        for x in vecs[0]:
            for y in vecs[1]:
                for z in vecs[2]:
                    orig[idx, :] = [x, y, z]
                    idx += 1

        nbrs = np.zeros((nx, ny, nz, *orig.shape), dtype=np.int64)
        for x in numba.prange(nx):
            for y in numba.prange(ny):
                for z in numba.prange(nz):
                    cids = (orig + np.array([x, y, z])) % dims
                    nbrs[x, y, z, :, :] = cids
        return nbrs


Cell = CellOrig if envutils.is_original() else CellNumba


class Frame(frame.Base):
    """
    Search neighbors and check clashes.
    """
    Cell = Cell

    def __init__(self,
                 *args,
                 gids=None,
                 cut=None,
                 struct=None,
                 srch=None,
                 delay=False,
                 **kwargs):
        """
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'Reader': radii and excluded pairs
        :param srch: whether to use distance cell to search neighbors
        :param delay: delay the setup if True
        """
        super().__init__(*args, **kwargs)
        self.gids = numpyutils.IntArray(shape=self.shape[0], on=gids)
        self.cut = cut
        self.struct = struct
        self.srch = srch
        self.cell = None
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the distance cell.
        """
        if self.cut is None:
            self.cut = self.radii.max()
        if self.srch is False or self.box is None:
            return
        # Distance > the 1/2 diagonal is not in the first image
        # cut > edge / 2 includes all cells in that direction
        self.cut = min(self.cut, self.box.span.max() / 2)
        if self.srch is None and not self.large(self.cut):
            return
        self.cell = self.Cell(self, self.box.span, self.cut)
        self.set(self.gids.on)

    @methodtools.lru_cache()
    @property
    def radii(self):
        """
        Get the radii.

        :return `Radius`: the radii
        """
        return Radius(struct=self.struct, num=self.shape[0])

    def getDists(self, grp, grps=None, less=True):
        """
        Get the distances between atom pairs.

        :param grp list: atom global ids
        :param grps list: atom global ids to compute distances with grp
        :param less bool: grps only include the gids less than the current gid
        :return numpy.ndarray: pair distances.
        """
        if grps is None:
            grps = (self.getGrp(x, less=less) for x in grp)
        dists = [self.box.norms(self[x] - self[y]) for x, y in zip(grp, grps)]
        return np.concatenate(dists) if dists else np.array([])

    def getGrp(self, gid, less=True):
        """
        Get the global atom id group.

        :param gid int: the global atom id
        :param less bool: only include the gids less than the current gid
        :return list of floats: global atom id group
        """
        if self.cell is not None:
            return self.cell.get(gid, less)
        return self.gids.less(gid) if less else self.gids.on

    def getClashes(self, grp, grps=None, less=True):
        """
        Get the clashes distances.

        :param grp list: global atom ids.
        :param grps list: atom global ids to compute clashes with grp
        :param less bool: grps only include the gids less than the current gid
        :return list of float: the clash distances
        """
        if grps is None:
            grps = (self.getGrp(x, less=less) for x in grp)
        clshs = (self.getClash(x, grp=y, less=less) for x, y in zip(grp, grps))
        return [y for x in clshs for y in x]

    def hasClash(self, gids, grp=None):
        """
        Whether the selected atoms have clashes with the existing atoms.

        :param gids list: global atom ids for atom selection.
        :return bool: whether the selected atoms have clashes.
        """
        if self.cell is not None:
            grp = self.cell.gidsGet(gids)
        dists = (self.getClash(x, grp=grp, less=False) for x in gids)
        try:
            next(itertools.chain.from_iterable(dists))
        except StopIteration:
            return False
        return True

    def getClash(self, gid, grp=None, less=True):
        """
        Get the clashes between xyz and atoms in the frame.

        :param gid int: the global atom id
        :param grps list: atom global ids to compute clashes with gid
        :param less bool: grps only include the gids less than the current gid
        :return list of floats: the clash distances
        """
        if grp is None:
            grp = self.getGrp(gid, less=less)
        gids = self.gids.diff(self.excluded[gid], on=grp)
        dists = self.box.norms(self[gids, :] - self[gid, :])
        thresholds = self.radii.get(gid, gids)
        return dists[np.nonzero(dists < thresholds)]

    @methodtools.lru_cache()
    @property
    def excluded(self):
        """
        Set the pair exclusion.

        :return dict: global atom id -> excluded global atom ids array.
        """
        return self.getExcluded(struct=self.struct, num=self.shape[0])

    @methodtools.lru_cache()
    @classmethod
    def getExcluded(cls, struct=None, num=1, incl14=True):
        """
        Set the pair exclusion. Atoms in bonds and angles are in the exclusion.
        The dihedral angles are in the exclusion if incl14=True.

        :param struct `lmpatomic.Struct`: the structure
        :param num int: total number of atoms
        :param incl14 bool: count 1-4 interaction in a dihedral as exclusion.
        :return dict: global atom id -> excluded global atom ids array.
        """
        if struct is None:
            return {i: np.array([i]) for i in range(num)}
        pairs = set(struct.bonds.getPairs())
        pairs = pairs.union(struct.angles.getPairs())
        pairs = pairs.union(struct.impropers.getPairs())
        if incl14:
            pairs = pairs.union(struct.dihedrals.getPairs())
        excluded = {i: {i} for i in struct.atoms.atom1}
        for id1, id2 in pairs:
            excluded[id1].add(id2)
            excluded[id2].add(id1)
        return {x: np.array(list(y)) for x, y in excluded.items()}

    def set(self, gids, state=True):
        """
        Add or remove gids.

        :param gids list: the global atom ids to be added.
        :param state bool: add if True; remove if False
        """
        self.gids[gids] = state
        if self.cell is not None:
            self.cell.set(gids, state)
