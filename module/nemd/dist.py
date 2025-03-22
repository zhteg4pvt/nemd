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
import pandas as pd

from nemd import box
from nemd import envutils
from nemd import frame
from nemd import numbautils
from nemd import symbols


class Radius(np.ndarray):
    """
    Class to get and hold vdw radius by atom id pair.

    NOTE: the radii here are more of diameters (or distances) between two sites.
    """

    MIN_DIST = 1.4
    SCALE = 0.45

    def __new__(cls, *args, struct=None, num=None, **kwargs):
        """
        :param struct `lmpatomic.Struct`: the structure for id map amd distances
        :param num int: the total number of atoms.
        :return `Radius`: the vdw radius for each atom pair.
        """
        # Type 0 for all atoms if no atom types provided.
        atypes = struct.atoms.type_id if struct else pd.Series([0] * num)
        dists = struct.pair_coeffs.dist if struct else pd.Series(0)
        # Data.GEOMETRIC is optimized for speed and is supported
        kwargs = dict(index=range(dists.index.max() + 1), fill_value=-1)
        radii = dists.reindex(**kwargs).values.tolist()
        radii = np.full((len(radii), len(radii)), radii)
        radii *= radii.transpose()
        radii = np.sqrt(radii)
        radii *= pow(2, 1 / 6) * cls.SCALE
        radii[radii < cls.MIN_DIST] = cls.MIN_DIST
        obj = np.asarray(radii).view(cls)
        kwargs = dict(index=range(atypes.index.max() + 1), fill_value=-1)
        obj.id_map = atypes.reindex(**kwargs).values
        return obj

    def imap(self, index):
        """
        Map the given index to the corresponding index in the array.

        :param index int, list, np.ndarray, or slice: the input index.
        :return int, list, np.ndarray, or slice: the mapped index.
        """
        if isinstance(index, slice):
            args = [index.start, index.stop, index.step]
            return slice(*[x if x is None else self.id_map[x] for x in args])
        # int, list, or np.ndarray
        return self.id_map[index]

    def __getitem__(self, index):
        """
        Get the item(s) at the given index(es).

        :param index int, list, np.ndarray, or slice: the input index.
        :return int or np.ndarray: the item(s) at the given index(es)
        """
        nindex = tuple(self.imap(x) for x in index)
        data = super().__getitem__(nindex)
        return np.asarray(data)

    def __setitem__(self, index, value):
        """
        Set the item(s) at the given index(es) to the given value(s).

        :param index int, list, np.ndarray, or slice: the input index.
        :param value int, list, np.ndarray: the value to set.
        """
        nindex = tuple(self.imap(x) for x in index)
        super().__setitem__(nindex, value)


class CellOrig(frame.Base):
    """
    Class to search neighbors and check clashes.
    """

    GRID_MAX = 20

    def __init__(self, gids=None, cut=None, struct=None, **kwargs):
        """
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'Reader': radii and excluded pairs
            are set from this object.
        """
        super().__init__(**kwargs)
        self.gids = gids
        self.cut = cut
        self.struct = struct
        self.cell = None
        if self.gids is None:
            self.gids = set(range(self.shape[0]))

    @functools.singledispatchmethod
    def setup(self, arg):
        """
        Set up the distance cell with additional arguments.
        """
        pass

    @setup.register
    def _(self, arg: frame.Frame):
        """
        Set up the distance cell starting from a trajectory frame.

        :param arg: the input trajectory frame.
        :type arg: `Frame`
        """
        self.resize(arg.shape, refcheck=False)
        self[:] = arg
        self.step = arg.step
        self.setup(arg.box)

    @setup.register
    def _(self, arg: box.Box):
        """
        Set up the distance cell starting from a new periodic boundary box.

        :param arg `Box`: the periodic boundary box.
        """
        self.box = arg
        self.setCell()

    def setCell(self):
        """
        Put atom ids into the corresponding cells.

        self.cell.shape = [X index, Y index, Z index, all atom ids]
        """
        idxs = self.getIdxs(list(range(self.shape[0])))
        self.cell = np.zeros((*self.cshape, idxs.shape[0] + 1), dtype=bool)
        for gid in self.gids:
            self.cell[tuple([*idxs[gid], gid])] = True

    def getIdxs(self, gids):
        """
        Get the cell id(s).

        :param gids list or int: the global atom id(s) to locate the cell id.
        :return `np.ndarray`: the cell id(s)
        """
        return (self[gids, :] / self.cspan).round().astype(int) % self.cshape

    @methodtools.lru_cache()
    @property
    def cspan(self):
        """
        The span of atom cells in each dimension.

        :return numpy.ndarray: span of neighboring cells in each dimension.
        """
        return self.box.span / self.cshape

    @methodtools.lru_cache()
    @property
    def cshape(self, max_num=GRID_MAX):
        """
        The number of the atom cells in each dimension.

        :param max_num int: maximum number of cells in each dimension.
        :return list of int: the number of the cell in each dimension
        """
        cut = self.cut if self.cut is not None else self.radii.max()
        return np.array(
            [min(max_num, math.ceil(x / cut)) for x in self.box.span])

    def getClashes(self, gids=None):
        """
        Get the clashes distances.

        :param gids set: global atom ids for atom selection.
        :return list of float: the clash distances
        """
        if gids is None:
            gids = self.gids
        return [y for x in gids for y in self.getClash(x)]

    def hasClash(self, gids):
        """
        Whether the selected atoms have clashes.

        :param gids set: global atom ids for atom selection.
        :return bool: whether the selected atoms have clashes.
        """
        dists = (self.getClash(x) for x in gids)
        try:
            next(itertools.chain.from_iterable(dists))
        except StopIteration:
            return False
        return True

    def getClash(self, gid):
        """
        Get the clashes between xyz and atoms in the frame.

        :param gid int: the global atom id
        :return list of floats: clash distances between atom pairs
        """
        neighbors = set(self.getNbrs(gid))
        neighbors = neighbors.difference(self.excluded[gid])
        if not neighbors:
            return []
        neighbors = list(neighbors)
        dists = self.box.norm(self[neighbors, :] - self[gid, :])
        thresholds = self.radii[gid, neighbors]
        return dists[np.nonzero(dists < thresholds)]

    def getNbrs(self, gid):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself).

        :param xyz 1x3 array of floats: xyz of one atom coordinates
        :return list of ints: the atom ids of the neighbor atoms
        """
        idx = self.nbr[tuple(self.getIdxs(gid))]
        return [y for x in idx for y in self.cell[tuple(x)].nonzero()[0]]

    @methodtools.lru_cache()
    @property
    def nbr(self):
        """
        The neighbor cells ids of all cells.

        :return 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
            the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        nbr = np.zeros((*self.cshape, *self.nbr_inc.shape), dtype=int)
        nodes = list(itertools.product(*[range(x) for x in self.cshape]))
        for node in nodes:
            nbr[node] = (self.nbr_inc + node) % self.cshape
        cols = list(itertools.product(*[range(x) for x in self.cshape]))
        unique_maps = [np.unique(nbr[tuple(x)], axis=0) for x in cols]
        shape = np.unique([x.shape for x in unique_maps], axis=0).max(axis=0)
        nbr = np.zeros((*self.cshape, *shape), dtype=int)
        for col, unique_map in zip(cols, unique_maps):
            nbr[col[0], col[1], col[2], :, :] = unique_map
        # getNbrMap() and the original mode generate nbr in different
        # order: np.unique(nbr[i, j, j, :, :], axis=0) remains the same
        return nbr

    @methodtools.lru_cache()
    @property
    def nbr_inc(self):
        """
        The neighbor cells ids when sitting on the (0,0,0) cell. (Cells with
        separation distances less than the cutoff are set as neighbors)

        :return nx3 numpy.ndarray: the neighbor cell ids.
        """

        def separation_dist(ijk):
            separation_ids = [y - 1 if y else y for y in ijk]
            return np.linalg.norm(self.cspan * separation_ids)

        cut = self.cut if self.cut is not None else self.radii.max()
        max_ids = [math.ceil(cut / x) + 1 for x in self.cspan]
        ijks = itertools.product(*[range(max_ids[x]) for x in range(3)])
        # Adjacent Cells are zero distance separated.
        nbr_ids = [x for x in ijks if separation_dist(x) < cut]
        # Keep itself (0,0,0) cell as multiple atoms may be in one cell.
        signs = itertools.product((-1, 1), (-1, 1), (-1, 1))
        signs = [np.array(x) for x in signs]
        uq_nbr_ids = set([tuple(y * x) for x in signs for y in nbr_ids])
        return np.array(list(uq_nbr_ids))

    @methodtools.lru_cache()
    @property
    def radii(self):
        """
        The vdw radius.

        :return lmpatomic.Radius: the vdw radius by type.
        """
        return Radius(struct=self.struct, num=self.shape[0])

    @methodtools.lru_cache()
    @property
    def excluded(self, include14=True):
        """
        Set the pair exclusion during clash check. Bonded atoms and atoms in
        angles are in the exclusion. The dihedral angles are in the exclusion
        if include14=True.

        :param include14 bool: If True, 1-4 interactions in a dihedral angle
            count as exclusion.
        :return dict: the key is the global atom id, and values are excluded
            global atom ids set.
        """
        excluded = {i: set([i]) for i in range(self.shape[0])}
        if self.struct is None:
            return excluded

        pairs = set(self.struct.bonds.getPairs())
        pairs = pairs.union(self.struct.angles.getPairs())
        pairs = pairs.union(self.struct.impropers.getPairs())
        if include14:
            pairs = pairs.union(self.struct.dihedrals.getPairs())
        for id1, id2 in pairs:
            excluded[id1].add(id2)
            excluded[id2].add(id1)
        return excluded

    def add(self, gids):
        """
        Add gids to atom cell and existing gids.

        :param gids list: the global atom ids to be added.
        """
        self.gids.update(gids)
        for idx, (ix, iy, iz) in zip(gids, self.getIdxs(gids)):
            self.cell[ix, iy, iz][idx] = True

    def remove(self, gids):
        """
        Remove gids from atom cell and existing gids.

        :param gids list: the global atom ids to be removed.
        """
        self.gids = self.gids.difference(gids)
        for idx, (ix, iy, iz) in zip(gids, self.getIdxs(gids)):
            self.cell[ix, iy, iz][idx] = False

    @property
    def ratio(self):
        """
        The ratio of the existing atoms to the total atoms.

        :return str: the ratio of the existing gids with respect to the total.
        """
        return f'{len(self.gids)} / {self.shape[0]}'

    def nbrDists(self, gids=None, nbrs=None):
        """
        Get the pair distances between existing atoms.

        :param gids list: the center atom global atom ids.
        :param nbrs list: the neighbor global atom ids.
        :return 'numpy.ndarray': the pair distances
        """
        grp = sorted(self.gids) if gids is None else sorted(gids)
        nbrs = map(self.getNbrs, grp) if nbrs is None else [nbrs] * len(grp)
        grps = [[z for z in y if z < x] for x, y in zip(grp, nbrs)]
        return self.pairDists(grp=grp, grps=grps)


class CellNumba(CellOrig):

    IS_NOPYTHON = envutils.is_nopython()

    def setCell(self):
        """
        Put atom ids into the corresponding cells.

        self.cell.shape = [X index, Y index, Z index, all atom ids]
        """
        gids = numba.int32(list(self.gids))
        self.cell = self.setCellNumba(gids, self.cspan, self.cshape)

    @numbautils.jit
    def setCellNumba(self, gids, cspan, cshape, nopython=IS_NOPYTHON):
        """
        Put atom ids into the corresponding cells.

        :param gids 'numba.int32': the global atom gids
        :param cspan 'numpy.ndarray': the length of the cell in each dimension
        :param cshape list of numba.int32: the cell number in each dimension
        :param nopython bool: whether numba nopython mode is on
        :return 'numpy.ndarray': map between cell id to atom ids
            [X index, Y index, Z index, all atom ids]
        """
        shape = (cshape[0], cshape[1], cshape[2], self.shape[0])
        cell = np.zeros(shape, dtype=numba.boolean if nopython else np.bool_)
        cell_ids = np.round(self / cspan).astype(np.int64) % cshape
        for aid, cell_id in zip(gids, cell_ids):
            cell[cell_id[0], cell_id[1], cell_id[2], aid] = True
        return cell

    def hasClash(self, gids):
        """
        Whether the selected atoms have clashes

        :param gids set: global atom ids for atom selection.
        :return bool: whether the selected atoms have clashes
        """
        idxs = self.getIdxs(gids)
        return self.hasClashNumba(gids, idxs, self.radii.id_map, self.radii,
                                  self.excluded, self.nbr, self.cell,
                                  self.box.span)

    @numbautils.jit
    def hasClashNumba(self, gids, idxs, id_map, radii, excluded, nbr, cell,
                      span):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself) via Numba.

        :param gids set: global atom ids for selection
        :param idxs list of 3x1 'numpy.ndarray': the cell id(s)
        :param id_map 1xn 'numpy.ndarray': map global atom ids to atom types
        :param radii nxn 'numpy.ndarray': the radius of atom type pairs
        :param excluded dict of int list: atom ids to be excluded in clash check
        :param nbr ixjxkxnx3 'numpy.ndarray': cell id to neighbor cell ids
        :param cell ixjxkxn array of floats: cell id to containing atom ids
        :param span 1x3 'numpy.ndarray': the span of the box
        :return float: the fist clash distance between the selected atoms
        """
        for gid, idx in zip(gids, idxs):
            ids = nbr[idx[0], idx[1], idx[2], :]
            nbrs = np.array([
                y for x in ids for y in cell[x[0], x[1], x[2], :].nonzero()[0]
                if y not in excluded[gid]
            ])
            if not nbrs.size:
                continue
            dists = numbautils.norm(self[nbrs, :] - self[gid, :], span)
            if (radii[id_map[gid], id_map[nbrs]] > np.array(dists)).any():
                return True

    @methodtools.lru_cache()
    @property
    def nbr(self):
        """
        The neighbor cells ids of all cells.

        :reteurn 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
        the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        return self.getNbrNumba(self.nbr_inc, self.cshape)

    @staticmethod
    @numbautils.jit
    def getNbrNumba(nbr_ids, cshape, nopython=IS_NOPYTHON):
        """
        Get map between node id to neighbor node ids.

        :param nbr_ids numpy.ndarray: Neighbors cells sitting on the (0,0,0)
        :param cshape numpy.ndarray: the number of cells in three dimensions
        :param nopython bool: whether numba nopython mode is on
        :return numpy.ndarray: map between node id to neighbor node ids
        """
        # Unique neighbor cell ids
        min_id = np.min(nbr_ids)
        shifted_nbr_ids = nbr_ids - min_id
        wrapped_nbr_ids = shifted_nbr_ids % cshape
        ushape = np.max(wrapped_nbr_ids) + 1
        boolean = numba.boolean if nopython else np.bool_
        uids = np.zeros((ushape, ushape, ushape), dtype=boolean)
        for wrapped_ids in wrapped_nbr_ids:
            uids[wrapped_ids[0], wrapped_ids[1], wrapped_ids[2]] = True
        uq_ids = np.array(list([list(x) for x in uids.nonzero()])).T + min_id
        # Build neighbor map based on unique neighbor ids
        shape = (cshape[0], cshape[1], cshape[2], len(uq_ids), 3)
        neigh_mp = np.empty(shape, dtype=numba.int32 if nopython else np.int32)
        for xid in numba.prange(cshape[0]):
            for yid in numba.prange(cshape[1]):
                for zid in numba.prange(cshape[2]):
                    idx = np.array([xid, yid, zid])
                    neigh_mp[xid, yid, zid, :, :] = (uq_ids + idx) % cshape
        return neigh_mp

    def getNbrs(self, gid):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself).

        :param gid int: the global atom id to locate the cell and the neighbors
        :return 'numpy.ndarray': the atom ids of the neighbor atoms
        """
        return self.getNbrsNumba(self.getIdxs(gid), self.nbr, self.cell)

    def getIdxs(self, gids):
        """
        Get the cell id(s).

        :param gids list or int: the global atom id(s) to locate the cell id.
        :return `np.ndarray`: the cell id(x)
        """
        return self.getIdxsNumba(np.array(gids), self.cspan, self.cshape)

    @numbautils.jit
    def getIdxsNumba(self, gids, cspan, cshape, nopython=IS_NOPYTHON):
        """
        Get the cell id for xyz.

        :param cspan 'numpy.ndarray': the length of the cell in each dimension
        :param cshape list of 'numba.int32': the cell number in each dimension
        :param nopython bool: whether numba nopython mode is on
        :return list of ints: the cell id
        """
        int32 = numba.int32 if nopython else np.int32
        return np.round(self[gids] / cspan).astype(int32) % cshape

    @staticmethod
    @numbautils.jit
    def getNbrsNumba(idx, nbr, cell):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself) via Numba.

        :param xyz 1x3 'numpy.ndarray': xyz of one atom coordinates
        :param nbr ixjxkxnx3 'numpy.ndarray': cell id to neighor cell ids
        :param cell ixjxkxn array of floats: cell id into with atom ids
        :return list of int: the atom ids of the neighbor atoms
        """
        ids = nbr[idx[0], idx[1], idx[2], :]
        # The atom ids from all neighbor cells
        return [y for x in ids for y in cell[x[0], x[1], x[2], :].nonzero()[0]]

    @methodtools.lru_cache()
    @property
    def excluded(self):
        """
        The excluded global atom ids in numba typed dictionary.

        :return numba.typed.Dict: the key is the global atom id, and values are
            excluded global atom ids set.
        """
        excluded = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64[:],
        )
        for key, val in super().excluded.items():
            excluded[key] = np.array(list(val)).astype(np.int64)
        return excluded


Cell = CellOrig if envutils.is_original() else CellNumba


class DistCell(Cell):
    """
    The cell class to analyze the pair distances.
    """

    def __init__(self, span=None, cut=symbols.DEFAULT_CUT, **kwargs):
        """
        :param span float: the minimum span of the cell
        :param cut float: the cutoff distance for neighbor search
        """
        super().__init__(cut=cut, **kwargs)
        self.dist = max(span / self.GRID_MAX, cut) if span > cut * 5 else None

    def getDists(self, frm):
        """
        Get the pair distances of the given frame.

        :param frm `frame.Frame`: the input trajectory frame.
        :return `numpy.ndarray`: the pair distances.
        """
        if self.dist is None:
            return frm.pairDists(grp=self.gids)
        self.setup(frm)
        return self.nbrDists()
