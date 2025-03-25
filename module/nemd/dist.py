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
from nemd import numbautils
from nemd import pbc
from nemd import symbols

IS_NOPYTHON = envutils.is_nopython()


class Radius(np.ndarray):
    """
    Van der Waals radius.

    He: 1.4345 OPLSUA vs 1.40 https://en.wikipedia.org/wiki/Van_der_Waals_radius
    """

    def __new__(cls, *args, struct=None, num=1, scale=0.9, min_dist=1.4):
        """
        :param struct `lmpatomic.Struct`: the structure for id map amd distances
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
        obj.map = struct.atoms.type_id.values if struct else np.zeros(num)
        return obj

    def get(self, *args):
        """
        Get radius of atom pairs.

        :param args tuple of int, list, np.ndarray: the global atom ids
        :return np.ndarray: the radii of the atom pair(s)
        """
        return self[tuple(self.map[x] for x in args)]


class AtomCell(np.ndarray):
    """
    FIXME: triclinic support
    """

    def __new__(cls, frm, cut=None, *args, **kwargs):
        dims = [
            min(CellOrig.GRID_MAX, math.ceil(x / cut)) for x in frm.box.span
        ]
        grids = frm.box.span / dims
        obj = np.zeros((*dims, frm.shape[0]), dtype=bool).view(cls)
        obj.cut = cut
        obj.dims = np.array(dims)
        obj.grids = np.array(grids)
        return obj

    def set(self, xyzs, gids, state=True):
        """
        Set the cell state.

        :param xyzs nx3 'numpy.ndarray': the coordinates to retrieve the cells
        :param gids set: the corresponding global atom ids
        :param state bool: the state to set
        """
        ixs, iys, izs = self.getIds(xyzs).transpose()
        self[ixs, iys, izs, gids] = state

    def getIds(self, xyzs):
        """
        Get the cell id(s).

        :param gids list or int: the global atom id(s) to locate the cell id.
        :return `np.ndarray`: the cell id(x)
        """
        return (xyzs / self.grids).round().astype(int) % self.dims

    def getGids(self, xyzs):
        """
        Get the neighbor global atom ids from the neighbor cells (including the
        current cell itself).

        :param xyz 1x3 array of floats: xyz of one atom coordinates
        :return list of ints: the atom ids of the neighbor atoms
        """
        idx = self.nbr[tuple(self.getIds(xyzs))]
        return [y for x in idx for y in self[tuple(x)].nonzero()[0]]

    @methodtools.lru_cache()
    @property
    def nbr(self):
        """
        The neighbor cells ids of all cells.

        :return 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
            the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        nbr = np.zeros((*self.dims, *self.nbr_inc.shape), dtype=int)
        nodes = list(itertools.product(*[range(x) for x in self.dims]))
        for node in nodes:
            nbr[node] = (self.nbr_inc + node) % self.dims
        cols = list(itertools.product(*[range(x) for x in self.dims]))
        unique_maps = [np.unique(nbr[tuple(x)], axis=0) for x in cols]
        shape = np.unique([x.shape for x in unique_maps], axis=0).max(axis=0)
        nbr = np.zeros((*self.dims, *shape), dtype=int)
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
            return np.linalg.norm(self.grids * separation_ids)

        max_ids = [math.ceil(self.cut / x) + 1 for x in self.grids]
        ijks = itertools.product(*[range(max_ids[x]) for x in range(3)])
        # Adjacent Cells are zero distance separated.
        nbr_ids = [x for x in ijks if separation_dist(x) < self.cut]
        # Keep itself (0,0,0) cell as multiple atoms may be in one cell.
        signs = itertools.product((-1, 1), (-1, 1), (-1, 1))
        signs = [np.array(x) for x in signs]
        uq_nbr_ids = set([tuple(y * x) for x in signs for y in nbr_ids])
        return np.array(list(uq_nbr_ids))


class CellOrig(frame.Base):
    """
    Search neighbors and check clashes.
    """
    AtomCell = AtomCell
    GRID_MAX = 20

    def __init__(self, gids=None, cut=None, struct=None, **kwargs):
        """
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'Reader': radii and excluded pairs
        """
        super().__init__(**kwargs)
        self.gids = gids
        self.cut = cut
        self.struct = struct
        self.cell = None
        self.radii = Radius(struct=self.struct, num=self.shape[0])
        if self.gids is None:
            self.gids = set(range(self.shape[0]))
        if self.cut is None:
            self.cut = self.radii.max()

    @functools.singledispatchmethod
    def setup(self, arg):
        """
        Set up the distance cell with additional arguments.
        """
        pass

    @setup.register
    def _(self, arg: frame.Frame):
        """
        Set up coordinates, step, box, and distance cell.

        :param arg `Frame`: the input trajectory frame.
        """
        self.resize(arg.shape, refcheck=False)
        self[:] = arg
        self.step = arg.step
        self.setup(arg.box)

    @setup.register
    def _(self, arg: pbc.Box):
        """
        Set up box and distance cell.

        :param arg `Box`: the simulation box
        """
        self.box = arg
        self.setCell()

    def setCell(self):
        """
        Put atom ids into the corresponding cells.

        self.cell.shape = [X index, Y index, Z index, all atom ids]
        """
        self.cell = self.AtomCell(self, cut=self.cut)
        if not self.gids:
            return
        self.cell.set(self[list(self.gids), :], self.gids)

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
        neighbors = set(self.cell.getGids(self[gid, :]))
        neighbors = neighbors.difference(self.excluded[gid])
        if not neighbors:
            return []
        neighbors = list(neighbors)
        dists = self.box.norm(self[neighbors, :] - self[gid, :])
        thresholds = self.radii.get(gid, neighbors)
        return dists[np.nonzero(dists < thresholds)]

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
        self.cell.set(self[gids, :], gids)

    def remove(self, gids):
        """
        Remove gids from atom cell and existing gids.

        :param gids list: the global atom ids to be removed.
        """
        self.gids = self.gids.difference(gids)
        self.cell.set(self[gids, :], gids, state=False)

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
        nbrs = map(self.cell.getGids,
                   self[grp, :]) if nbrs is None else [nbrs] * len(grp)
        grps = [[z for z in y if z < x] for x, y in zip(grp, nbrs)]
        return self.pairDists(grp=grp, grps=grps)


class AtomCellNumba(AtomCell):

    def set(self, *args, **kwargs):
        """
        See the parent.
        """
        numbautils.set(self, self.grids, self.dims, *args, **kwargs)

    def getIds(self, *args):
        """
        See the parent.
        """
        return numbautils.get_ids(self.grids, self.dims, *args)

    def getGids(self, *args):
        """
        See the parent.
        """
        return numbautils.get_atoms(self.getIds(*args), self.nbr, self)

    @methodtools.lru_cache()
    @property
    def nbr(self):
        """
        See the parent.
        """
        return numbautils.get_nbr(self.nbr_inc, self.dims)


class CellNumba(CellOrig):

    AtomCell = AtomCellNumba

    def hasClash(self, gids):
        """
        Whether the selected atoms have clashes

        :param gids set: global atom ids for atom selection.
        :return bool: whether the selected atoms have clashes
        """
        idxs = self.cell.getIds(self[gids, :])
        return self.hasClashNumba(gids, idxs, self.radii.map, self.radii,
                                  self.excluded, self.cell.nbr, self.cell,
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
