# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Distance cell module to search neighbors and check clashes.
"""
import itertools
import math

import methodtools
import numpy as np

from nemd import envutils
from nemd import frame
from nemd import numbautils
from nemd import numpyutils


class Radius(np.ndarray):
    """
    Van der Waals radius.

    He: 1.4345 OPLSUA vs 1.40 https://en.wikipedia.org/wiki/Van_der_Waals_radius
    """

    @methodtools.lru_cache()
    def __new__(cls, *args, struct=None, num=1, scale=0.9, min_dist=1.4):
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
        obj.map = struct.atoms.type_id.values if struct else np.zeros(num)
        return obj

    def get(self, *args):
        """
        Get radius of atom pairs.

        :param args tuple of int, list, np.ndarray: the global atom ids
        :return np.ndarray: the radii of the atom pair(s)
        """
        return self[tuple(self.map[x] for x in args)]


class Cell(np.ndarray):
    """
    Grid box and track atoms.

    FIXME: triclinic support
    """

    def __new__(cls, frm):
        """
        :param frm 'Frame': the distance frame
        """
        dims = tuple(min(math.floor(x / frm.cut), 20) for x in frm.box.span)
        obj = np.zeros([*dims, frm.shape[0]], dtype=bool).view(cls)
        obj.frm = frm
        obj.dims = np.array(dims)
        obj.grids = frm.box.span / dims
        obj.nbrs = cls.getNbrs(dims, *np.ceil(obj.grids / frm.cut).astype(int))
        return obj

    def set(self, gids, state=True):
        """
        Set the cell state.

        :param gids list of int: the global atom ids
        :param state bool: the state to set
        """
        ixs, iys, izs = self.getCids(gids).transpose()
        self[ixs, iys, izs, gids] = state

    def getCids(self, gids):
        """
        Get the cell id(s).

        :param gids list of int: the global atom ids
        :return `np.ndarray`: the cell id(s)
        """
        return (self.frm[gids] / self.grids).round().astype(int) % self.dims

    def get(self, gid):
        """
        Get the global atom ids of neighbor (and self) cells.

        :param gids list of int: the global atom ids
        :return list of ints: the neighbor atom ids around the coordinates
        """
        nbrs = self.nbrs[tuple(self.getCids(gid))]
        return [y for x in nbrs for y in self[tuple(x)].nonzero()[0]]

    @methodtools.lru_cache()
    @classmethod
    def getNbrs(cls, dims, *nums):
        """
        The neighbor cells ids of all cells.

        :param dims: number of grids per dimension
        :param nums: number of cutoffs per grid
        :return 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
            the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        nbr = cls.getNbr(nums, dims)
        nbrs = np.zeros((*dims, *nbr.shape), dtype=int)
        cids = list(itertools.product(*[range(x) for x in dims]))
        for cid in cids:
            nbrs[cid] = (nbr + cid) % dims
        # Duplicated due to PBCs
        uniques = [np.unique(nbrs[x], axis=0) for x in cids]
        shape = np.array([x.shape for x in uniques]).max(axis=0)
        nbrs = np.zeros((*dims, *shape), dtype=int)
        for cid, unique in zip(cids, uniques):
            nbrs[cid] = unique
        return nbrs

    @methodtools.lru_cache()
    @classmethod
    def getNbr(cls, nums, dims):
        """
        The neighbor cells of the (0,0,0) cell without considering the PBC.

        :param nums: number of cutoffs per grid
        :param dims numpy.ndarray: the number of cells in three dimensions
        :return nx3 numpy.ndarray: the neighbor cell ids.
        """
        signs = itertools.product((-1, 1), (-1, 1), (-1, 1))
        signs = [np.array(x) for x in signs]
        # Neighbors are cells separation distance <= the cutoff. Adjacent cells
        # are 0 distance separated, and one cell may contain multiple atoms.
        ijks = list(itertools.product(*[range(x + 1) for x in nums]))
        nbr = np.prod(list(itertools.product(signs, ijks)), axis=1)
        nbr = np.unique(nbr, axis=0)
        # Unique neighbor cell ids
        min_cid = np.min(nbr)
        shifted = nbr - min_cid
        wrapped = shifted % dims
        uids = np.zeros([np.max(wrapped) + 1] * 3, dtype=bool)
        for cid in wrapped:
            uids[tuple(cid)] = True
        return np.array([x for x in uids.nonzero()]).T + min_cid


class CellNumba(Cell):

    def set(self, *args, **kwargs):
        """
        See the parent.
        """
        numbautils.set(self, self.grids, self.dims, self.frm, *args, **kwargs)

    def get(self, *args):
        """
        See the parent.
        """
        return numbautils.get(self, self.nbrs, self.grids, self.dims, self.frm,
                              *args)

    @methodtools.lru_cache()
    @classmethod
    def getNbrs(cls, dims, *nums):
        """
        See the parent.
        """
        return numbautils.get_nbrs(np.array(dims), cls.getNbr(nums, dims))


class Frame(frame.Base):
    """
    Search neighbors and check clashes.
    """
    Cell = Cell if envutils.is_original() else CellNumba

    def __init__(self,
                 gids=None,
                 cut=None,
                 struct=None,
                 search=None,
                 **kwargs):
        """
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'Reader': radii and excluded pairs
        :param search: use distance cell to search neighbors
        """
        super().__init__(**kwargs)
        self.gids = numpyutils.IntArray(gids, mval=self.shape[0])
        self.cut = cut
        self.struct = struct
        self.cell = None
        if search is False:
            return
        if self.cut is None:
            self.cut = self.radii.max()
        # First image distance
        self.cut = min(self.cut, self.box.span.min() / 2)
        if search is None and self.box.span.min() < self.cut * 5:
            return
        self.cell = self.Cell(self)
        self.add(self.gids.values)

    @property
    def max_dist(self):
        """
        Get maximum accurate distance.

        :return float: atoms further away may find the shortcut to the image or
            get lost during the neighbor search.
        """
        return self.box.span.min() / 2 if self.cell is None else self.cut

    def getDists(self, grp=None, grps=None):
        """
        Get the distances between atom pairs.

        :param grp list: atom global ids
        :param grps list of list: each sublist contains atom global ids to
            compute distances with each atom in grp.
        return numpy.ndarray: pair distances.
        """
        grp = sorted(self.gids.values if grp is None else grp)
        if grps is None:
            if self.cell is None:
                grps = [grp[i:] for i in range(1, len(grp))]
            else:
                grps = [self.cell.get(x) for x in grp]
                grps = [[z for z in y if z < x] for x, y in zip(grp, grps)]
        vecs = [self[x, :] - self[y, :] for x, y in zip(grps, grp)]
        if not vecs:
            return np.array([])
        return np.concatenate([self.box.norms(x) for x in vecs])

    def getClashes(self, gids=None):
        """
        Get the clashes distances.

        :param gids set: global atom ids for atom selection.
        :return list of float: the clash distances
        """
        if gids is None:
            gids = self.gids.values
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
        nbrs = self.gids if self.cell is None else numpyutils.IntArray(
            self.cell.get(gid), mval=self.shape[0])

        nbrs = nbrs.difference(self.excluded[gid])
        dists = self.box.norms(self[nbrs, :] - self[gid, :])
        thresholds = self.radii.get(gid, nbrs)
        return dists[np.nonzero(dists < thresholds)]

    @methodtools.lru_cache()
    @property
    def radii(self):
        """
        Get the radii.

        :return `Radius`: the radii
        """
        return Radius(struct=self.struct, num=self.shape[0])

    @methodtools.lru_cache()
    @property
    def excluded(self, include14=True):
        """
        Set the pair exclusion.

        :param include14 bool: count 1-4 interaction in a dihedral as exclusion.
        :return dict: global atom id -> excluded global atom ids set.
        """
        return self.getExclusions(struct=self.struct,
                                  num=self.shape[0],
                                  include14=include14)

    @methodtools.lru_cache()
    @classmethod
    def getExclusions(cls, struct=None, num=1, include14=True):
        """
        Set the pair exclusion. Atoms in bonds and angles are in the exclusion.
        The dihedral angles are in the exclusion if include14=True.

        :param struct `lmpatomic.Struct`: the structure
        :param num int: total number of atoms
        :param include14 bool: count 1-4 interaction in a dihedral as exclusion.
        :return dict: global atom id -> excluded global atom ids list.
        """
        if struct is None:
            return {i: {i} for i in range(num)}
        pairs = set(struct.bonds.getPairs())
        pairs = pairs.union(struct.angles.getPairs())
        pairs = pairs.union(struct.impropers.getPairs())
        if include14:
            pairs = pairs.union(struct.dihedrals.getPairs())
        excluded = {i: {i} for i in struct.atoms.atom1}
        for id1, id2 in pairs:
            excluded[id1].add(id2)
            excluded[id2].add(id1)
        return {x: list(y) for x, y in excluded.items()}

    def add(self, gids):
        """
        Add gids to atom cell and existing gids.

        :param gids list: the global atom ids to be added.
        """
        self.gids[gids] = True
        if self.cell is None:
            return
        self.cell.set(gids)

    def remove(self, gids):
        """
        Remove gids from atom cell and existing gids.

        :param gids list: the global atom ids to be removed.
        """
        self.gids[gids] = False
        if self.cell is None:
            return
        self.cell.set(gids, state=False)

    @property
    def ratio(self):
        """
        The ratio of the existing atoms to the total atoms.

        :return str: the ratio of the existing gids with respect to the total.
        """
        return f'{len(self.gids)} / {self.shape[0]}'
