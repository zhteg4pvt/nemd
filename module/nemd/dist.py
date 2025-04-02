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


class Cell(np.ndarray):
    """
    Grid box and track atoms.

    FIXME: triclinic support
    """

    def __new__(cls, frm):
        """
        :param frm 'Frame': the distance frame
        """
        dims = cls.getShape(frm.box.span, frm.cut, upper=20)
        obj = np.zeros([*dims, frm.shape[0]], dtype=bool).view(cls)
        obj.frm = frm
        obj.dims = np.array(dims)
        obj.grids = frm.box.span / dims
        obj.nbrs = cls.getNbrs(cls.getShape(obj.grids, frm.cut), dims)
        return obj

    @staticmethod
    def getShape(data, cut, upper=None):
        """
        Get the shape of the data with respect to the frame cut-off.

        :param data iterable: calcualte the shape of this data
        :param cut float: the cut off
        :param upper int: the upper limit of the shape
        :return tuple: data shape
        """
        shape = tuple(math.floor(x / cut) for x in data)
        return tuple(min(x, upper) for x in shape) if upper else shape

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

    def get(self, gid, less=False):
        """
        Get the global atom ids of neighbor (and self) cells.

        :param gid list of int: the global atom ids
        :param gt bool: only include the global atom ids greater than the gid
        :return list of ints: the neighbor atom ids around the coordinates
        """
        cids = self.nbrs[tuple(self.getCids(gid))]
        gids = [self[tuple(x)].nonzero()[0] for x in cids]
        if less:
            gids = [x[x < gid] for x in gids]
        return [y for x in gids for y in x]

    @methodtools.lru_cache()
    @classmethod
    def getNbrs(cls, shape, dims):
        """
        The neighbor cells ids of all cells.

        :param shape tuple: grid shape (number of cutoffs per grid)
        :param dims tuple: dimension shape (number of grids per dimension)
        :return 3x3x3xNx3 numpy.ndarray: the query cell id is 3x3x3 tuple, and
            the return neighbor cell ids are Nx3 numpy.ndarray.
        """
        orig = cls.getOrigNbrs(shape, dims)
        nbrs = np.zeros((*dims, *orig.shape), dtype=int)
        for cid in itertools.product(*[range(x) for x in dims]):
            nbrs[cid] = (orig + cid) % dims
        return nbrs

    @methodtools.lru_cache()
    @classmethod
    def getOrigNbrs(cls, shape, dims):
        """
        The neighbor cells of the (0,0,0) cell without considering the PBC.

        :param shape tuple: grid shape (number of cutoffs per grid)
        :param dims tuple: dimension shape (number of grids per dimension)
        :return nx3 numpy.ndarray: the neighbor cell ids.
        """
        signs = list(itertools.product((-1, 1), (-1, 1), (-1, 1)))
        # Neighbors are cells separation distance <= the cutoff. Adjacent cells
        # are 0 distance separated, and one cell may contain multiple atoms.
        ijks = list(itertools.product(*[range(x + 1) for x in shape]))
        nbrs = np.prod(list(itertools.product(signs, ijks)), axis=1)
        nbrs = np.unique(nbrs, axis=0)
        # Unique neighbor cell ids
        min_cid = np.min(nbrs, axis=0)
        wrapped = (nbrs - min_cid) % dims
        cids = np.zeros([np.max(wrapped) + 1] * 3, dtype=bool)
        cids[tuple(wrapped.transpose())] = True
        return np.array(cids.nonzero()).T + min_cid


class CellNumba(Cell):
    """
    See the parent. (accelerated by numba)
    """

    def set(self, *args, **kwargs):
        """
        See the parent.
        """
        numbautils.set(self, self.grids, self.dims, self.frm, *args, **kwargs)

    def get(self, *args, **kwargs):
        """
        See the parent.
        """
        return numbautils.get(self, self.grids, self.dims, self.frm, self.nbrs,
                              *args, **kwargs)

    @methodtools.lru_cache()
    @classmethod
    def getNbrs(cls, shape, dims):
        """
        See the parent.
        """
        return numbautils.get_nbrs(cls.getOrigNbrs(shape, dims),
                                   np.array(dims))


class Frame(frame.Base):
    """
    Search neighbors and check clashes.
    """
    Cell = Cell if envutils.is_original() else CellNumba

    def __init__(self,
                 *args,
                 gids=None,
                 cut=None,
                 struct=None,
                 srch=None,
                 **kwargs):
        """
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'Reader': radii and excluded pairs
        :param srch: whether to use distance cell to search neighbors
        """
        super().__init__(*args, **kwargs)
        self.gids = numpyutils.IntArray(shape=self.shape[0], on=gids)
        self.cut = cut
        self.struct = struct
        self.srch = srch
        self.cell = None
        if self.cut is None:
            self.cut = self.radii.max()
        if srch is False or self.box is None:
            return
        # Distance > the 1/2 diagonal is not in the first image
        # cut > edge / 2 includes all cells in that direction
        self.cut = min(self.cut, self.box.span.max() / 2)
        if srch is None and not self.useCell(self.box.span, self.cut):
            return
        self.cell = self.Cell(self)
        self.set(self.gids.on)

    @methodtools.lru_cache()
    @property
    def radii(self):
        """
        Get the radii.

        :return `Radius`: the radii
        """
        return Radius(struct=self.struct, num=self.shape[0])

    @staticmethod
    def useCell(span, cut):
        """
        Whether to use the distance cell.

        :param span np.ndarray: the box span
        :param cut float: the cut-off
        :return bool: whether to use the distance cell.
        """
        return np.prod(span) / np.power(cut, 3) >= 1000

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
            return self.cell.get(gid, less=less)
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
        gids = self.gids.difference(self.excluded[gid], on=grp)
        dists = self.box.norms(self[gids, :] - self[gid, :])
        thresholds = self.radii.get(gid, gids)
        return dists[np.nonzero(dists < thresholds)]

    def hasClash(self, gids):
        """
        Whether the selected atoms have clashes with the existing atoms.

        :param gids list: global atom ids for atom selection.
        :return bool: whether the selected atoms have clashes.
        """
        dists = (self.getClash(x, less=False) for x in gids)
        try:
            next(itertools.chain.from_iterable(dists))
        except StopIteration:
            return False
        return True

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
            self.cell.set(gids, state=state)