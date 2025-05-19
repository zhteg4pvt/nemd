# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module provides conformer search using three methods.

1) Grid: separate the conformers by a buffer distance via translation.
2) Pack: pack the conformers without clashes by rotation and translation.
3) Grow: break the molecule into the smallest rigid fragments, place the
  initiators with random rotations and large separation distances, and add back
  the connected fragments after rotating the bonds to avoid clashes.
"""
import collections
import functools
import itertools
import logging
import math
import warnings

import networkx as nx
import numpy as np
import rdkit
import scipy
from rdkit import Chem

from nemd import dist
from nemd import lmpfull
from nemd import logutils
from nemd import numpyutils
from nemd import pbc
from nemd import symbols

logger = logutils.Logger.get(__file__)


class Conf(lmpfull.Conf):
    """
    Customized for coordinate manipulations.
    """

    def centroid(self, weights=None, ignoreHs=False, aids=None):
        """
        Compute the centroid of the whole conformer ar the selected atoms.

        :param weights _vectd: weight the atomic coordinates if provided.
        :param ignoreHs bool: whether to ignore Hs in the calculation.
        :param aids list: the selected atom ids.
        :return np.ndarray: the centroid of the selected atoms.
        """
        if aids is not None:
            on_bits = numpyutils.IntArray(shape=self.GetNumAtoms(), on=aids)
            weights = rdkit.rdBase._vectd()
            weights.extend(on_bits.astype(int).tolist())
        centroid = Chem.rdMolTransforms.ComputeCentroid(self,
                                                        weights=weights,
                                                        ignoreHs=ignoreHs)
        return np.array(centroid)

    def rotate(self, rotation, mtrx=np.identity(4)):
        """
        Rotate the conformer by three initial vectors and three target vectors.

        :param rotation 'np.ndarray': Each row is one initial vector
        :param mtrx 4x4 'np.ndarray': 3D transformation matrix
        """
        mtrx[:-1, :-1] = rotation.as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def translate(self, vec, mtrx=np.eye(4)):
        """
        Do translation on this conformer using this vector.

        :param vec 'np.ndarray': 3D translational vector
        :param mtrx 4x4 'np.ndarray': 3D transformation matrix
        """
        mtrx[:-1, -1] = vec
        Chem.rdMolTransforms.TransformConformer(self, mtrx)


class ConfError(RuntimeError):
    """
    When the last trial failed.
    """
    pass


class PackedConf(Conf):
    """
    Customized to pack without clashes by rotation and translation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oxyz = None

    def setConformer(self, max_trial=1000):
        """
        Place this conformer into the cell without clash.

        :param max_trial int: the max trial number when placing into the cell.
        :raise ConfError: when the last trial failed.
        """
        for point in self.mol.struct.dist.getPoint():
            self.translate(-self.centroid())
            self.rotateRandomly()
            self.translate(point)
            if self.checkClash(self.gids):
                return

        # FIXME: Failed to fill the void with initiator too often
        logger.debug(f'Only {self.mol.struct.dist.ratio} placed')
        raise ConfError

    def checkClash(self, gids, init=False):
        """
        Check the clashes.

        :param gids list: the global atom ids.
        :return bool: True if no clashes are found.
        """
        self.mol.struct.dist[self.gids, :] = self.GetPositions()
        if not self.mol.struct.dist.hasClash(gids):
            self.mol.struct.dist.set(gids, init=init)
            return True

    def rotateRandomly(self, seed=None, high=2**32):
        """
        Randomly rotate the conformer.

        :param seed int: the random seed to generate the rotation matrix.
        :param high int: the exclusive upper limit of the random seed.
        """
        # The random state determines the generated number.
        seed = np.random.randint(0, high) if seed is None else seed % high
        self.rotate(scipy.spatial.transform.Rotation.random(random_state=seed))

    def reset(self):
        """
        Reset the coordinates.
        """
        self.setPositions(self.oxyz)


class GrownConf(PackedConf):

    def __init__(self, *args, **kwargs):
        super(GrownConf, self).__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None
        self.failed_num = 0
        self.frags = []

    def reset(self):
        """
        Rest the attributes that are changed during one grow attempt.
        """
        super().reset()
        self.failed_num = 0
        self.frags = [self.ifrag]
        for frag in self.ifrag.next():
            frag.reset()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments if not, copy to
        current conformer, and set up the fragment objects.
        """
        if self.ifrag:
            return
        mol = self.GetOwningMol()
        self.init_aids = mol.init_aids
        self.ifrag = mol.frag.copy(self)
        self.frags = [self.ifrag]

    def getSwingAtoms(self, *dihe):
        """
        Get the swing atoms when the dihedral angle changes.

        :param dihe list of four ints: the atom ids that form a dihedral angle
        :return list of ints: the swing atom ids when the dihedral angle changes.
        """
        oxyz = self.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(self, *dihe)
        self.setDihedralDeg(dihe, oval + 5)
        xyz = self.GetPositions()
        changed = np.isclose(oxyz, xyz)
        self.setDihedralDeg(dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def setConformer(self, max_trial=5):
        """
        Set the conformer of one fragment by rotating the dihedral angle,
        back moving, and relocation.

        :param max_trial int: the max number of trials for one conformer.
        :raise ConfError: 1) max_trial reached; 2) init cannot be placed.
        """
        frag = self.frags.pop(0)

        try:
            self.frags += frag.setDihedral()
        except ConfError:
            pass
        else:
            # Placed by rotating the bond.
            return

        if self.backMove(frag):
            # Back to a previous frag (some gids removed and fragments added)
            return

        # The relocate the init fragment as the molecule has grown to a dead end
        self.failed_num += 1
        if self.failed_num == max_trial:
            logger.debug(f'Placed {self.mol.struct.dist.ratio} atoms with  '
                         f'conformer {self.gid} reaching the max trials.')
            # FIXME: Failed conformer search should try to reduce clash criteria
            raise ConfError

        self.mol.struct.dist.set(self.gids[self.init_aids], state=False)
        self.ifrag.reset()
        # The method backmove() deletes some existing gids
        self.mol.struct.dist.box.reset()
        self.placeInitFrag()
        self.reportRelocation()

    def backMove(self, frag):
        """
        Back move fragment so that the obstacle can be walked around later.

        :param frag 'fragments.Monomer': fragment to perform back move
        :return bool: True if back move is successful.
        """
        # 1）Find the previous fragment with available dihedral candidates.
        while frag.pfrag and not frag.vals:
            frag = frag.pfrag
        # 2）Find the next fragments who have been placed into the cell.
        nxt_frags = list(frag.next(partial=True))
        [x.reset() for x in nxt_frags[1:]]
        ratom_aids = [y for x in nxt_frags for y in x.aids]
        self.mol.struct.dist.set(self.gids[ratom_aids], state=False)
        # 3）The next fragments of the frag may have been added to the growing
        # self.frags before this backmove step. These added next fragments
        # may have never been growed even once.
        nnxt_frags = [y for x in nxt_frags for y in x.nfrags]
        self.frags = [frag] + list(set(self.frags).difference(nnxt_frags))
        return frag.vals

    def placeInitFrag(self):
        """
        Place the initiator fragment into the cell with random position, random
        orientation, and large separation.

        :raise ValueError: when no void to place the initiator fragment of the
            dead molecule.
        """
        for point in self.mol.struct.dist.getPoint(void=True):
            self.translate(-self.centroid())
            self.rotateRandomly()
            self.translate(point)
            if self.checkClash(self.gids[self.init_aids], init=True):
                return

        # FIXME: Failed to fill the void with initiator too often
        logger.debug(f'Only {self.mol.struct.dist.ratio} placed')
        raise ConfError

    def centroid(self, **kwargs):
        return super().centroid(aids=self.init_aids, **kwargs)

    def reportRelocation(self):
        """
        Report the status after relocate an initiator fragment.
        """
        idists = self.mol.struct.dist.initDists()
        grp = self.gids[self.init_aids]
        other = list(self.mol.struct.dist.gids.diff(grp))
        grps = [other for _ in grp]
        min_dist = self.mol.struct.dist.getDists(grp, grps=grps).min()
        logger.debug(f"Relocate the initiator of {self.gid} conformer "
                     f"(initiator: {idists.min():.2f}-{idists.max():.2f}; "
                     f"close contact: {min_dist:.2f}) ")
        logger.debug(f'{self.mol.struct.dist.ratio} atoms placed.')

    @property
    @functools.cache
    def frag_total(self):
        """
        Return the number of the total fragments.

        :return int: number of the total fragments.
        """
        # ifrag without dihe means rigid body and counts as 1 fragment
        return len(list(self.ifrag.next())) + 1 if self.ifrag.dihe else 1


class GriddedMol(lmpfull.Mol):
    """
    A subclass of Chem.rdchem.Mol to handle gridded conformers.
    """
    Conf = Conf

    def __init__(self, *args, buffer=4, **kwargs):
        """
        :param buffer float: the buffer between conformers.
        """
        super().__init__(*args, **kwargs)
        # size = xyz span + buffer
        self.buffer = np.array([buffer, buffer, buffer])
        # The number of molecules per box edge
        self.conf_num = np.array([1, 1, 1])
        # The xyz shift within one box
        self.vecs = []
        self.vectors = None
        if self.struct and self.struct.options and self.struct.options.buffer:
            self.buffer[:] = self.struct.options.buffer

    def setConformers(self, vectors):
        """
        Place the conformers into boxes based on the shifting vectors.

        :param vectors np.ndarray: the translational vectors to move the
            conformer by multiple boxes distances.
        """
        cids = np.arange(len(self.confs))
        conf_num = np.prod(self.conf_num)
        vecs = self.vecs[cids % conf_num]
        ids = cids // conf_num
        self.vectors = vecs + vectors[ids]
        return vectors[ids[-1] + 1:]

    def setConfNumPerEdge(self, size):
        """
        Set the number of molecules per edge of the box.

        :param size np.ndarray: the box size (the largest molecule size) to
            place this conformer in.
        """
        self.conf_num = np.floor(size / self.size).astype(int)

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        # Grid layout assumes all conformers from one molecule are the same
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

    def setVecs(self, size):
        """
        Set the translational vectors for this conformer so that this conformer
        can be placed in the given box (the largest molecule size).

        :param size np.ndarray: the box size to place this molecule in.
        """
        ptc = [
            np.linspace(-0.5, 0.5, x, endpoint=False) for x in self.conf_num
        ]
        ptc = [x - x.mean() for x in ptc]
        self.vecs = np.array([
            x * size for x in itertools.product(*[[y for y in x] for x in ptc])
        ])

    @property
    def box_num(self):
        """
        Return the number of boxes (the largest molecule size) needed to place
            all conformers.
        """
        return math.ceil(len(self.confs) / np.prod(self.conf_num))


class PackedMol(lmpfull.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """

    Conf = PackedConf

    def updateAll(self):
        """
        Store the original coordinates of all conformers in addition to the
        regular updateAll().
        """
        super().updateAll()
        for conf in self.GetConformers():
            conf.oxyz = conf.GetPositions()


class GrownMol(PackedMol):

    Conf = GrownConf
    POLYM_HT = 'polym_ht'
    MAID = 'maid'
    EDGES = 'edges'

    @property
    @functools.cache
    def frag(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        # dihe is not known and will be handled in setMonomers()
        return Initiator(self.GetConformer())

    @property
    @functools.cache
    def init_aids(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        aids = [y for x in self.frag.next() for y in x.aids]
        return list(set(range(self.gids.shape[0])).difference(aids))

    def getDihes(self, sources=None, targets=None):
        """
        Get a list of dihedral angles.

        :param sources list: source atom ids.
        :param targets list: target atom ids.
        :return list of list: each sublist has four atom ids.
        """
        if sources is None and targets is None:
            sources, targets = self.getHeadTail()
        longest = []
        for source, target in itertools.product(sources, targets):
            _, _, dihes = self.findPath(source=source, target=target)
            dihes = [
                x for x in zip(dihes[:-3], dihes[1:-2], dihes[2:-1], dihes[3:])
                if self.isRotatable(x[1:-1])
            ]
            if len(dihes) > len(longest):
                longest = dihes
        return longest

    def getHeadTail(self):
        """
        If the molecule is built from monomers, the atom pairs from
        selected from the first and last monomers.

        :return tuple: sources and targets to search paths.
        """
        if not self.polym:
            return [None], [None]

        head_tail = [x for x in self.GetAtoms() if x.HasProp(self.POLYM_HT)]
        mono_ids = {x.GetProp(self.MAID): [] for x in head_tail}
        for atom in self.GetAtoms():
            maid = atom.GetProp(self.MAID)
            if maid not in mono_ids:
                continue
            mono_ids[maid].append(atom.GetIdx())

        st_atoms = list(mono_ids.values())
        return st_atoms[0], [y for x in st_atoms[1:] for y in x]

    def findPath(self, source=None, target=None):
        """
        Find the shortest path between source and target. If source and target
        are not provided, the shortest paths between all pairs are computed and
        the long path is returned.

        :param source int: the atom id that serves as the source.
        :param target int: the atom id that serves as the target.
        :return list of ints: the atom ids that form the shortest path.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shortest_path = nx.shortest_path(self.graph,
                                             source=source,
                                             target=target)

        if target is not None:
            shortest_path = {target: shortest_path}
        if source is not None:
            shortest_path = {source: shortest_path}
        path_length, path = -1, None
        for a_source_node, target_path in shortest_path.items():
            for a_target_node, a_path in target_path.items():
                if path_length >= len(a_path):
                    continue
                source_node = a_source_node
                target_node = a_target_node
                path = a_path
                path_length = len(a_path)
        return source_node, target_node, path


class Struct(lmpfull.Struct):

    def __init__(self, *args, density=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.density = density


class GriddedStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    Mol = GriddedMol

    def run(self):
        """
        Set conformers for all molecules.
        """
        self.setVectors()
        self.setBox()
        self.setConformers()
        self.setDensity()

    def setVectors(self):
        """
        Set translational vectors based on the box for all molecules.
        """
        for mol in self.mols:
            mol.setConfNumPerEdge(self.size)
            mol.setVecs(self.size)

    @property
    @functools.cache
    def size(self):
        """
        Return the maximum size over all molecules.

        :return `np.ndarray`: the maximum size.
        """
        return np.array([x.size for x in self.mols]).max(axis=0)

    def setBox(self):
        """
        Set the over-all periodic boundary box.
        """
        total_box_num = sum(x.box_num for x in self.mols)
        edges = self.size * math.ceil(math.pow(total_box_num, 1. / 3))
        self.box = pbc.Box.fromParams(*edges, tilted=False)
        logger.debug(f'Box: {self.box.span.max():.2f} {symbols.ANGSTROM}.')

    def setConformers(self):
        """
        Set coordinates.
        """
        grids = [np.arange(0, x, y) for x, y in zip(self.box.span, self.size)]
        meshgrid = np.meshgrid(*grids, indexing='ij')
        vectors = np.stack(meshgrid, axis=-1).reshape(-1, 3)
        np.random.shuffle(vectors)
        for mol in self.mols:
            vectors = mol.setConformers(vectors)

    def setDensity(self):
        """
        Set the density of the structure.
        """
        vol = self.box.span.prod()
        vol *= math.pow(scipy.constants.centi / scipy.constants.angstrom, 3)
        self.density = self.molecular_weight * scipy.constants.Avogadro / vol

    def GetPositions(self):
        xyzs = [
            x.confs[0].GetPositions() + y for x in self.mols for y in x.vectors
        ]
        return np.concatenate(xyzs, dtype=np.float32)


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedBox(pbc.Box):
    """
    Customized box class for packed structures.
    """

    def getPoint(self):
        """
        Get a random point within the box.

        :return 'pandas.core.series.Series': the random point within the box.
        """
        point = np.random.rand(3) * self.span
        return point + self.lo


class PackedStruct(Struct):
    """
    Pack molecules by random rotation and translation.
    """
    Mol = PackedMol
    Box = PackedBox

    def __init__(self, *args, **kwargs):
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        super().__init__(*args, **kwargs)
        self.dist = None

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        NOTE: the final density of the output cell may be smaller than the
        target if the max number of trial attempt is reached.

        :param density float: the target density
        """
        # self.density is initialized in Struct.__init__() method
        self.density = density
        self.run()

    def run(self):
        """
        Create amorphous cell by randomly placing initiators of the conformers,
        and grow the conformers by adding fragments one by one.
        """
        self.setBox()
        self.setConformers()

    def setBox(self):
        """
        Set periodic boundary box.
        """
        vol = self.molecular_weight / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = self.Box.fromParams(edge, tilted=False)
        logger.debug(f'Cubic box of size {edge:.2f} angstrom is created.')
        self.dist = Frame(self.GetPositions(),
                          box=self.box,
                          struct=self,
                          gids=[])

    def setConformers(self, max_trial=50):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached or the chance of achieving the goal is too low.
        """
        conf_num, placed = self.conf_total, []
        for trial_id in range(1, max_trial + 1):
            with logger.oneLine(logging.DEBUG) as log:
                nth = -1
                for conf_id, conf in enumerate(self.conf):
                    try:
                        conf.setConformer()
                    except ConfError:
                        self.reset()
                        break
                    # One conformer successfully placed
                    if nth != math.floor((conf_id + 1) / conf_num * 10):
                        # Print progress every 10% if conformer number > 10
                        nth = math.floor((conf_id + 1) / conf_num * 10)
                        log(f"{int((conf_id + 1) / conf_num * 100)}%")
                else:
                    # All molecules successfully placed
                    return
            # Current conformer failed
            logger.debug(f'{trial_id} trail fails.')
            logger.debug(f'Only {conf_id} / {conf_num} molecules placed.')
            placed.append(conf_id)
            if not bool(trial_id % int(max_trial / 10)):
                delta = conf_num - np.average(placed)
                std = np.std(placed)
                if not std:
                    raise DensityError
                zscore = abs(delta) / std
                if scipy.stats.norm.cdf(-zscore) * max_trial < 1:
                    # With successful conformer number following norm
                    # distribution, max_trial won't succeed for one time
                    raise DensityError
        self.reset()
        raise DensityError

    @property
    def conf_total(self):
        """
        Return the total number of conformers.

        :return int: the total number of conformers.
        """
        return sum([len(x.confs) for x in self.mols])

    def reset(self):
        """
        Reset the state so that a new attempt can happen.
        """
        for conf in self.conf:
            conf.reset()
        self.dist = Frame(np.concatenate([x.GetPositions()
                                          for x in self.conf]),
                          box=self.box,
                          struct=self,
                          gids=[])
        try:
            self.dist.box.reset()
        except AttributeError:
            pass


class GrownBox(PackedBox):
    """
    Customized box class for grown structures.
    """
    _metadata = ['graph', 'orig_graph', 'cshape', 'cspan']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cshape = None
        self.cspan = None
        self.graph = nx.Graph()
        self.orig_graph = self.graph.copy()

    def setUp(self, conf_num=None):
        """
        Set up the graph.

        :param conf_num int: the number of molecules in the box.
        """
        num = math.ceil(pow(conf_num, 1 / 3)) + 1
        grid = self.span.min() / num
        self.cshape = (self.span / grid).round().astype(int)
        self.cspan = self.span / self.cshape
        nodes = list(itertools.product(*[range(x) for x in self.cshape]))
        self.graph.add_nodes_from(nodes)
        for node in nodes:
            nbrs = (node + self.nbr_inc()) % self.cshape
            for nbr in nbrs:
                self.graph.add_edge(node, tuple(nbr))
        self.orig_graph = self.graph.copy()

    @staticmethod
    @functools.cache
    def nbr_inc(nth=1):
        """
        The nth neighbor cells ids when sitting on the (0,0,0) cell.

        :return nx3 numpy.ndarray: the neighbor cell ids.
        """
        first = math.ceil(nth / 3)
        second = math.ceil((nth - first) / 2)
        third = nth - first - second
        row = np.array([first, second, third])
        data = []
        for signs in itertools.product([-1, 1], [-1, 1], [-1, 1]):
            rows = signs * np.array([x for x in itertools.permutations(row)])
            data.append(np.unique(rows, axis=0))
        return np.unique(np.concatenate(data), axis=0)

    def rmGraphNodes(self, xyz):
        """
        Remove nodes occupied by existing atoms.

        :return nx3 numpy.ndarray: xyz of the atoms whose nodes to be removed.
        """
        if not xyz.size:
            return
        nodes = (xyz / self.cspan).round()
        nodes = [tuple(x) for x in nodes.astype(int)]
        self.graph.remove_nodes_from(nodes)

    def getVoid(self):
        """
        Get the points from the largest void.

        :return `numpy.ndarray`: each row is one random point from the void.
        """
        largest_component = max(nx.connected_components(self.graph), key=len)
        void = np.array(list(largest_component))
        void_max = void.max(axis=0)
        void_span = void_max - void.min(axis=0)
        infinite = (void_span + 1 == self.cshape).any()
        if infinite:
            # The void tunnels the PBC and thus points are uniformly distributed
            yield self.cspan * (np.random.normal(0, 0.5, void.shape) + void)
            return
        # The void is surrounded by atoms and thus the center is preferred
        imap = np.zeros(void_max + 1, dtype=bool)
        imap[tuple(np.transpose(void))] = True
        # FIXME: PBC should be considered
        center = void.mean(axis=0).astype(int)
        max_nth = np.abs(void - center).max(axis=0).sum()
        for nth in range(max_nth):
            nbrs = center + self.nbr_inc(nth=nth)
            nbrs = nbrs[(nbrs <= void_max).all(axis=1).nonzero()]
            nbrs = nbrs[imap[tuple(np.transpose(nbrs))].nonzero()]
            np.random.shuffle(nbrs)
            yield self.cspan * (np.random.normal(0, 0.5, nbrs.shape) + nbrs)

    def reset(self):
        """
        Rest the state.
        """
        self.graph = self.orig_graph.copy()


class Frame(dist.Frame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_gids = []

    def set(self, gids, init=False, **kwargs):
        """
        Add a new atom to the distance cell.

        :param gids list: the global atom ids to be added.
        :param init bool: whether these atoms are from an initiator fragment.
        """
        super().set(gids, **kwargs)
        if not init:
            return
        self.init_gids.append(gids)

    def initDists(self):
        """
        Get the pair distances between existing atoms.

        :return: the pair distances between
        :rtype: 'numpy.ndarray'
        """
        dat = []
        init_gids = sorted(self.init_gids, key=lambda x: x[0])
        for idx in range(1, len(init_gids)):
            grp = init_gids[idx]
            grps = [list(itertools.chain(*init_gids[:idx]))]
            dat.append(self.getDists(grp, grps=grps))
        return np.concatenate(dat)

    def getPoint(self, void=False, max_trial=1000):
        """
        Remove nodes occupied by existing atoms.

        :return `numpy.ndarray`:
        """
        if void:
            self.box.rmGraphNodes(self[self.gids.on])
            return (y for x in self.box.getVoid() for y in x)
        return (self.box.getPoint() for _ in range(max_trial))

    @property
    def ratio(self):
        """
        The ratio of the existing atoms to the total atoms.

        :return str: the ratio of the existing gids with respect to the total.
        """
        return f'{len(self.gids.on)} / {self.shape[0]}'


class GrownStruct(PackedStruct):

    Mol = GrownMol
    Box = GrownBox

    def setUp(self, *args, **kwargs):
        """
        See parent.
        """
        super().setUp(*args, **kwargs)
        for conf in self.conf:
            conf.fragmentize()
        logger.debug(f"Monomer total: {sum(x.frag_total for x in self.conf)}.")

    def setBox(self):
        """
        Set up the box in addition to the parent.
        """
        super().setBox()
        self.box.setUp(self.conf_total)

    def setConformers(self, max_trial=10):
        """
        Looping conformer and set one fragment configuration each time.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: when the max number of trials is reached.
        """
        logger.debug("*" * 10 + f" {self.density} " + "*" * 10)
        for _ in range(max_trial):
            try:
                confs = collections.deque(self.setInits())
            except ConfError:
                self.reset()
                continue
            while confs:
                try:
                    confs[0].setConformer()
                except ConfError:
                    # Reset and try again as this conformer cannot be placed.
                    self.reset()
                    break
                # Successfully set one fragment of this conformer.
                if confs[0].frags:
                    confs.rotate(-1)
                    continue
                # Successfully placed all fragments of one conformer
                confs.popleft()
                logger.debug(f'{self.conf_total - len(confs)} finished; '
                             f'{sum(x.failed_num for x in self.conf)} failed.')
            else:
                # Successfully placed all conformers.
                return
        # Max trial reached at this density.
        self.reset()
        raise DensityError

    def setInits(self):
        """
        Place the initiators into cell.

        :return generator of `GrownConf`: the non-rigid conformer.
        """
        logger.debug(f'Placing {self.conf_total} initiators...')
        with logger.oneLine(logging.DEBUG) as log:
            tenth, threshold, = self.conf_total / 10., 0
            for index, conf in enumerate(self.conf, start=1):
                conf.placeInitFrag()
                if index >= threshold:
                    log(f"{int(index / self.conf_total * 100)}%")
                    threshold = round(threshold + tenth, 1)
                if conf.ifrag.dihe:
                    yield conf

        logger.debug(f'{self.conf_total} initiators have been placed.')
        if self.conf_total == 1:
            return
        logger.debug(f'Minimum separation: {self.dist.initDists().min():.2f}')


class Monomer:
    """
    Dihedral angle controlled fragment.
    """

    def __init__(self, conf, dihe=None, delay=False):
        """
        :param conf 'GrownConf': the conformer that this fragment belongs to
        :param dihe list: the dihedral that changes the swinging atom position.
        :param delay bool: whether to delay the initialization of the fragment.
        """
        self.conf = conf  # Conformer object this fragment belongs to
        self.delay = delay
        self.aids = []  # Atom ids of the swing atoms
        self.nfrags = []  # Next fragments
        self.dihe = dihe  # dihedral angle four-atom ids
        self.pfrag = None  # Previous fragment
        self.ovals = np.linspace(0, 360, 36, endpoint=False)  # Original values
        self.vals = list(self.ovals)  # Available dihedral values candidates
        if self.delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the fragment.
        """
        self.aids = self.conf.getSwingAtoms(*self.dihe)

    def setFrags(self):
        """
        Set next fragments by searching for rotatable bond path.
        """
        while dihes := self.conf.mol.getDihes(self.dihe[1:2], self.aids):
            frags = [self] + [Monomer(self.conf, dihe=x) for x in dihes]
            for frag, nfrag in zip(frags[:-1], frags[1:]):
                frag.aids = sorted(set(frag.aids).difference(nfrag.aids))
                frag.nfrags.append(nfrag)
                nfrag.pfrag = frag

    def setCopies(self, conf):
        """
        Set the nfrags with copoies.

        :param conf GrownConf: the conformer object this fragment belongs to.
        """
        self.nfrags = [x.copy(conf, pfrag=self) for x in self.nfrags]

    def copy(self, conf, pfrag=None, randomize=True):
        """
        Copy the current fragment to a new one.

        :param conf GrownConf: the conformer object this fragment belongs to.
        :param pfrag `Monomer`: the previous fragment.
        :param randomize bool: randomize the dihedral values candidates if True.
        :return Monomer: the copied fragment.
        """
        frag = Monomer(conf, dihe=self.dihe, delay=True)
        frag.pfrag = pfrag
        frag.nfrags = self.nfrags
        frag.aids = self.aids
        frag.vals = self.vals[:]
        if randomize:
            np.random.shuffle(frag.ovals)
            frag.vals = list(frag.ovals)
        return frag

    def next(self, partial=False):
        """
        Return the fragment and fragments following this one.

        :param partial bool: return fragments with partial candidates if True.
        :return generator: the fragment and fragments following this one.
        """
        frags = [self]
        while frags:
            frag = frags.pop()
            if partial and frag.isFull():
                continue
            yield frag
            frags.extend(frag.nfrags)

    def isFull(self):
        """
        Whether the dihedral candidates is the full set.

        :return bool: True if the dihedral candidates is the full set.
        """
        return len(self.vals) == self.ovals.shape[0]

    def reset(self):
        """
        Reset the current dihedral angle candidates.
        """
        self.vals = list(self.ovals)

    def setDihedral(self):
        """
        Set part of the conformer by rotating the dihedral angle.

        :return bool: True if successfully place one fragment.
        """
        while self.vals:
            self.conf.setDihedralDeg(self.dihe, self.vals.pop())
            if self.conf.checkClash(self.conf.gids[self.aids]):
                return self.nfrags
        raise ConfError

    def __str__(self):
        """
        Print the dihedral angle four-atom ids and the swing atom ids.
        """
        return f"{self.dihe}: {self.aids}"


class Initiator(Monomer):
    """
    Initiator fragment.
    """

    def setUp(self):
        """
        See parent.
        """
        # FIXME: the initiator broken by the first rotatable bond may not be
        #   the smallest rigid body. (side-groups contains rotatable bonds)
        self.dihe = next(iter(self.conf.GetOwningMol().getDihes()), None)
        if self.dihe is None:
            # Rigid body
            return
        super().setUp()
        for frag in self.next():
            frag.setFrags()

    def copy(self, conf):
        """
        Copy the current initial fragment and all the fragments retrieved by it.

        :param conf GrownConf: the conformer object this fragment belongs to
        :return Initiator: the copied initial fragment.
        """
        copied = super().copy(conf)
        for frag in copied.next():
            frag.setCopies(conf)
        return copied