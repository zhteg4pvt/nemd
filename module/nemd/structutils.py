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
import functools
import itertools
import math
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import scipy
from scipy.spatial.transform import Rotation

from nemd import box
from nemd import constants
from nemd import dist
from nemd import lmpfull
from nemd import logutils
from nemd import symbols

logger = logutils.get_logger(__file__)


class GriddedConf(lmpfull.Conformer):

    def centroid(self, aids=None, ignoreHs=False):
        """
        Compute the centroid of the whole conformer ar the selected atoms.

        :param atom_ids list: the selected atom ids
        :param ignoreHs bool: whether to ignore Hs in the calculation.
        :return np.ndarray: the centroid of the selected atoms.
        """
        weights = None
        if aids is not None:
            bv = rdkit.DataStructs.ExplicitBitVect(self.GetNumAtoms())
            bv.SetBitsFromList(aids)
            weights = rdkit.rdBase._vectd()
            weights.extend(bv.ToList())
        centroid = rdkit.Chem.rdMolTransforms.ComputeCentroid(
            self, weights=weights, ignoreHs=ignoreHs)
        return np.array([centroid.x, centroid.y, centroid.z])

    def rotate(self, ivect, tvect):
        """
        Rotate the conformer by three initial vectors and three target vectors.

        :param ivect 3x3 'numpy.ndarray': Each row is one initial vector
        :param tvect 3x3 'numpy.ndarray': Each row is one corresponding target
            vector
        """
        mtrx = np.identity(4)
        rotation, _ = Rotation.align_vectors(tvect, ivect)
        mtrx[:-1, :-1] = rotation.as_matrix()
        rdkit.Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def translate(self, vec=None, transform=None):
        """
        Do translation on this conformer using this vector.

        :param vect 'numpy.ndarray': 3D translational vector
        """
        if transform is None:
            transform = constants.EYE4.copy()
        if vec is not None:
            transform[:-1, -1] = vec
        rdkit.Chem.rdMolTransforms.TransformConformer(self, transform)


class ConfError(RuntimeError):
    """
    When max number of the failure for this conformer has been reached.
    """
    pass


class PackedConf(GriddedConf):

    MAX_TRIAL_PER_CONF = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oxyz = None

    def setConformer(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Place molecules one molecule into the cell without clash.

        :param max_trial int: the max trial number for each molecule to be
            placed into the cell.
        :raise ConfError: if the conformer always has clashes with the existing
            atoms in the cell after the maximum trial.
        """
        for _ in range(max_trial):
            centroid = self.centroid()
            self.translate(-centroid)
            self.rotateRandomly()
            pnt = self.mol.struct.dcell.box.getPoint()
            self.translate(pnt)
            self.mol.struct.dcell[self.gids, :] = self.GetPositions()
            if self.hasClash():
                continue
            self.mol.struct.dcell.add(self.gids)
            return
        raise ConfError

    def rotateRandomly(self, seed=None):
        """
        Randomly rotate the conformer.

        NOTE: the random state is set according to the numpy random seed.
        :param seed: the random seed to generate the rotation matrix.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        mtrx = np.identity(4)
        mtrx[:-1, :-1] = Rotation.random(random_state=seed).as_matrix()
        rdkit.Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def hasClash(self, aids=None):
        """
        Whether the conformer has any clashes with the existing atoms.

        :param aids: the conformer atom ids.
        :return bool: True if clashes are found.
        """
        gids = self.id_map[self.aids if aids is None else aids]
        return self.mol.struct.dcell.hasClash(gids)

    def reset(self):
        """
        Reset the cooridinates of the conformer.
        """
        self.setPositions(self.oxyz)


class GrownConf(PackedConf):

    MAX_TRIAL_PER_CONF = 5

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
        if self.ifrag:
            self.ifrag.reset()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments if not, copy to
        current conformer, and set up the fragment objects.
        """
        mol = self.GetOwningMol()
        mol.fragmentize()
        self.init_aids = mol.init_aids.copy()
        self.ifrag = mol.ifrag.copyInit(self)
        self.frags = [self.ifrag]

    def getSwingAtoms(self, *dihe):
        """
        Get the swing atoms when the dihedral angle changes.

        :param dihe list of four ints: the atom ids that form a dihedral angle
        :return list of ints: the swing atom ids when the dihedral angle changes.
        """
        oxyz = self.GetPositions()
        oval = self.getDihedralDeg(dihe)
        self.setDihedralDeg(dihe, oval + 5)
        xyz = self.GetPositions()
        changed = np.isclose(oxyz, xyz)
        self.setDihedralDeg(dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def getDihedralDeg(self, dihe):
        """
        Get the angle degree of the given dihedral.

        :param dihe tuple of int: the dihedral atom indices.
        :param return float: the angle degree.
        """
        return rdkit.Chem.rdMolTransforms.GetDihedralDeg(self, *dihe)

    @functools.cache
    def getNumFrags(self):
        """
        Return the number of the total fragments.

        :return int: number of the total fragments.
        """
        # ifrag without dihe means rigid body and counts as 1 fragment
        return len(self.ifrag.fragments()) + 1 if self.ifrag.dihe else 1

    def placeInitFrag(self):
        """
        Place the initiator fragment into the cell with random position, random
        orientation, and large separation.

        :raise ValueError: when no void to place the initiator fragment of the
            dead molecule.
        """
        for point in self.mol.struct.dcell.getVoid():
            centroid = np.array(self.centroid(aids=self.init_aids))
            self.translate(-centroid)
            self.rotateRandomly()
            self.translate(point)
            self.mol.struct.dcell[self.gids, :] = self.GetPositions()
            if self.hasClash(self.init_aids):
                continue
            self.mol.struct.dcell.add(self.id_map[self.init_aids], init=True)
            return

        # FIXME: Failed to fill the void with initiator too often
        logger.debug(f'Only {self.mol.struct.dcell.ratio} placed')
        raise ConfError

    def setFrag(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Set part of the conformer by rotating the dihedral angle, back moving,
        and relocation.

        :param max_trial int: the max number of trials for one conformer.
        :raise ConfError: if the max number of trials for this conformer is
            reached.
        """
        frag = self.frags.pop(0)
        if not frag.dihe:
            # ifrag without dihe means rigid body
            return

        if self.setDihedral(frag):
            return

        if self.backMove(frag):
            return

        # The molecule has grown to a dead end
        if self.failed_num > max_trial:
            logger.debug(
                f'Placed {self.mol.struct.dcell.ratio} atoms reaching max'
                f' trial number for conformer {self.gid}.')
            raise ConfError
        self.failed_num += 1
        self.ifrag.reset()
        # The method backmove() deletes some existing gids
        self.mol.struct.dcell.box.reset()
        self.placeInitFrag()
        self.reportRelocation()

    def setDihedral(self, frag):
        """
        Set part of the conformer by rotating the dihedral angle.

        :param frag 'fragments.Fragment': fragment to set the dihedral angle.
        :return bool: True if successfully place one fragment.
        """
        while frag.vals:
            frag.setDihedralDeg()
            self.mol.struct.dcell[self.gids, :] = self.GetPositions()
            if self.hasClash(frag.aids):
                continue
            self.mol.struct.dcell.add(self.id_map[frag.aids])
            self.frags += frag.nfrags
            return True

        return False

    def backMove(self, frag):
        """
        Back move fragment so that the obstacle can be walked around later.

        :param frag 'fragments.Fragment': fragment to perform back move
        :return bool: True if back move is successful.
        """
        # 1）Find the previous fragment with available dihedral candidates.
        pfrag = frag.getPreAvailFrag()
        found = bool(pfrag)
        frag = pfrag if found else self.ifrag
        # 2）Find the next fragments who have been placed into the cell.
        nxt_frags = frag.getNxtFrags()
        [x.resetVals() for x in nxt_frags]
        ratom_aids = [y for x in nxt_frags for y in x.aids] + frag.aids
        if not found:
            ratom_aids += frag.conf.init_aids
        self.mol.struct.dcell.remove(self.id_map[ratom_aids])
        # 3）The next fragments of the frag may have been added to the growing
        # self.frags before this backmove step. These added next fragments
        # may have never been growed even once.
        nnxt_frags = [y for x in nxt_frags for y in x.nfrags] + frag.nfrags
        self.frags = [frag] + [x for x in self.frags if x not in nnxt_frags]
        return found

    def reportRelocation(self):
        """
        Report the status after relocate an initiator fragment.
        """
        idists = self.mol.struct.dcell.initPairDists()
        nbrs = self.id_map[self.init_aids]
        min_dist = self.mol.struct.dcell.nbrDists(nbrs=nbrs).min()
        logger.debug(f"Relocate the initiator of {self.gid} conformer "
                     f"(initiator: {idists.min():.2f}-{idists.max():.2f}; "
                     f"close contact: {min_dist:.2f}) ")
        logger.debug(f'{self.mol.struct.dcell.ratio} atoms placed.')


class GriddedMol(lmpfull.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """

    ConfClass = GriddedConf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # size = xyz span + buffer
        buffer = self.struct.options.buffer if self.struct.options.buffer else 4
        self.buffer = np.array([buffer, buffer, buffer])
        # The number of molecules per box edge
        self.conf_num = np.array([1, 1, 1])
        # The xyz shift within one box
        self.vecs = []
        self.vectors = None

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

    def GetPositions(self):
        xyz = self.confs[0].GetPositions()
        return np.concatenate([xyz + x for x in self.vectors],
                              dtype=np.float32)

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
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    ConfClass = PackedConf

    def updateAll(self):
        """
        Store the original coordinates of all conformers in addition to the
        regular updateAll().
        """
        super().updateAll()
        for conf in self.GetConformers():
            conf.oxyz = conf.GetPositions()


class GrownMol(PackedMol):

    ConfClass = GrownConf
    POLYM_HT = 'polym_ht'
    MAID = 'maid'
    EDGES = 'edges'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        if self.ifrag is not None:
            return
        # dihe is not known and will be handled in setFragments()
        self.ifrag = Fragment(self.GetConformer())
        frags = self.ifrag.fragments()
        frag_aids_set = set([y for x in frags for y in x.aids])
        all_aids = set([x.GetIdx() for x in self.GetAtoms()])
        self.init_aids = list(all_aids.difference(frag_aids_set))

    def findHeadTailPair(self):
        """
        If the molecule is built from monomers, the atom pairs from
        selected from the first and last monomers.

        :return list or iterator of int tuple: each tuple is an atom id pair
        """
        if not self.is_polym:
            return [(None, None)]

        head_tail = [x for x in self.GetAtoms() if x.HasProp(self.POLYM_HT)]
        mono_ids = {x.GetProp(self.MAID): [] for x in head_tail}
        for atom in self.GetAtoms():
            maid = atom.GetProp(self.MAID)
            if maid not in mono_ids:
                continue
            mono_ids[maid].append(atom.GetIdx())

        st_atoms = list(mono_ids.values())
        sources = st_atoms[0]
        targets = [y for x in st_atoms[1:] for y in x]

        return itertools.product(sources, targets)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density = None
        self.box = None


class GriddedStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    MolClass = GriddedMol

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = np.zeros([3])

    def run(self):
        """
        Set conformers for all molecules.
        """
        self.setSize()
        self.setVectors()
        self.setBox()
        self.setConformers()
        self.setDensity()

    def setSize(self):
        """
        Set the size as the maximum size over all molecules.
        """
        self.size = np.array([x.size for x in self.molecules]).max(axis=0)

    def setVectors(self):
        """
        Set translational vectors based on the box for all molecules.
        """
        for mol in self.molecules:
            mol.setConfNumPerEdge(self.size)
            mol.setVecs(self.size)

    def setBox(self):
        """
        Set the over-all periodic boundary box.
        """
        # vectors shifts molecules by the largest box size
        total_box_num = sum(x.box_num for x in self.molecules)
        edges = self.size * math.ceil(math.pow(total_box_num, 1. / 3))
        self.box = box.Box.fromEdges(edges)
        span = self.box.span.max()
        logger.debug(f'Cubic box of {span:.2f} {symbols.ANGSTROM} is created.')

    def setConformers(self):
        """
        Set coordinates.
        """
        # idxes = (self.box.span / self.size).round().astype(int)
        # idxes = [list(range(x)) for x in idxes]
        # vectors = np.array([x * self.size for x in itertools.product(*idxes)])
        grids = [np.arange(0, x, y) for x, y in zip(self.box.span, self.size)]
        # vectors shifts molecules by the largest box size
        meshgrid = np.meshgrid(*grids, indexing='ij')
        vectors = np.stack(meshgrid, axis=-1).reshape(-1, 3)
        np.random.shuffle(vectors)
        for mol in self.molecules:
            vectors = mol.setConformers(vectors)

    def setDensity(self):
        """
        Set the density of the structure.
        """
        vol = self.box.span.prod()
        vol *= math.pow(scipy.constants.centi / scipy.constants.angstrom, 3)
        self.density = self.mw * scipy.constants.Avogadro / vol


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedCell(dist.Cell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig = self.copy()
        self.orig_gids = self.gids.copy()
        self.orig_cell = None

    def setCell(self):
        """
        Put atom ids into the corresponding cells.

        self.atom_cell.shape = [X index, Y index, Z index, all atom ids]
        """
        super().setCell()
        self.orig_cell = self.cell.copy()

    def reset(self):
        """
        Reset the distance cell.
        """
        self[:] = self.orig.copy()
        self.gids = self.orig_gids.copy()
        self.cell = self.orig_cell.copy()


class PackedBox(box.Box):
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

    MolClass = PackedMol
    MAX_TRIAL_PER_DENSITY = 50
    Cell = PackedCell
    Box = PackedBox

    def __init__(self, *args, **kwargs):
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        super().__init__(*args, **kwargs)
        self.dcell = None

    def finalize(self):
        """
        Finalize the structure.
        """
        super().finalize()
        self.setCell()

    def setCell(self, **kwargs):
        """
        Set the distance cell with the trajectory frame.
        """
        data = np.concatenate([x.GetPositions() for x in self.conformer])
        self.dcell = self.Cell(data=data, gids=set(), struct=self, **kwargs)

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
        vol = self.mw / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = self.Box.fromEdges([edge] * 3)
        logger.debug(f'Cubic box of size {edge:.2f} angstrom is created.')
        self.dcell.setup(self.box)

    def setConformers(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached or the chance of achieving the goal is too low.
        """
        trial_id, conf_num, finished, nth = 1, self.conformer_total, [], -1
        for trial_id in range(1, max_trial + 1):
            for conf_id, conf in enumerate(self.conformer):
                try:
                    conf.setConformer()
                except ConfError:
                    self.reset()
                    break
                # One conformer successfully placed
                if nth != math.floor((conf_id + 1) / conf_num * 10):
                    # Print progress every 10% if conformer number > 10
                    nth = math.floor((conf_id + 1) / conf_num * 10)
                    pct = f"{int((conf_id + 1) / conf_num * 100)}%"
                    logger.debug(pct, newline=pct == "100%")
                # Whether all molecules successfully placed
                if conf_id == conf_num - 1:
                    return
            # Current conformer failed
            logger.debug(f'{trial_id} trail fails.')
            logger.debug(f'Only {conf_id} / {conf_num} molecules placed.')
            finished.append(conf_id)
            if not bool(trial_id % int(max_trial / 10)):
                delta = conf_num - np.average(finished)
                std = np.std(finished)
                if not std:
                    raise DensityError
                zscore = abs(delta) / std
                if scipy.stats.norm.cdf(-zscore) * max_trial < 1:
                    # With successful conformer number following norm
                    # distribution, max_trial won't succeed for one time
                    raise DensityError
        self.reset()
        raise DensityError

    def reset(self):
        """
        Reset the state so that a new attempt can happen.
        """
        for conf in self.conformer:
            conf.reset()
        self.dcell.reset()


class GrownBox(PackedBox):
    """
    Customized box class for grown structures.
    """

    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + [
        'graph', 'orig_graph', 'cshape', 'cspan'
    ]
    _internal_names_set = set(_internal_names)

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


class GrownCell(PackedCell):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_gids = []

    def add(self, gids, init=False):
        """
        Add a new atom to the distance cell.

        :param gids list: the global atom ids to be added.
        :param init bool: whether these atoms are from an initiator fragment.
        """
        super().add(gids)
        if not init:
            return
        self.init_gids.append(gids)

    def initPairDists(self):
        """
        Get the pair distances between existing atoms.

        :return: the pair distances between
        :rtype: 'numpy.ndarray'
        """
        dat = []
        init_gids = sorted(self.init_gids, key=lambda x: x[0])
        for idx in range(1, len(init_gids)):
            grp1 = init_gids[idx]
            grp2 = [list(itertools.chain(*init_gids[:idx]))]
            dat.append(self.pairDists(grp1=grp1, grp2=grp2))
        return np.concatenate(dat)

    def reset(self):
        """
        Reset the object.
        """
        super().reset()
        self.box.reset()
        self.init_gids = []

    def getVoid(self):
        """
        Remove nodes occupied by existing atoms.

        :return `numpy.ndarray`: each value is one random point from the voids.
        """
        self.box.rmGraphNodes(self[list(self.gids), :])
        return (y for x in self.box.getVoid() for y in x)


class GrownStruct(PackedStruct):

    MolClass = GrownMol
    MAX_TRIAL_PER_DENSITY = 10
    Cell = GrownCell
    Box = GrownBox

    def finalize(self):
        """
        See parent class.
        """
        super().finalize()
        self.fragmentize()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        if all([x.ifrag for x in self.conformer]):
            return

        for conf in self.conformer:
            conf.fragmentize()
        total_frag_num = sum([x.getNumFrags() for x in self.conformer])
        logger.debug(f"{total_frag_num} fragments in total.")

    def setBox(self):
        """
        Set up the box in addition to the parent.
        """
        super().setBox()
        self.box.setUp(self.conformer_total)

    def placeInitFrags(self):
        """
        Place the initiators into cell.
        """
        logger.debug(f'Placing {self.conformer_total} initiators...')

        tenth, threshold, = self.conformer_total / 10., 0
        for index, conf in enumerate(self.conformer, start=1):
            conf.placeInitFrag()
            if index >= threshold:
                msg = f"{int(index / self.conformer_total * 100)}%"
                logger.debug(msg, newline=index == self.conformer_total)
                threshold = round(threshold + tenth, 1)

        logger.debug(f'{self.conformer_total} initiators have been placed.')
        if self.conformer_total == 1:
            return
        dist = self.dcell.initPairDists().min()
        logger.debug(f'({dist:.2f} as the minimum pair distance)')

    def setConformers(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Looping conformer one by one and set one fragment configuration each
        time until all to full length.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        logger.debug("*" * 10 + f" {self.density} " + "*" * 10)
        for _ in range(max_trial):
            try:
                self.placeInitFrags()
            except ConfError:
                self.reset()
                continue
            conformers = list(self.conformer)
            while conformers:
                conf = conformers.pop(0)
                try:
                    conf.setFrag()
                except ConfError:
                    # Reset and try again as this conformer cannot be placed.
                    # 1 ) ifrag cannot be place; 2 ) failed_num reached maximum
                    self.reset()
                    break
                # Successfully set one fragment of this conformer.
                if conf.frags:
                    # The conformer has more fragments to grow.
                    conformers.append(conf)
                    continue
                # Successfully placed all fragments of one conformer
                finished_num = self.conformer_total - len(conformers)
                failed_num = sum([x.failed_num for x in self.conformer])
                logger.debug(f'{finished_num} finished; {failed_num} failed.')
                if not conformers:
                    # Successfully placed all conformers.
                    return
        # Max trial reached at this density.
        self.reset()
        raise DensityError


class Fragment:
    """
    Class to set a portion of the conformer by rotating the dihedral angle.
    """

    def __str__(self):
        """
        Print the dihedral angle four-atom ids and the swing atom ids.
        """
        return f"{self.dihe}: {self.aids}"

    def __init__(self, conf, dihe=None, delay=False):
        """
        :param conf 'GrownConf': the conformer that this fragment belongs to
        :param dihe list of dihedral atom ids: the dihedral that changes the
            atom position in this fragment.
        :param delay bool: whether to delay the initialization of the fragment.
        """
        self.conf = conf  # Conformer object this fragment belongs to
        self.dihe = dihe  # dihedral angle four-atom ids
        self.aids = []  # Atom ids of the swing atoms
        self.pfrag = None  # Previous fragment
        self.nfrags = []  # Next fragments
        self.ovals = np.linspace(0, 360, 36, endpoint=False)  # shuffle on copy
        self.vals = list(self.ovals)  # Available dihedral values candidates
        self.val = None  # Chosen dihedral angle value
        self.fval = True  # All dihedral values are available (new frag)
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the fragment.
        """
        if self.dihe is not None:
            self.aids = self.conf.getSwingAtoms(*self.dihe)
            return
        self.setFragments()

    def resetVals(self):
        """
        Reset the dihedral angle values and state.
        """
        self.val, self.fval = None, True
        self.vals = list(self.ovals)

    def reset(self):
        """
        Reset the state of all fragments.
        """
        for frag in self.fragments():
            frag.resetVals()

    def getOwningMol(self):
        """
        Get the owning GrownMol object.

        :return GrownMol: get the molecule this fragment belongs to.
        """
        return self.conf.GetOwningMol()

    def copyInit(self, conf):
        """
        Copy the current initial fragment and all the fragments retrieved by it.
        The connections between all new fragments are established as well.

        :param mol GrownConf: the conformer object this initial fragment belongs to
        :return Fragment: the copied initial fragment.
        """
        ifrag = self.copy(conf)
        all_nfrags = [ifrag]
        while all_nfrags:
            frag = all_nfrags.pop()
            nfrags = [x.copy(conf) for x in frag.nfrags]
            frag.nfrags = nfrags
            for nfrag in nfrags:
                nfrag.pfrag = frag
            all_nfrags += nfrags
        return ifrag

    def copy(self, conf, randomize=True):
        """
        Copy the current fragment to a new one.

        :param conf GrownConf: the conformer object this fragment belongs to.
        :param randomize bool: randomize the dihedral values candidates if True.
        :return Fragment: the copied fragment.
        """
        frag = Fragment(conf, dihe=self.dihe, delay=True)
        frag.aids = self.aids
        frag.pfrag = self.pfrag
        frag.nfrags = self.nfrags
        frag.val = self.val
        frag.fval = self.fval
        # Another conformer may have different value and candidates
        frag.ovals = self.ovals.copy()
        frag.vals = self.vals[:]
        if randomize:
            np.random.shuffle(frag.ovals)
            frag.vals = list(frag.ovals)
        return frag

    def setFragments(self):
        """
        Set up fragments by iteratively searching for rotatable bond path and
        adding them as next fragments.
        """
        # Finish up the ifrag setup and set next fragments
        frags = self.setNextFrags()
        # Set next fragments
        while frags:
            frag = frags.pop(0)
            nfrags = frag.setNextFrags()
            frags += nfrags
        # Set previous fragments
        for frag in self.fragments():
            for nfrag in frag.nfrags:
                nfrag.pfrag = frag

    def setNextFrags(self):
        """
        Set fragments by searching for rotatable bond path and adding them as
        next fragments.

        FIXME: the initiator fragment broken by the first rotatable bond in the
            longest path may not be the smallest rigid fragment. (side-groups
            may contains rotatable bonds)

        :return list of 'Fragment': fragment self and newly added fragments.
            (the atom ids of the current fragments changed)
        """
        dihes = self.getNewDihes()
        if not dihes:
            # This removes self frag out of the to_be_fragmentized list
            return []
        if not self.dihe:
            # This is an initial fragment with unknown dihedral angle
            self.dihe = dihes.pop(0)
            self.setUp()
        frags = [self] + [Fragment(self.conf, dihe=x) for x in dihes]
        for frag, nfrag in zip(frags[:-1], frags[1:]):
            frag.aids = sorted(set(frag.aids).difference(nfrag.aids))
            frag.nfrags.append(nfrag)
        return frags

    def getNewDihes(self):
        """
        Get a list of dihedral angles that travel along the fragment rotatable
        bond to one fragment atom ids so that the path has the most rotatable
        bonds.

        NOTE: If the current dihedral angle of this fragment is not set, the
        dihedral angle list travel along the path with the most rotatable bonds.

        :return list of list: each sublist has four atom ids.
        """
        if self.dihe:
            pairs = zip([self.dihe[1]] * len(self.aids), self.aids)
        else:
            pairs = self.getOwningMol().findHeadTailPair()

        dihes, num = [], 0
        for source, target in pairs:
            mol = self.getOwningMol()
            _, _, path = mol.findPath(source=source, target=target)
            a_dihes = zip(path[:-3], path[1:-2], path[2:-1], path[3:])
            a_dihes = [x for x in a_dihes if mol.isRotatable(x[1:-1])]
            if len(a_dihes) < num:
                continue
            num = len(a_dihes)
            dihes = a_dihes
        return dihes

    def setDihedralDeg(self, val=None):
        """
        Set the dihedral angle of the current fragment. If val is not provided,
        randomly choose one from the candidates.

        :param val float: the dihedral angle value to be set.
        """
        if val is None:
            val = self.vals.pop()
            self.fval = False
        self.val = val
        self.conf.setDihedralDeg(self.dihe, self.val)

    def getPreAvailFrag(self):
        """
        Get the first previous fragment that has available dihedral candidates.

        :return 'Fragment': The previous fragment found.
        """
        frag = self.pfrag
        if frag is None:
            return None
        while frag.pfrag and not frag.vals:
            frag = frag.pfrag
        if frag.pfrag is None and not frag.vals:
            # FIXME: Failed conformer search should try to reduce clash criteria
            return None
        return frag

    def getNxtFrags(self):
        """
        Get the next fragments that don't have full dihedral value candidates.

        :return list: list of next fragments
        """
        frags, nfrags = [], [self]
        while nfrags:
            nfrags = [y for x in nfrags for y in x.nfrags if not y.fval]
            frags += nfrags
        return frags

    def fragments(self):
        """
        Return all fragments.

        :return list of Fragment: all fragment of this conformer.
        """
        frags, nfrags = [], [self]
        while nfrags:
            frags += nfrags
            nfrags = [y for x in nfrags for y in x.nfrags]
        return frags


class GridConfWrapper(lmpfull.Conformer):

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        self.start = None
        self.gid = 0

    def setUp(self, *args, cid=0, start=None):
        self.gid = cid
        self.start = start

    def GetOwningMol(self):
        return self.conf.GetOwningMol()

    def setBondLength(self, *args, **kwargs):
        self.conf.setBondLength(*args, **kwargs)

    def setAngleDeg(self, *args, **kwargs):
        self.conf.setAngleDeg(*args, **kwargs)

    def GetPositions(self, *args, **kwargs):
        return self.conf.GetPositions(*args, **kwargs)

    def setPositions(self, *args, **kwargs):
        return self.conf.setPositions(*args, **kwargs)


class GriddedMol2(GriddedMol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """

    ConfWrapper = GridConfWrapper


class GriddedStruct2(GriddedStruct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    MolClass = GriddedMol2
