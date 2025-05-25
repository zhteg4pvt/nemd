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

        :param rotation 'scipy.spatial.transform.Rotation': Rotation in 3D.
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

    def setConformer(self, gids=None):
        """
        Place this conformer into the cell without clash.

        :raise ConfError: when the last trial failed.
        """
        if gids is None:
            gids = self.gids
        for point in self.mol.struct.dist.getPoints():
            self.translate(-self.centroid())
            self.rotateRandomly()
            self.translate(point)
            if self.checkClash(gids):
                return
        raise ConfError

    def checkClash(self, gids):
        """
        Check the clashes.

        :param gids list: the global atom ids.
        :return bool: True if clashes not found.
        """
        self.mol.struct.dist[self.gids, :] = self.GetPositions()
        if not self.mol.struct.dist.hasClash(gids):
            self.mol.struct.dist.set(gids)
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
        Reset.
        """
        self.setPositions(self.oxyz)


class GrownConf(PackedConf):
    """
    Customized for fragments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init = self.gids
        self.failed = 0
        self.frag = None
        self.frags = []

    def setUp(self):
        """
        Set Up.
        """
        if not self.mol.frag:
            return
        self.init = self.gids[self.mol.init]
        self.frag = self.mol.frag.new(self)
        self.frags = [self.frag]

    def grow(self, max_trial=5):
        """
        Grow the conformer by bond rotation, back move, and relocation.

        :param max_trial int: the max number of trials for one conformer.
        :raise ConfError: 1) max_trial reached; 2) init cannot be placed.
        """
        frag = self.frags.pop(0)
        while frag.vals:
            self.setDihedralDeg(frag.dihe, frag.vals.pop())
            if self.checkClash(frag.ids):
                self.frags += frag.nfrags
                return

        # 1）Find the previous fragment with available dihedral candidates.
        while frag.pfrag and not frag.vals:
            frag = frag.pfrag
        # 2）Find the next fragments who have been placed into the cell.
        frags = list(frag.next(partial=True))
        [x.reset() for x in frags[1:]]
        self.mol.struct.dist.set([y for x in frags for y in x.ids], False)
        # 3）The next fragments may have been added to the growing list
        nfrags = [y for x in frags for y in x.nfrags]
        self.frags = [frag] + list(set(self.frags).difference(nfrags))
        if bool(frag.vals):
            return

        # The relocate the init fragment as the molecule has grown to a dead end
        self.failed += 1
        if self.failed == max_trial:
            raise ConfError

        self.frag.reset()
        self.mol.struct.dist.set(self.init, state=False)
        self.setConformer()
        logger.debug(
            f"Initiator {self.gid} is relocated "
            f"{self.mol.struct.dist.getDists(self.init).min():.2f} separated. "
            f"({self.mol.struct.dist.getDists().min():.2f} to initiators)")

    def setConformer(self, **kwargs):
        """
        Place the initiator fragment into the cell without clashes.
        """
        # FIXME: the largest void center should have higher priority
        #  especially importantly when relocating a dead molecule.
        super().setConformer(self.init)

    def centroid(self, **kwargs):
        """
        See parent.
        """
        return super().centroid(aids=self.mol.init, **kwargs)

    def getSwingAtoms(self, dihe):
        """
        Get the atoms that change the positions on dihedral angle rotation.

        :param dihe list: dihedral angle atom ids.
        :return list: list of swing atom ids.
        """
        oxyz = self.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(self, *dihe)
        self.setDihedralDeg(dihe, oval + 5)
        changed = ~np.isclose(oxyz, self.GetPositions()).all(axis=1)
        self.setDihedralDeg(dihe, oval)
        return self.mol.aids[changed.nonzero()].tolist()

    def reset(self):
        """
        Rest the attributes that are changed during one grow attempt.
        """
        super().reset()
        self.failed = 0
        if not self.frag:
            return
        for frag in self.frag.next():
            frag.reset()
        self.frags = [self.frag]


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
        self.buffer = np.array([buffer, buffer, buffer])
        self.num = np.array([1, 1, 1])  # Conformer number per box edge
        self.vecs = []  # Translational vectors with in the box
        self.vectors = None  # Translational vectors with in the pbc
        if self.struct and self.struct.options and self.struct.options.buffer:
            self.buffer[:] = self.struct.options.buffer

    def run(self, size):
        """
        Set the conformer number per edge and within-box translational vectors.

        :param size np.ndarray: the box size to place this molecule in.
        """
        self.num = np.floor(size / self.size).astype(int)
        ptc = [np.linspace(-0.5, 0.5, x, endpoint=False) for x in self.num]
        ptc = [x - x.mean() for x in ptc]
        self.vecs = list(itertools.product(*[[y for y in x] for x in ptc]))
        self.vecs *= size

    def setConformers(self, vecs):
        """
        Place the conformers into boxes based on the shifting vectors.

        :param vecs np.ndarray: the translational vectors to move the conformer.
        """
        cids = np.arange(len(self.confs))
        conf_total = np.prod(self.num)
        ids = cids // conf_total
        self.vectors = self.vecs[cids % conf_total] + vecs[ids]
        return vecs[ids[-1] + 1:]

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        # Assumes all conformers from one molecule are the same
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

    @property
    def box_num(self):
        """
        Return the number of boxes needed to place all conformers.
        """
        return math.ceil(len(self.confs) / np.prod(self.num))


class PackedMol(lmpfull.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """
    Conf = PackedConf

    def updateAll(self):
        """
        See parent.
        """
        super().updateAll()
        for conf in self.GetConformers():
            conf.oxyz = conf.GetPositions()


class GrownMol(PackedMol):
    """
    Customized for fragments.
    """
    Conf = GrownConf
    POLYM_HT = 'polym_ht'
    MAID = 'maid'

    def setUp(self, *args, **kwargs):
        """
        See parent.
        """
        super().setUp(*args, **kwargs)
        for conf in self.confs:
            conf.setUp()

    def shift(self, pre):
        """
        See parent.
        """
        super().shift(pre)
        if pre is None or self.frag is None:
            return
        start = pre.gids.max() + 1
        for conf in self.confs:
            conf.init += start
            for frag in conf.frag.next():
                frag.ids += start

    @property
    @functools.cache
    def frag(self):
        """
        Return the first fragment connected to the initiator.
        """
        dihes = self.getDihes()
        if dihes:
            # FIXME: the initiator broken by the first rotatable bond may not be
            #  the smallest rigid body. (side-groups contains rotatable bonds)
            return First(self.confs[0], dihes[0])

    @property
    @functools.cache
    def init(self):
        """
        Return the initiator atom aids.

        :return list: the initiator atom aids.
        """
        aids = numpyutils.IntArray(on=self.aids)
        if self.frag:
            aids[[y for x in self.frag.next() for y in x.ids]] = False
        return aids.on

    def getDihes(self, sources=(None, ), targets=(None, )):
        """
        Get a list of dihedral angles.

        :param sources list: source atom ids.
        :param targets list: target atom ids.
        :return list: each sublist has four atom ids.
        """
        if sources[0] is None and self.polym:
            # FIXME: sources should be all initiator atoms; targets should be
            #  the atoms of all terminators
            polym_ht = [x for x in self.GetAtoms() if x.HasProp(self.POLYM_HT)]
            sources, targets = [[x.GetIdx()] for x in polym_ht]

        longest = []
        for source, target in itertools.product(sources, targets):
            dihes = self.findPath(source=source, target=target)
            dihes = [
                x for x in zip(dihes[:-3], dihes[1:-2], dihes[2:-1], dihes[3:])
                if self.isRotatable(x[1:-1])
            ]
            if len(dihes) > len(longest):
                longest = dihes
        return longest

    def findPath(self, source=None, target=None):
        """
        Return the shortest path if source and target provided else the longest
        of all shortest paths between atom pairs.

        :param source int: the atom id that serves as the source.
        :param target int: the atom id that serves as the target.
        :return list: atom ids that form the longest shortest path.
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
        length = -1
        for a_source, target_path in shortest_path.items():
            for a_target, a_path in target_path.items():
                if len(a_path) > length:
                    length = len(a_path)
                    source = a_source
                    target = a_target
        return shortest_path[source][target]


class Struct(lmpfull.Struct):
    """
    Customized with density.
    """

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
        self.setVecs()
        self.setBox()
        self.setConformers()
        self.setDensity()

    def setVecs(self):
        """
        Set translational vectors within the box.
        """
        for mol in self.mols:
            mol.run(self.size)

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
        box_total = sum(x.box_num for x in self.mols)
        edges = self.size * math.ceil(math.pow(box_total, 1. / 3))
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
        """
        Get the atom positions.

        :return `np.ndarray`: the coordinates
        """
        xyzs = [x.confs[0].GetPositions() for x in self.mols]
        xyzs = [x + z for x, y in zip(xyzs, self.mols) for z in y.vectors]
        return np.concatenate(xyzs, dtype=np.float32)


class Box(pbc.Box):
    """
    Customized box class for packed structures.
    """

    def getPoints(self, size=1000):
        """
        Get randomized points.

        :param size int: the number of points.
        :return `np.ndarray`: each row is a point.
        """
        return np.random.rand(size, 3) * self.span + self.lo.values


class PackFrame(dist.Frame):
    """
    Customized for packing.
    """

    def getPoints(self):
        """
        Get randomized points.

        :return `np.ndarray`: each row is a point.
        """
        if self.cell is None:
            return self.box.getPoints()
        nodes = ~self.cell.cell.any(axis=3)
        nodes = np.array(nodes.nonzero()).transpose()
        np.random.shuffle(nodes)
        randomized = nodes + np.random.normal(0, 0.5, nodes.shape)
        return randomized * self.cell.grids


class GrownFrame(PackFrame):
    """
    Customized for fragments.
    """

    def getDists(self, grp=None):
        """
        Get the distances to the initiator.

        :param grp 'np.ndarray': the initiator global atom ids.
        :return 'np.ndarray': the pair distances between gid groups.
        """
        if grp is not None:
            other = list(self.gids.diff(grp))
            return super().getDists(grp, grps=[other for _ in grp])

        grps = [y.init for x in self.struct.mols for y in x.confs]
        pairs = ([grps[i], grps[:i]] for i in range(1, len(grps)))
        return np.concatenate([
            super(GrownFrame, self).getDists(x, grps=[np.concatenate(y)])
            for x, y in pairs
        ])


class PackedStruct(Struct):
    """
    Pack molecules by random rotation and translation.
    """
    Mol = PackedMol
    Frame = PackFrame

    def __init__(self, *args, **kwargs):
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        super().__init__(*args, **kwargs)
        self.dist = None
        self.placed = []

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        :param density float: the target density.
        :return bool: True if successfully set.
        """
        # The density will be reduced when the attempt exceeds the max trial.
        self.density = density
        self.setBox()
        self.setFrame()
        return self.setConformers()

    def setBox(self):
        """
        Set periodic boundary box.
        """
        vol = self.molecular_weight / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = Box.fromParams(edge, tilted=False)
        logger.debug(f'Cubic box of size {edge:.2f} angstrom is created.')

    def setFrame(self):
        """
        Set the distance frame.
        """
        self.dist = self.Frame(self.GetPositions(),
                               gids=[],
                               box=self.box,
                               struct=self)

    def setConformers(self):
        """
        Place all molecules into the cell at certain density.

        :return bool: True if successfully set.
        """
        logger.debug("*" * 10 + f" {self.density} " + "*" * 10)
        while self.isPossible():
            self.attempt()
            if self.placed[-1] == self.conf_total:
                return True
            logger.debug(
                f'Trial {len(self.placed)}: {self.placed[-1]} placed.')
        self.placed = []

    def attempt(self):
        """
        One attempt on setting the conformers.
        """
        with logger.oneLine(logging.DEBUG) as log:
            tenth, threshold, = self.conf_total / 10., 0
            for index, conf in enumerate(self.conf):
                try:
                    conf.setConformer()
                except ConfError:
                    # FIXME: Failed to fill the void with initiator too often
                    self.reset()
                    self.placed.append(index)
                    return
                # One conformer successfully placed
                if index >= threshold:
                    log(f"{int(index / self.conf_total * 100)}%")
                    threshold = round(threshold + tenth, 1)
            self.placed.append(self.conf_total)

    def isPossible(self, intvl=5):
        """
        Whether further attempt is statistically possible.

        :param intvl int: the interval to check possibility.
        :return bool: whether it is statistically possible.
        """
        if not self.placed or len(self.placed) % intvl != 0:
            return True
        std = np.std(self.placed)
        if std:
            zscore = abs(self.conf_total - np.average(self.placed)) / std
            return scipy.stats.norm.cdf(-zscore) > 0.05

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
        self.setFrame()


class GrownStruct(PackedStruct):
    """
    Grow the packed initiators by rotating the bonds to avoid clashes.
    """
    Mol = GrownMol
    Frame = GrownFrame

    def attempt(self):
        """
        See parent.
        """
        logger.debug(f'Placing {self.conf_total} initiators...')
        super().attempt()
        if self.placed[-1] != self.conf_total:
            return
        logger.debug(f'{self.conf_total} initiators have been placed.')
        if self.conf_total != 1:
            logger.debug(f'Closest contact: {self.dist.getDists().min():.2f}')
        confs = collections.deque([x for x in self.conf if x.frag])
        while confs:
            try:
                confs[0].grow()
            except ConfError:
                # FIXME: Failed attempt should try to reduce clash criteria
                self.reset()
                self.placed[-1] = self.conf_total - len(confs)
                return
            # Successfully set one fragment.
            if confs[0].frags:
                confs.rotate(-1)
                continue
            confs.popleft()
            logger.debug(f'{self.conf_total - len(confs)} finished.')


class Fragment:
    """
    Dihedral angle controlled fragment.
    """
    NUM = 36

    def __init__(self, conf, dihe, delay=False):
        """
        :param conf 'GrownConf': the conformer that this fragment belongs to
        :param dihe list: the dihedral that changes the swinging atom position.
        :param delay bool: whether to delay the initialization of the fragment.
        """
        self.conf = conf
        self.dihe = dihe
        self.delay = delay
        self.ids = None  # aid (molecule) or gids (conformer) of the swing atoms
        self.pfrag = None  # Previous fragment
        self.nfrags = []  # Next fragments
        self.ovals = None  # Original dihedral value candidates
        self.vals = None  # Available dihedral value candidates
        if self.delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the fragment.
        """
        self.ids = self.conf.getSwingAtoms(self.dihe)

    def setNfrags(self):
        """
        Set next fragments by searching for rotatable bond path.
        """
        while dihes := self.conf.mol.getDihes(self.dihe[1:2], self.ids):
            frags = [self] + [Fragment(self.conf, dihe=x) for x in dihes]
            for frag, nfrag in zip(frags[:-1], frags[1:]):
                frag.ids = sorted(set(frag.ids).difference(nfrag.ids))
                frag.nfrags.append(nfrag)
                nfrag.pfrag = frag

    def new(self, conf):
        """
        Create a new fragment based on the template.

        :param conf GrownConf: the conformer object this fragment belongs to.
        :return Fragment (sub)class: the copied fragment.
        """
        frag = self.__class__(conf, self.dihe, delay=True)
        frag.ids = conf.gids[self.ids]
        frag.nfrags = self.nfrags
        frag.ovals = np.linspace(0, 360, self.NUM, endpoint=False)
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
            if partial and len(frag.vals) == self.NUM:
                continue
            yield frag
            frags.extend(frag.nfrags)

    def reset(self):
        """
        Reset the current dihedral angle candidates.
        """
        self.vals = list(self.ovals)


class First(Fragment):
    """
    First monomer.
    """

    def setUp(self):
        """
        See parent.
        """
        super().setUp()
        for frag in self.next():
            frag.setNfrags()

    def new(self, conf, **kwargs):
        """
        See parent.
        """
        frag = super().new(conf)
        for pfrag in frag.next():
            pfrag.nfrags = [x.new(conf) for x in pfrag.nfrags]
            for nfrag in pfrag.nfrags:
                nfrag.pfrag = pfrag
        return frag
