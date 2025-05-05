# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module handles conformer, molecule and structure.
"""
import functools

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import symbols


class Conformer(Chem.rdchem.Conformer):
    """
    A subclass of Chem.rdchem.Conformer with additional attributes and methods.
    """

    def __init__(self, *args, mol=None, gid=0, start=0, **kwargs):
        """
        :param mol:  `Chem.rdchem.Mol`: the molecule this conformer belongs to.
        :param gid int: the conformer gid.
        :param start int: the starting global atom id.
        """
        super().__init__(*args, **kwargs)
        self.mol = mol
        self.gid = gid
        self.id_map = self.mol.id_map + start if self.mol else None

    @property
    @functools.cache
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return self.id_map.tolist()

    def HasOwningMol(self):
        """
        Returns whether this conformer belongs to a molecule.

        :return `bool`: the molecule this conformer belongs to.
        """
        return self.mol is not None

    def GetOwningMol(self):
        """
        Get the Mol that owns this conformer.

        :return `Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        return self.mol

    def setPositions(self, xyz):
        """
        Reset the positions of the atoms to the original xyz coordinates.

        :param xyz `np.ndarray`: the xyz coordinates.
        """
        for idx in range(xyz.shape[0]):
            self.SetAtomPosition(idx, xyz[idx, :])


class Mol(Chem.rdchem.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """
    ConfWrapper = None
    ConfClass = Conformer

    def __init__(self,
                 *args,
                 is_polym=None,
                 vecs=None,
                 struct=None,
                 delay=False,
                 **kwargs):
        """
        :param struct 'Mol': the molecule instance
        :param is_polym bool: whether the molecule is built from moieties.
        :param vecs list: scaled lattice vectors
        :param struct 'Struct': owning structure
        :param delay bool: customization is delayed for later setup or testing.
        """
        super().__init__(*args, **kwargs)
        self.is_polym = is_polym
        self.vecs = vecs
        self.struct = struct
        self.delay = delay
        self.confs = []
        self.id_map = None
        mol = next(iter(args), None)
        if not mol:
            return
        if self.is_polym is None:
            self.is_polym = getattr(mol, 'is_polym', False)
        if self.vecs is None:
            self.vecs = getattr(mol, 'vecs', None)
        if self.delay:
            return
        self.setUp(mol.GetConformers())

    def setUp(self, confs, gid=0, start=0):
        """
        Set up the conformers including global ids and references.

        :param confs `Chem.rdchem.Conformers`: the conformers to set up.
        :param gid int: the conformer gid.
        :param start int: the starting global atom id.
        """
        if self.struct:
            gid, start = self.struct.getGid()
        ConfClass = self.ConfClass
        if self.ConfWrapper is not None:
            ConfClass = self.ConfWrapper
            confs[0] = self.ConfClass(confs[0], mol=self)
        # FIXME: every aid maps to a gid, but some gids may not map back (e.g.
        #  the virtual in tip4p water https://docs.lammps.org/Howto_tip4p.html)
        #  coarse-grained may have multiple aids mapping to one single gid
        #  united atom may have hydrogen aid mapping to None
        self.id_map = np.arange(0, self.GetNumAtoms(), dtype=np.uint32)
        for cid, conf in enumerate(confs, start=gid):
            conf = ConfClass(conf, mol=self, gid=np.uint32(cid), start=start)
            self.confs.append(conf)
            start += np.uint32(self.GetNumAtoms())

    def GetConformer(self, idx=0):
        """
        Get the conformer of the molecule.

        :param idx int: the conformer id to get.
        :return `Conformer`: the selected conformer.
        """
        return self.confs[idx]

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return `Conformer`: the selected conformer.
        """
        return self.confs

    def GetNumConformers(self):
        """
        Get the number of conformers of the molecule.

        :return int: the number of conformers.
        """
        return len(self.confs)

    def AddConformer(self, conf, **kwargs):
        """
        Add conformer to the molecule.

        :param conf `Chem.rdchem.Conformer`: the conformer to add.
        """
        idx = super().AddConformer(conf, **kwargs)
        self.setUp([super().GetConformer(idx)])

    def EmbedMolecule(self, randomSeed=-1, max_seed=2**31 - 1, **kwargs):
        """
        Embed the molecule to generate a conformer.

        :param randomSeed int: the random seed for the embedding.
        :param max_seed int: the maximum random seed.
        """
        if randomSeed > max_seed:
            randomSeed = np.random.randint(0, max_seed)
        AllChem.EmbedMolecule(self, randomSeed=randomSeed, **kwargs)
        Chem.GetSymmSSSR(self)
        # Parent EmbedMolecule add one after clearing previous conformers.
        self.confs.clear()
        self.setUp(super().GetConformers())

    @classmethod
    def MolFromSmiles(cls, smiles, united=True, **kwargs):
        """
        Create a molecule from SMILES.

        :param smiles str: the SMILES string.
        :param united bool: hide keep Hydrogen atoms in CH, CH3, CH3, and CH4.
        :return `Mol`: the molecule instance.
        """
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
        if not united:
            return cls(Chem.AddHs(mol), **kwargs)

        # Hide Hs in CH, CH3, CH3, and CH4
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != symbols.CARBON or atom.GetIsAromatic():
                continue
            atom.SetIntProp(symbols.IMPLICIT_H, atom.GetNumImplicitHs())
            atom.SetNoImplicit(True)

        # FIXME: support different chiralities for monomers
        for chiral in Chem.FindMolChiralCenters(mol, includeUnassigned=True):
            # CIP stereochemistry assignment for the molecule’s atoms (R/S)
            # and double bonds (Z/E)
            mol.GetAtomWithIdx(chiral[0]).SetProp('_CIPCode', 'R')

        return cls(Chem.AddHs(mol), **kwargs)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return Chem.Descriptors.ExactMolWt(self)

    mw = molecular_weight

    @property
    def atom_total(self):
        """
        The total number of atoms in all conformers.

        :return int: the total number of atoms in all conformers.
        """
        return self.GetNumAtoms() * self.GetNumConformers()

    @property
    @functools.cache
    def graph(self):
        """
        Get the networkx graph on the molecule.

        :return `networkx.Graph`: the graph of the molecule.
        """
        graph = nx.Graph()
        edges = [[x.GetBeginAtom(), x.GetEndAtom()] for x in self.GetBonds()]
        edges = [tuple([x[0].GetIdx(), x[1].GetIdx()]) for x in edges]
        if edges:
            graph.add_edges_from(edges)
        else:
            graph.add_nodes_from(x.GetIdx() for x in self.GetAtoms())
        return graph

    @functools.cache
    def isRotatable(self, bond):
        """
        Whether the bond between the two atoms is rotatable.

        :param bond list or tuple of two ints: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """
        in_ring = self.GetBondBetweenAtoms(*bond).IsInRing()
        return not in_ring and tuple(sorted(bond)) in self.rotatable

    @property
    @functools.cache
    def rotatable(
        self,
        mol=Chem.MolFromSmarts(
            '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')):
        """
        Get the rotatable bonds of the molecule.

        https://ctr.fandom.com/wiki/Break_rotatable_and_report_the_fragments

        :param mol `Chem.rdchem.Mol`: the rotatable mols
        :return list of tuples of two ints: the atom ids of two bonded atoms.
        """
        # https://ctr.fandom.com/wiki/Break_rotatable_and_report_the_fragments
        return self.GetSubstructMatches(mol, maxMatches=1000000)

    def GetPositions(self):
        """
        Get the position of all conformers.

        :return `np.ndarray`: the coordinates of all conformers.
        """
        return np.concatenate([x.GetPositions() for x in self.confs])


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    MolClass = Mol

    def __init__(self, struct=None):
        """
        :param struct 'Struct': the structure with molecules.
        """
        self.mols = []
        self.setUp(struct.mols if struct else [])

    def setUp(self, mols):
        """
        Set up the structure.

        :param mols list of 'Chem.rdchem.Mol': the molecules to add.
        """
        for mol in mols:
            self.mols.append(self.MolClass(mol, struct=self))

    @classmethod
    def fromMols(cls, mols, *args, **kwargs):
        """
        Create structure instance from mols.

        :param mols list of 'Chem.rdchem.Mol': the molecules to add.
        :return 'Struct': the structure containing the molecules.
        """
        struct = cls(*args, **kwargs)
        struct.setUp(mols)
        return struct

    def getGid(self):
        """
        Get the global ids to start with.

        :return int, int: the conformer gid, the global atom id.
        """
        gid = max([x.gid for x in self.conformer] or [-1]) + 1
        start = max([x.id_map.max() for x in self.conformer] or [-1]) + 1
        return gid, start

    @property
    def conformer(self):
        """
        Return generator of all conformers from all molecules.

        :return generator of `Conformer`: the conformers of all molecules.
        """
        return (x for y in self.mols for x in y.confs)

    @property
    def atom_total(self):
        """
        Return The total number of atoms in all conformers across all molecules.

        :return int: the total number of atoms in all conformers.
        """
        return sum([x.atom_total for x in self.mols])

    @property
    def conformer_total(self):
        """
        Return the total number of conformers.

        :return int: the total number of conformers.
        """
        return sum([len(x.confs) for x in self.mols])
