# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Conformer, molecule and structure.
"""
import functools

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import symbols


class Conformer(Chem.rdchem.Conformer):
    """
    Customized Chem.rdchem.Conformer.
    """

    def __init__(self, *args, mol=None, gid=0, start=0, **kwargs):
        """
        :param mol: `Mol`: the molecule this conformer belongs to.
        :param gid int: the conformer gid.
        :param start int: the starting global atom id.
        """
        super().__init__(*args, **kwargs)
        self.mol = mol
        self.gid = gid
        self.gids = self.mol.gids + start if self.mol else None

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
    Customized Chem.rdchem.Mol.
    """
    ConfClass = Conformer

    def __init__(self,
                 *args,
                 struct=None,
                 polym=None,
                 vecs=None,
                 delay=False,
                 **kwargs):
        """
        :param args[0] 'Chem.rdchem.Mol': the molecule instance
        :param struct 'Struct': owning structure
        :param polym bool: whether the molecule is built from moieties.
        :param vecs list: scaled lattice vectors
        :param delay bool: customization is delayed for later setup or testing.
        """
        super().__init__(*args, **kwargs)
        self.struct = struct
        self.polym = polym
        self.vecs = vecs
        self.delay = delay
        self.confs = []
        self.gids = np.arange(self.GetNumAtoms(), dtype=np.uint32)
        if self.delay:
            return
        self.setUp(next(iter(args), None))

    def setUp(self, mol):
        """
        Set up.

        :param mol `Chem.rdchem.Mol`: the original molecule.
        """
        if mol is None:
            return
        if self.polym is None:
            self.polym = getattr(mol, 'polym', False)
        if self.vecs is None:
            self.vecs = getattr(mol, 'vecs', False)
        self.setConfs(mol.GetConformers())

    def setConfs(self, confs):
        """
        Set the conformers.

        :param confs `Chem.rdchem.Conformers`: the original conformers.
        """
        gid, start = self.struct.getNext() if self.struct else self.getNext()
        # FIXME: every aid maps to a gid, but some gids may not map back (e.g.
        #  the virtual in tip4p water https://docs.lammps.org/Howto_tip4p.html)
        #  coarse-grained may have multiple aids mapping to one single gid
        #  united atom may have hydrogen aid mapping to None
        for cid, conf in enumerate(list(confs), start=gid):
            conf = self.ConfClass(conf, mol=self, gid=cid, start=start)
            self.confs.append(conf)
            start += self.GetNumAtoms()

    def getNext(self):
        """
        Get the next ids on extending with the conformer.

        :return int, int: the conformer gid, the global atom id.
        """
        ids = np.array([[x.gid, x.gids.max()] for x in self.confs])
        return ids.max(axis=0) + 1 if ids.size else [0, 0]

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return `Conformer`: the selected conformer.
        """
        return self.confs

    def AddConformer(self, conf, **kwargs):
        """
        Add conformer to the molecule.

        :param conf `Chem.rdchem.Conformer`: the conformer to add.
        """
        idx = super().AddConformer(conf, **kwargs)
        self.setConfs([super().GetConformer(idx)])

    def GetConformer(self, idx=0):
        """
        Get the conformer of the molecule.

        :param idx int: the conformer id to get.
        :return `Conformer`: the selected conformer.
        """
        return self.confs[idx]

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
        self.setConfs(super().GetConformers())

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

    def GetNumConformers(self):
        """
        Get the number of conformers of the molecule.

        :return int: the number of conformers.
        """
        return len(self.confs)

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

        :param bond tuple: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """
        return not self.GetBondBetweenAtoms(*bond).IsInRing() and \
            tuple(sorted(bond)) in self.rotatable

    @property
    @functools.cache
    def rotatable(
        self,
        mol=Chem.MolFromSmarts(
            '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')):
        """
        Return all rotatable bonds.

        https://ctr.fandom.com/wiki/Break_rotatable_and_report_the_fragments

        :param mol `Chem.rdchem.Mol`: the rotatable molecules
        :return list of tuples of two ints: the atom ids of two bonded atoms.
        """
        return self.GetSubstructMatches(mol, maxMatches=1000000)


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
        Create structure instance from molecules.

        :param mols list of 'Chem.rdchem.Mol': the molecules to add.
        :return 'Struct': the structure containing the molecules.
        """
        struct = cls(*args, **kwargs)
        struct.setUp(mols)
        return struct

    def getNext(self):
        """
        Get the next ids on extending with the conformer.

        :return int, int: the conformer gid, the global atom id.
        """
        ids = np.array([x.getNext() for x in self.mols])
        return ids.max(axis=0) if ids.size else [0, 0]

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
        return sum([x.GetNumAtoms() * x.GetNumConformers() for x in self.mols])
