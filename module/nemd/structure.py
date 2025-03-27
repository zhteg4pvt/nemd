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

    def __init__(self, *args, mol=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mol = mol
        self.gid = None
        self.start = 0

    def setUp(self, mol, cid=0, start=0):
        """
        Set up the conformer global id, global atoms ids, and owning molecule.

        :param mol `Chem.rdchem.Mol`: the original molecule.
        :param cid int: the conformer gid to start with.
        :param gid int: the starting global id.
        """
        self.mol = mol
        self.gid = cid
        self.start = start

    @property
    def id_map(self):
        """
        Return map from atom ids to the global atom ids.

        :return 'np.ndarray': the map from atom ids to global atom ids.
        """
        return self.GetOwningMol().id_map + self.start

    @property
    @functools.cache
    def aids(self):
        """
        Return the atom ids of this conformer.

        :return list of int: the atom ids of this conformer.
        """
        return np.where(self.id_map != -1)[0].tolist()

    @property
    @functools.cache
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return self.id_map[self.id_map != -1].tolist()

    def HasOwningMol(self):
        """
        Returns whether this conformer belongs to a molecule.

        :return `bool`: the molecule this conformer belongs to.
        """
        return bool(self.GetOwningMol())

    def GetOwningMol(self):
        """
        Get the Mol that owns this conformer.

        :return `Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        return self.mol

    def setPositions(self, xyz):
        """
        Reset the positions of the atoms to the original xyz coordinates.
        """
        for id in range(xyz.shape[0]):
            self.SetAtomPosition(id, xyz[id, :])


class Mol(Chem.rdchem.Mol):
    """
    A subclass of Chem.rdchem.Mol with additional attributes and methods.
    """

    ConfWrapper = None
    ConfClass = Conformer
    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    ROTATABLE_MOL = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')

    def __init__(self,
                 mol=None,
                 is_polym=False,
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
        # conformers in super(Mol, self).GetConformers() are rebuilt
        if mol is None:
            # create an empty molecule
            super().__init__()
        else:
            super().__init__(mol, **kwargs)
        self.is_polym = is_polym | getattr(mol, 'is_polym', False)
        self.vecs = vecs or getattr(mol, 'vecs', None)
        self.struct = struct
        self.delay = delay
        self.confs = []
        self.id_map = None
        if self.delay:
            return
        if mol is None:
            return
        self.setUp(mol.GetConformers())

    def setUp(self, confs, cid=0, start=0):
        """
        Set up the conformers including global ids and references.

        :param confs `Chem.rdchem.Conformers`: the conformers from the original
            molecule.
        :param cid int: the conformer gid to start with.
        :param gid int: the starting global id.
        """
        if self.struct:
            cid, start = self.struct.getCids()
        ConfClass = self.ConfClass
        if self.ConfWrapper is not None:
            ConfClass = self.ConfWrapper
            confs[0] = self.ConfClass(confs[0], mol=self)
        # FIXME: every aid maps to a gid, but some gids may not map back (e.g.
        #  the virtual in tip4p water https://docs.lammps.org/Howto_tip4p.html)
        #  coarse-grained may have multiple aids mapping to one single gid
        #  united atom may have hydrogen aid mapping to None
        self.id_map = np.arange(0, self.GetNumAtoms(), dtype=np.uint32)
        for cid, conf in enumerate(confs, start=cid):
            conf = ConfClass(conf)
            conf.setUp(self, cid=np.uint32(cid), start=np.uint32(start))
            self.confs.append(conf)
            start += self.GetNumAtoms()

    def GetConformer(self, id=0):
        """
        Get the conformer of the molecule.

        :param id int: the conformer id to get.
        :return `Conformer`: the selected conformer.
        """
        return self.confs[id]

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
        # AddConformer handles the super().GetOwningMol()
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
        # EmbedMolecule clear previous conformers, and only add one.
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
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for chirality in chiral:
            # CIP stereochemistry assignment for the molecule’s atoms (R/S)
            # and double bonds (Z/E)
            mol.GetAtomWithIdx(chirality[0]).SetProp('_CIPCode', 'R')

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
            return graph
        # When bonds don't exist, just add the atom.
        for atom in self.GetAtoms():
            graph.add_node(atom.GetIdx())
        return graph

    @functools.cache
    def isRotatable(self, bond):
        """
        Whether the bond between the two atoms is rotatable.

        :param bond list or tuple of two ints: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """
        in_ring = self.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    @property
    @functools.cache
    def rotatable_bonds(self):
        """
        Get the rotatable bonds of the molecule.

        :return list of tuples of two ints: the atom ids of two bonded atoms.
        """

        return self.GetSubstructMatches(self.ROTATABLE_MOL, maxMatches=1000000)

    def GetPositions(self):
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
        self.molecules = []
        self.density = None
        if struct is None:
            return
        for mol in struct.molecules:
            self.addMol(mol)
        self.finalize()

    @classmethod
    def fromMols(cls, mols, *args, **kwargs):
        """
        Create structure instance from molecules.

        :param mols list of 'Chem.rdchem.Mol': the molecules to be added.
        :return 'Struct': the structure containing the molecules.
        """
        struct = cls(*args, **kwargs)
        for mol in mols:
            struct.addMol(mol)
        struct.finalize()
        return struct

    def addMol(self, mol):
        """
        Initialize molecules and conformers with id and map set.

        :param mol 'Mol': the molecule to be added.
        :return 'Mol': the added molecule.
        """
        mol = self.MolClass(mol, struct=self)
        self.molecules.append(mol)
        return mol

    def finalize(self):
        """
        Finalize the structure after all molecules are added.
        """
        pass

    def getCids(self):
        """
        Get the global ids to start with.

        :retrun int, int: the conformer gid, the global atom id.
        """
        cid = max([x.gid for x in self.conformer] or [-1]) + 1
        gid = max([x.id_map.max() for x in self.conformer] or [-1]) + 1
        return cid, gid

    @property
    def conformer(self):
        """
        Return generator of all conformers from all molecules.

        :return generator of `Conformer`: the conformers of all molecules.
        """
        return (x for y in self.molecules for x in y.confs)

    @property
    def atom(self):
        """
        Return generator of allatoms from molecules.

        Note: the number of these atoms is different atom_total as atom_toal
        includes atoms from all conformers.

        :return generator of Chem.rdchem.Atom: the atoms from all molecules.
        """
        return (y for x in self.molecules for y in x.GetAtoms())

    @property
    def atom_total(self):
        """
        The total number of atoms in all conformers across all molecules.

        :return int: the total number of atoms in all conformers.
        """
        return sum([x.atom_total for x in self.molecules])

    @property
    def conformer_total(self):
        """
        Get the total number of all conformers.

        :return int: the total number of all conformers.
        """
        return sum([len(x.confs) for x in self.molecules])
