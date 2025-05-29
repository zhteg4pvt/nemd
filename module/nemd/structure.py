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


class Conf(Chem.rdchem.Conformer):
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
        self.gids = self.mol.aids + start if self.mol else None

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
    Conf = Conf

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
        self.aids = np.arange(self.GetNumAtoms(), dtype=np.uint32)
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
        for conf in mol.GetConformers():
            self.append(conf)

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return `Conformer`: the selected conformer.
        """
        return self.confs

    def append(self, conf):
        """
        Append one conformer.

        :param conf `Chem.rdchem.Conformers`: the conformer to append.
        """
        pre = next(reversed(self.confs), None)
        gid = pre.gid + 1 if pre else 0
        start = pre.gids.max() + 1 if pre else 0
        # FIXME: every aid maps to a gid, but some gids may not map back (e.g.
        #  the virtual in tip4p water https://docs.lammps.org/Howto_tip4p.html)
        #  coarse-grained may have multiple aids mapping to one single gid
        #  united atom may have hydrogen aid mapping to None
        conf = self.Conf(conf, mol=self, gid=gid, start=start)
        self.confs.append(conf)

    def shift(self, pre):
        """
        Shift the ids of the conformers.

        :param pre `Conformer`: the previous conformer.
        """
        if pre is None:
            return
        gid, start = pre.gid + 1, pre.gids.max() + 1
        for conf in self.confs:
            conf.gid += gid
            conf.gids += start

    def AddConformer(self, conf, **kwargs):
        """
        Add conformer to the molecule.

        :param conf `Chem.rdchem.Conformer`: the conformer to add.
        """
        idx = super().AddConformer(conf, **kwargs)
        self.append(super().GetConformer(idx))

    def GetConformer(self, idx=0):
        """
        Get the conformer of the molecule.

        :param idx int: the conformer id to get.
        :return `Conformer`: the selected conformer.
        """
        return self.confs[idx]

    def GetNumConformers(self):
        """
        Get the number of conformers of the molecule.

        :return int: the number of conformers.
        """
        return len(self.confs)

    def EmbedMolecule(self,
                      randomSeed=1,
                      size=2**31,
                      clearConfs=True,
                      **kwargs):
        """
        Embed the molecule to generate a conformer.

        :param randomSeed int: the random seed for the embedding.
        :param size int: the maximum random seed - the minimum seed + 1.
        :param clearConfs bool: clear all existing conformations.
        :return int: the added conformer id.
        """
        idx = AllChem.EmbedMolecule(self,
                                    randomSeed=randomSeed % size,
                                    **kwargs)
        Chem.GetSymmSSSR(self)
        if clearConfs:
            self.confs.clear()
        self.append(super().GetConformer(idx))
        return idx

    def EmbedMultipleConfs(self,
                           *args,
                           randomSeed=1,
                           size=2**31,
                           clearConfs=True,
                           **kwargs):
        """
        Embed the molecule with multiple conformers.

        :param randomSeed int: the random seed for the embedding.
        :param size int: the maximum random seed - the minimum seed + 1.
        :param clearConfs bool: clear all existing conformations.
        """
        # All conformers have the same coordinates when randomSeed == 0
        indices = AllChem.EmbedMultipleConfs(self,
                                             *args,
                                             randomSeed=randomSeed % size,
                                             clearConfs=clearConfs,
                                             **kwargs)
        if clearConfs:
            self.confs.clear()
        for idx in indices:
            self.append(super().GetConformer(idx))

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
        return tuple(sorted(bond)) in self.rotatable

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

    def GetMolFrags(self, *args, **kwargs):
        return Chem.GetMolFrags(self, *args, **kwargs)

    @property
    def smiles(self):
        """
        Get the SMILES string.

        :return str: the SMILES string.
        """
        return Chem.MolToSmiles(self)


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """
    Mol = Mol

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
        pre = next(reversed(list(self.conf)), None)
        for original in mols:
            mol = self.Mol(original, struct=self)
            mol.shift(pre)
            self.mols.append(mol)
            pre = next(reversed(mol.confs), None)

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

    @property
    def conf(self):
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
