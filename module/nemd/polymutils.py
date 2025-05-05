# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module builds polymers.
"""
import collections
import functools

import networkx as nx
import numpy as np
from rdkit import Chem

from nemd import cru
from nemd import logutils
from nemd import rdkitutils
from nemd import structure
from nemd import structutils
from nemd import symbols

MARKER = 'marker'
WILD_MOL = Chem.MolFromSmiles(symbols.WILD_CARD)
MAID = structutils.GrownMol.MAID


class Cap(collections.UserList):

    @property
    def cap(self):
        """
        Get the capping atom indexes.

        :return list of int: the capping atom indexes.
        """
        return [x[0] for x in self]

    @property
    def aid(self):
        """
        Get the capped atom indexes.

        :return list of int: the capped atom indexes.
        """
        return [x[1] for x in self]


class Moiety(cru.Moiety):
    """
    Class to hold a moiety.
    """
    ConfClass = structutils.GriddedConf
    ONE_ZERO_ZERO = np.array([1, 0, 0])
    RES_NAME = 'res_name'
    RES_NUM = 'res_num'

    # CARBON_MOL = Chem.MolFromSmiles(symbols.CARBON)

    def __init__(self, *args, info=None, **kwargs):
        """
        :param info dict: the residue information
        """
        super().__init__(*args, **kwargs)
        self.setResidueInfo(info)

    def setResidueInfo(self, prop):
        """
        Set the residue each atom. For example, name, number, etc.

        :param prop dict: the residue information
        """
        if prop is None:
            return
        for atom in self.GetAtoms():
            info = atom.GetMonomerInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()
            for key, value in prop.items():
                match key:
                    case self.RES_NAME:
                        info.SetResidueName(value)
                    case self.RES_NUM:
                        info.SetResidueNumber(value)
            atom.SetMonomerInfo(info)

    def setMAID(self):
        """
        Set moiety atom id.
        """
        for atom in self.GetAtoms():
            atom.SetIntProp(MAID, atom.GetIdx())

    @property
    @functools.cache
    def head(self):
        """
        The head/tail atoms of the cru molecule.

        :return pd.DataFrame: the head atoms with their monomer atom index
        """
        return self.getRole(self.HEAD_ID)

    @property
    @functools.cache
    def tail(self):
        """
        The head/tail atoms of the cru molecule.

        :return pd.DataFrame: the head atoms with their monomer atom index
        """
        return self.getRole(self.TAIL_ID)

    def getRole(self, role_id):
        """
        The atom indexes of this role.

        :param role int: the role id of the atom, e.g. 0 for Head or 1 for Tail
        :return 'Cap': the atom indexes of capping and capped atoms
        """
        capping = self.capping(role_id=role_id)
        capping_aids = [x.GetIdx() for x in capping]
        aids = [x.GetNeighbors()[0].GetIdx() for x in capping]
        return Cap([(x, y) for x, y in zip(capping_aids, aids)])

    def extend(self, mol):
        """
        Extend the molecule by connecting the tail of this moiety with the head
        atom of the input.

        :param mol 'Moiety': the frag to extend with
        """
        t_atoms, t_cap_aids = [], []
        if self.GetNumAtoms():
            # select one available tail
            t_atoms = [self.GetAtomWithIdx(self.tail.aid[0])]
            t_cap_aids = self.tail.cap[:1]
        h_atoms, h_cap_aids = [], []
        if mol.GetNumAtoms():
            h_atoms = [mol.GetAtomWithIdx(x) for x in mol.head.aid]
            h_cap_aids = mol.head.cap
        if not all([self.GetNumAtoms(), mol.GetNumAtoms()]):
            # One of the molecules is empty so that implicit hydrogen is tuned
            for atom in t_atoms + h_atoms:
                atom.SetBoolProp(structutils.GrownMol.POLYM_HT, True)
                implicit_h = atom.GetIntProp(symbols.IMPLICIT_H) + 1
                atom.SetIntProp(symbols.IMPLICIT_H, implicit_h)
        # FIXME: support multiple tails bonded to the copies of molecules
        # Increase the residue number of the mol
        nums = [x.GetMonomerInfo().GetResidueNumber() for x in self.GetAtoms()]
        res_num = max(nums or [0]) + 1
        for atom in self.GetAtoms():
            info = atom.GetMonomerInfo()
            info.SetResidueNumber(info.GetResidueNumber() + res_num)
        # Combine the molecules and remove the capping atoms
        edcombo = Chem.EditableMol(Chem.CombineMols(self, mol))
        for cap_aid in sorted(t_cap_aids + h_cap_aids, reverse=True):
            edcombo.RemoveAtom(cap_aid)
        return Moiety(edcombo.GetMol())

    def copy(self, res_num=None):
        """
        Copy the moiety and set the res_num.

        :param res_num int: assign the atoms of the copied with this res_num
        :return 'Moiety': the copied fragment
        """
        info = None if res_num is None else {self.RES_NUM: res_num}
        return Moiety(self, info=info)

    def embed(self, *args, **kwargs):
        """
        Embed the conformer.
        """
        # Get XYZ with wild cards as carbon atoms
        mol = Moiety(self)
        for atom in mol.wild_card:
            atom.SetAtomicNum(6)
        mol.EmbedMolecule(*args, **kwargs)
        xyz = mol.GetConformer().GetPositions()
        # Create conformer and set XYZ
        conf = structure.Conformer(self.GetNumAtoms())
        for atom in self.GetAtoms():
            maid = atom.GetIntProp(MAID)
            conf.SetAtomPosition(atom.GetIdx(), xyz[maid])
        self.AddConformer(conf, assignId=True)
        Chem.GetSymmSSSR(self)

    def setAllTrans(self):
        """
        Set the backbone with all-trans dihedral angles.
        """
        conf = self.GetConformer()
        aids = nx.shortest_path(self.graph, self.head.cap[0], self.tail.cap[0])
        for dihe in zip(aids[:-3], aids[1:-2], aids[2:-1], aids[3:]):
            if not self.isRotatable(dihe[1:-1]):
                continue
            conf.setDihedralDeg(dihe, 180)

    def getPositions(self, bond=None):
        """
        Get the conformer coordinates.

        :param bond Chem.rdchem.Bond: the residue starts from this bond
        """
        idx = bond.GetIntProp(Moieties.END) if bond else None
        conf = self.getConforer(idx=idx)
        if idx is None:
            return conf.GetPositions()
        conf = structutils.GriddedConf(conf)
        vec = np.array([bond.GetDoubleProp(x) for x in Moieties.VEC])
        conf.rotate(self.ONE_ZERO_ZERO, vec)
        xyz = np.array([bond.GetDoubleProp(x) for x in Moieties.XYZ])
        conf.translate(xyz)
        return conf.GetPositions()

    @functools.cache
    def getConforer(self, idx=None):
        """
        Get the conformer rotated and translated according to the capping atoms.

        :param idx int: the index of the capping atoms.
        :return structutils.GriddedConf: the rotated and translated conformer.
        """
        conf = super().GetConformer()
        if idx is None:
            conf.translate(-conf.centroid())
            return conf
        cap = self.GetAtomWithIdx(idx)
        atom = cap.GetNeighbors()[0]
        coords = conf.GetAtomPosition(atom.GetIdx())
        conf.translate(-np.array([coords.x, coords.y, coords.z]))
        coords = conf.GetAtomPosition(cap.GetIdx())
        vec = -np.array([coords.x, coords.y, coords.z])
        conf.rotate(vec, self.ONE_ZERO_ZERO)
        return conf


class Moieties(cru.Moieties):
    """
    Class to hold moieties and build a polymer.
    """

    BEGIN = 'begin'
    END = 'end'
    BEGIN_END = [BEGIN, END]
    VEC = ['vx', 'vy', 'vz']
    XYZ = symbols.XYZU

    def __init__(self, cru, cru_num, options=None):
        """
        :param cru str: constitutional repeat unit
        :param cru_num int: the number of the conformers
        :param options 'argparse.Namespace': Command line options
        """
        super().__init__()
        self.cru = cru
        self.cru_num = cru_num
        self.options = options
        self.inr = Moiety(info=dict(res_num=0))
        self.ter = Moiety()
        self.mers = []
        self.mols = []
        self.length = {}

    def setUp(self):
        """
        Set moieties mol based on the input smiles.
        """
        mol = structure.Mol.MolFromSmiles(self.cru)
        for idx, mol_frag in enumerate(Chem.GetMolFrags(mol, asMols=True)):
            name = self.getName(idx)
            moiety = Moiety(mol_frag, info=dict(res_name=name))
            moiety.setMAID()
            self[name] = moiety
            match self.getRole(moiety):
                case self.INITIATOR:
                    self.inr = moiety
                case self.TERMINATOR:
                    self.ter = moiety
                case self.MONOMER:
                    self.mers.append(moiety)
                case self.REGULAR:
                    self.mols.append(mol_frag)
        if self.mers:
            self.setPolymer()

    @classmethod
    def getName(cls, idx):
        """
        Get the name of the moiety based on the index.

        :param idx int: index of the moiety based on which name is determined.
        """
        return chr(65 + idx)

    def setPolymer(self):
        """
        Build and return the polymer.

        :return 'Moiety': the polymer built from moieties.
        """
        # FIXME: Support input sequence (e.g., AABA) and moiety ratios
        seq = [
            np.random.choice(self.mers).copy(x) for x in range(self.cru_num)
        ]
        mol = self.bond(seq)
        mol = self.inr.extend(mol)
        mol = mol.extend(self.ter)
        self.mols.append(mol)

    def bond(self, sequence):
        """
        Create bonds between the atoms of a sequence of monomers.

        :param sequence list: the list of monomers to bond
        :param mol 'Moiety': the frag with bond created.
        """
        mol = Moiety(functools.reduce(Chem.CombineMols, sequence))
        # FIXME: Support head-head and tail-tail coupling
        pres, nexs = mol.tail[:-1], mol.head[1:]
        edcombo, props = Chem.EditableMol(mol), {}
        # Form bonds between the pres and nexs
        for pre, nex in zip(pres, nexs):
            caps, aids = zip(pre, nex)
            edcombo.AddBond(*aids, order=Chem.rdchem.BondType.SINGLE)
            maids = [mol.GetAtomWithIdx(x).GetIntProp(MAID) for x in caps]
            props[aids] = maids
        # Record capping atoms moiety atom ids on the formed bonds
        polym = edcombo.GetMol()
        for aids, maids in props.items():
            bond = polym.GetBondBetweenAtoms(*aids)
            for key, val in zip(self.BEGIN_END, maids):
                bond.SetIntProp(key, val)
        # Remove the capping atoms
        editable = Chem.EditableMol(polym)
        for aid in sorted(pres.cap + nexs.cap, reverse=True):
            editable.RemoveAtom(aid)
        return Moiety(editable.GetMol())

    def setVec(self, bond, xyzs):
        """
        Set the bond vector, which is the target direction from the previous
        moiety to the next moiety. The target coordinates of the end atom in the
        next moiety is also saved.

        :param bond 'rdkit.Chem.rdchem.Bond': the bond to set the vector to.
        :param xyzs 'numpy.ndarray': the coordinates of the previous moiety
        """
        oxyz = xyzs[bond.GetBeginAtom().GetIntProp(MAID)]
        vec = xyzs[bond.GetIntProp(self.BEGIN)] - oxyz
        for prop, val in zip(self.VEC, vec):
            bond.SetDoubleProp(prop, val)
        vec *= self.getLength(bond) / np.linalg.norm(vec)
        for prop, val in zip(self.XYZ, oxyz + vec):
            bond.SetDoubleProp(prop, val)

    def getLength(self, bond):
        """
        Get the length of a bond.

        :param bond 'rdkit.Chem.rdchem.Bond': the bond to get the hash value of.
        :return float: the bond length in Angstroms.
        """
        key = self.hash(bond)
        if key in self.length:
            return self.length[key]
        # Mark atoms to bond
        moieties = [self[x].copy() for x in key[::2]]
        for moiety, idx in zip(moieties, key[1::2]):
            moiety.GetAtomWithIdx(idx).SetBoolProp(MARKER, True)
        # Combine moieties and add the bond
        cmol = Chem.CombineMols(*moieties)
        bonded = [x.GetIdx() for x in cmol.GetAtoms() if x.HasProp(MARKER)]
        edcombo = Chem.EditableMol(cmol)
        edcombo.AddBond(*bonded, order=Chem.rdchem.BondType.SINGLE)
        mol = Chem.DeleteSubstructs(edcombo.GetMol(), WILD_MOL)
        # Measure the bond length
        mol = structure.Mol(mol)
        with rdkitutils.capture_logging():
            mol.EmbedMolecule(useRandomCoords=True,
                              randomSeed=self.options.seed)
        bonded = [x.GetIdx() for x in mol.GetAtoms() if x.HasProp(MARKER)]
        lgth = Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), *bonded)
        return self.length.setdefault(key, lgth)

    def hash(self, bond):
        """
        Get the hash value of a bond.

        :param bond 'rdkit.Chem.rdchem.Bond': the bond to get the hash value of.
        :return tuple: (the moiety name, the moiety atom index,
            the bonded moiety name, the bonded moiety atom index)
        """
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        names = [x.GetMonomerInfo().GetResidueName() for x in atoms]
        nums = [x.GetIntProp(MAID) for x in atoms]
        name_nums = [tuple(x) for x in zip(names, nums)]
        if name_nums[0] < name_nums[1]:
            name_nums = name_nums[::-1]
        return name_nums[0] + name_nums[1]


class Repeated(list):

    def __init__(self, *args, repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat = repeat

    def __iter__(self):
        for _ in range(self.repeat):
            yield self[0]


class Mol(structure.Mol, logutils.Base):
    """
    Class to hold a regular molecule or a polymer built from moieties.
    """

    def __init__(self,
                 mol,
                 mol_num,
                 moieties=None,
                 options=None,
                 delay=False,
                 **kwargs):
        """
        :param mol `rdkit.Chem.rdchem.Mol`: the molecule or polymer
        :param mol_num int: the number of molecules of this type of polymer
        :param options 'argparse.Namespace': command-line options
        :param delay bool: if True, the object is initialized without building
            the polymer (base class __init__ is called in the setUp method).
        """
        logutils.Base.__init__(self, **kwargs)
        self.moieties = moieties
        self.mol_num = mol_num
        self.options = options
        if delay:
            return
        super().__init__(mol, is_polym=bool(self.moieties.mers), delay=delay)
        if self.moieties.mers:
            self.log(f"Polymer SMILES: {Chem.MolToSmiles(self)}")
            self.embedMoieties()
            self.embedPolymer()
        else:
            self.embedMol()
        self.addConfRefs()

    def embedMol(self):
        """
        Embed the molecule with coordinates.
        """
        with rdkitutils.capture_logging(self.logger):
            # e.g. Mg+2 triggers the following ERROR:
            #   ERROR UFFTYPER: Unrecognized charge state for atom: 0
            # in addition, other WARNING messages are also captured
            #   WARNING UFFTYPER: Warning: hybridization set to SP3 for atom 0
            self.EmbedMolecule(useRandomCoords=True,
                               randomSeed=self.options.seed)

    def embedMoieties(self):
        """
        Embed the molecule or moieties with coordinates.
        """
        for moiety in self.moieties.values():
            with rdkitutils.capture_logging(self.logger):
                moiety.embed(useRandomCoords=True,
                             randomSeed=self.options.seed)
            moiety.setAllTrans()

    def embedPolymer(self):
        """
        Build and set the conformer.
        """
        if not self.is_polym:
            return
        Chem.GetSymmSSSR(self)
        conf = structure.Conformer(self.GetNumAtoms())
        bonds = [None]
        while bonds:
            bond = bonds.pop()
            bonds += self.setPositions(conf, bond=bond)
        self.AddConformer(conf, assignId=True)

    def setPositions(self, conf, bond=None):
        """
        Set the positions of this conformer.

        :param conf structure.Conformer: the conformer to set the positions
        :param bond Chem.rdchem.Bond: the residue starts from this bond
        :return list: the bonds between the current and next moieties
        """
        res_num = bond.GetEndAtom().GetMonomerInfo().GetResidueNumber() \
            if bond else min(self.res_atoms)
        name = self.res_atoms[res_num][0].GetMonomerInfo().GetResidueName()
        # Set the coordinates according to the moiety
        xyz = self.moieties[name].getPositions(bond=bond)
        for atom in self.res_atoms[res_num]:
            maid = atom.GetIntProp(MAID)
            conf.SetAtomPosition(atom.GetIdx(), xyz[maid])
        # Record the target vector
        bonds = list(self.getBonds(res_num))
        for bond in bonds:
            self.moieties.setVec(bond, xyz)
        return bonds

    @property
    @functools.cache
    def res_atoms(self):
        """
        Group the atoms by residue number and return a dictionary.

        :return dict: the atoms grouped by residue number
        """
        atoms = collections.defaultdict(list)
        for atom in self.GetAtoms():
            res_num = atom.GetMonomerInfo().GetResidueNumber()
            atoms[res_num].append(atom)
        return atoms

    def getBonds(self, res_num):
        """
        Get the bonds between the current moiety and the next moiety.

        :param res_num int: the current residue number
        :return iterator of 'rdkit.Chem.rdchem.Bond': bond between two moieties
        """
        for atom in self.res_atoms[res_num]:
            for neigh in atom.GetNeighbors():
                nex_num = neigh.GetMonomerInfo().GetResidueNumber()
                if nex_num <= res_num:
                    continue
                yield self.GetBondBetweenAtoms(atom.GetIdx(), neigh.GetIdx())

    def addConfRefs(self):
        """
        Add multiple conformer references pointing to the first one.

        FIXME: Add multiple conformers at once for real.
        Currently, each AddConformer(conf, **kwargs) call becomes more expensive

        1000   x [Ar]: 3.438e-05 per call
        10000  x [Ar]: 0.0001331 per call
        100000 x [Ar]: 0.001781 per call
        """
        self.confs = Repeated(self.confs, repeat=self.mol_num)

    @classmethod
    def write(cls, mol, filename):
        """
        Write the polymer and monomer into sdf files.

        :param mol 'rdkit.Chem.rdchem.Mol': The molecule to write out
        :param filename str: The file path to write into
        """

        with Chem.SDWriter(filename) as fh:
            try:
                maids = [x.GetIntProp(Moiety.MAID) for x in mol.GetAtoms()]
            except KeyError:
                fh.write(mol)
                return
            mol.SetProps([Moiety.MAID])
            mol.SetProp(Moiety.MAID, ' '.join(map(str, maids)))
            fh.write(mol)

    @classmethod
    def read(cls, filename):
        """
        Read molecule from file path.

        :param filename str: the file path to read molecule from.
        :return 'rdkit.Chem.rdchem.Mol': The molecule with properties.
        """
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        mol = next(suppl)
        Chem.GetSymmSSSR(mol)
        try:
            maids = mol.GetProp(Moiety.MAID).split()
        except KeyError:
            return mol
        for atom, mono_atom_idx in zip(mol.GetAtoms(), maids):
            atom.SetProp(Moiety.MAID, mono_atom_idx)
        return mol
