# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module builds polymers.
"""
import collections
import functools

import itertools
import methodtools
import numpy as np
import scipy
from rdkit import Chem

from nemd import cru
from nemd import logutils
from nemd import rdkitutils
from nemd import structure
from nemd import structutils
from nemd import symbols

MAID = structutils.GrownMol.MAID
BEGIN = 'begin'
END = 'end'


class Conf(structutils.Conf):
    """
    Customized for alignment.
    """
    AXIS = (1, 0, 0)

    def getAligned(self, bond=None):
        """
        Get the conformer coordinates aligned according to the bond.

        :param bond `Chem.rdchem.Bond`: bond from previous moiety to this one
        :return `np.ndarray`: the conformer aligned according to the bond.
        """
        xyz = self.getXYZ(bond)
        if bond is None:
            return xyz

        vec = [bond.GetDoubleProp(x) for x in Mol.VEC]
        rot, _ = scipy.spatial.transform.Rotation.align_vectors(vec, self.AXIS)
        xyz = rot.apply(xyz)
        xyz += [bond.GetDoubleProp(x) for x in symbols.XYZU]
        return xyz

    @functools.cache
    def getXYZ(self, bond):
        """
        Get the xyz of the moiety aligned to origin and axis.

        :param bond `Chem.rdchem.Bond`: the bond ends with this moiety
        :return `np.ndarray`: the transformed coordinates.
        """
        if not bond:
            self.translate(-self.centroid())
            return self.GetPositions()

        cap_idx = bond.GetIntProp(END)
        cap = self.GetOwningMol().GetAtomWithIdx(cap_idx)
        centroid = self.centroid(aids=[x.GetIdx() for x in cap.GetNeighbors()])
        self.translate(-centroid)
        rotation, _ = scipy.spatial.transform.Rotation.align_vectors(
            self.AXIS, -np.array(self.GetAtomPosition(cap_idx)))
        self.rotate(rotation=rotation)
        return self.GetPositions()


class Moiety(cru.Moiety):
    """
    Moiety.
    """
    Conf = Conf
    RES_NUM = 'res_num'
    SERIAL = 'serial'

    def __init__(self, *args, info=None, **kwargs):
        """
        :param info dict: the residue information
        """
        super().__init__(*args, **kwargs)
        self.info = info
        self.setup()

    def setup(self):
        """
        Set up.
        """
        if self.info is None:
            return
        for atom in self.GetAtoms():
            info = atom.GetMonomerInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()
            for key, value in self.info.items():
                match key:
                    case self.RES_NUM:
                        info.SetResidueNumber(value)
                    case self.SERIAL:
                        info.SetSerialNumber(value)
            atom.SetMonomerInfo(info)

    def bond(self, mol):
        """
        Bond the tails of current moiety and the heads of the input moieties.

        :param mol 'Moiety': the moiety to extend with.
        :return 'Moiety': the extended molecule.
        """
        return self.combine(mol) or self.cap(mol)

    def combine(self, mol, marker='marker', tail=None, head=None, res=-1):
        """
        Combine molecules with bond creation and capping removal.

        :param mol list: the moiety to combine with.
        :param marker str: the marker of bonded atoms.
        :param res int: the previous residue number.
        :return `Moiety`: the combined moiety.
        """
        if any([self.empty(), mol.empty()]):
            return
        for atom in itertools.chain(self.GetAtoms(), mol.GetAtoms()):
            atom.ClearProp(marker)
        tails = self.tail if tail is None else [tail]
        mols = [mol.new() for _ in range(len(tails))]
        for idx, (tail, mol) in enumerate(zip(tails, mols)):
            tail.SetIntProp(marker, idx)
            atom = mol.head[0] if head is None else mol.GetAtomWithIdx(head.GetIdx())
            atom.SetIntProp(marker, idx)
        for mol in [self] + mols:
            res = mol.incrRes(delta=res + 1)
        combined = functools.reduce(Chem.CombineMols, [self] + mols)
        pairs = collections.defaultdict(list)
        for atom in combined.GetAtoms():
            if atom.HasProp(marker):
                pairs[atom.GetIntProp(marker)].append(atom)
        editable = EditableMol(combined).addBonds(pairs.values())
        return Moiety(editable.GetMol())

    def empty(self):
        """
        Whether the moiety is empty.

        :return 'bool': True when the moiety is empty.
        """
        return self.GetNumAtoms() == 0

    def new(self, info=None):
        """
        Create a new moiety.

        :param info dict: the residue information.
        :return 'Moiety': the copied fragment.
        """
        return Moiety(self, info=info)

    def incrRes(self, delta=0, max_res=0):
        """
        Increase the residue number.

        :param delta int: the increment.
        :param max_res int: the maximum residule number.
        :return int: the maximum residue number.
        """
        for atom in self.GetAtoms():
            info = atom.GetMonomerInfo()
            res = info.GetResidueNumber() + delta
            info.SetResidueNumber(res)
            max_res = max(max_res, res)
        return max_res

    @property
    @functools.cache
    def tail(self):
        """
        Get the capping atom of the tail.

        :return `Chem.Atom`: the capping atom of the tail.
        """
        return self.getCapping(self.TAIL_ID)

    @property
    @functools.cache
    def head(self):
        """
        Get the capping atom of the head.

        :return `Chem.Atom`: the capping atom of the head.
        """
        return self.getCapping(self.HEAD_ID)

    def cap(self, mol, implicit=symbols.IMPLICIT):
        """
        Cap the moiety by star atom mutation or deletion.

        :param mol 'Moiety': the moiety to extend with.
        :param implicit str: the implicit hydrogen property.
        :return `Moiety`: the capped moiety.
        """
        if self.empty() == mol.empty():
            return
        stars = self.tail + mol.head
        for atom in stars:
            if atom.GetNeighbors()[0].HasProp(implicit):
                atom = atom.GetNeighbors()[0]
                atom.SetIntProp(implicit, atom.GetIntProp(implicit) + 1)
            else:
                atom.SetAtomicNum(1)
            atom.SetBoolProp(structutils.GrownMol.POLYM_HT, True)
        stars = [x for x in stars if x.GetSymbol() == symbols.STAR]
        editable = EditableMol(self if self.GetNumAtoms() else mol)
        editable.removeAtoms([x.GetIdx() for x in stars])
        return Moiety(editable.GetMol())

    def EmbedMolecule(self, *args, **kwargs):
        """
        Embed the moiety.
        """
        for atom in self.stars:
            atom.SetAtomicNum(6)
        super().EmbedMolecule(*args, **kwargs)
        for atom in self.stars:
            atom.SetAtomicNum(0)
        conf = self.GetConformer()
        src, tgt = [[y.GetIdx() for y in x] for x in [self.head, self.tail]]
        for dihe in self.getDihes(sources=src, targets=tgt):
            conf.setDihedralDeg(dihe, 180)


class EditableMol(Chem.EditableMol):
    """
    Customized for moieties.
    """

    def removeAtoms(self, aids):
        """
        Remove atoms.

        :param aids list: the ids of the atoms to remove.
        """
        for aid in sorted(aids, reverse=True):
            self.RemoveAtom(aid)

    def addBonds(self, pairs):
        """
        Add and mark bonds.

        :param pairs list: each item is a capping atom pair of a head & a tail.
        :return `EditableMol`: the edited molecule.
        """
        maids = {}
        # Form bonds between the head and tail
        for caps in pairs:
            bonded = tuple([x.GetNeighbors()[0].GetIdx() for x in caps])
            self.AddBond(*bonded, order=Chem.rdchem.BondType.SINGLE)
            maids[bonded] = [x.GetIntProp(MAID) for x in caps]
        # Record capping atoms moiety atom ids on the formed bonds
        chain = self.GetMol()
        for bonded, (begin, end) in maids.items():
            bond = chain.GetBondBetweenAtoms(*bonded)
            bond.SetIntProp(BEGIN, begin)
            bond.SetIntProp(END, end)
        chain = EditableMol(chain)
        chain.removeAtoms([y.GetIdx() for x in pairs for y in x])
        return chain


class Sequence(list):
    """
    Monomer Sequence.
    """

    def build(self):
        """
        Build a chain out of the sequence.

        :return 'Moiety': the chain built from monomers.
        """
        mols = [x.new(dict(res_num=i)) for i, x in enumerate(self)]
        mol = Moiety(functools.reduce(Chem.CombineMols, mols))
        # FIXME: Support head-head and tail-tail coupling
        pairs = list(zip(mol.getCapping()[:-1], mol.getCapping(0)[1:]))
        editable = EditableMol(mol)
        editable = editable.addBonds(pairs)
        return Moiety(editable.GetMol())


class Moieties(list, logutils.Base):
    """
    Build s polymer from moieties.
    """

    def __init__(self, cru, cru_num=1, mol_num=0, options=None, **kwargs):
        """
        :param cru str: constitutional repeat unit
        :param cru_num int: the number monomer cru per chain
        :param mol_num int: the number of the conformers per chain
        :param options 'argparse.Namespace': Command line options
        """
        logutils.Base.__init__(self, **kwargs)
        super().__init__()
        self.cru = cru
        self.cru_num = cru_num
        self.mol_num = mol_num
        self.options = options
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        mols = structure.Mol.MolFromSmiles(self.cru).GetMolFrags(asMols=True)
        self.extend(Moiety(x, info=dict(serial=i)) for i, x in enumerate(mols))
        for moiety in [self.inr, self.ter, *self.mers]:
            for atom in moiety.GetAtoms():
                atom.SetIntProp(MAID, atom.GetIdx())

    @methodtools.lru_cache()
    @property
    def inr(self):
        """
        Get the initiator.

        :return list: the initiator moiety.
        """
        return self.getMoiety()

    def getMoiety(self, role=cru.INITIATOR):
        """
        Get moiety of certain role.

        :param role str: moiety role.
        :return `Moiety`: the selected moiety.
        """
        try:
            moiety = next(x for x in self if x.role == role)
        except StopIteration:
            return Moiety(info=dict(res_num=0))
        else:
            for atom in moiety.GetAtoms():
                atom.SetBoolProp(structutils.GrownMol.POLYM_HT, True)
            return moiety

    @methodtools.lru_cache()
    @property
    def ter(self):
        """
        Get the terminator.

        :return list: the terminator moiety.
        """
        return self.getMoiety(cru.TERMINATOR)

    @methodtools.lru_cache()
    @property
    def mers(self):
        """
        Get the monomers.

        :return list: the monomer moieties.
        """
        return [x for x in self if x.role == cru.MONOMER]

    def run(self):
        """
        Main method.
        """
        if not self.mols:
            # Build polymer
            chain =  self.inr.bond(self.getChain()) if self.mers else self.inr
            polym = chain.bond(self.ter)
            self.log(f"Polymer SMILES: {Chem.MolToSmiles(polym)}")
            self.mols.append(polym)
        # Embed conformer
        for idx, mol in enumerate(self.mols):
            self.mols[idx] = Mol(mol,
                                 mol_num=self.mol_num,
                                 moieties=self,
                                 options=self.options,
                                 logger=self.logger)

    @methodtools.lru_cache()
    @property
    def mols(self):
        """
        Get the molecules.

        :return list: regular molecules or built polymer.
        """
        return [x for x in self if x.role == cru.REGULAR]

    def getChain(self):
        """
        Get the moiety sequence.

        :return `Moiety`: Chan build from moieties.
        """
        # FIXME: Support input sequence (e.g., AABA) and moiety ratios
        return Sequence(np.random.choice(self.mers, self.cru_num)).build()

    @methodtools.lru_cache()
    def getLength(self, hashed):
        """
        Get the length of a bond between moieties.

        :param hashed tuple: moiety serial number, the moiety atom index,
            the bonded moiety serial number, the bonded moiety atom index
        :return float: the bond length in Angstroms.
        """
        # Mark atoms to form bond
        moieties = [self[x].new() for x in hashed[::2]]
        atoms = [x.GetAtomWithIdx(y) for x, y in zip(moieties, hashed[1::2])]
        tail, head = [next(y for y in x.GetNeighbors() if y.GetSymbol() == symbols.STAR) for x in atoms]
        mol = moieties[0].combine(moieties[1], tail=tail, head=head)
        with rdkitutils.capture_logging():
            mol.EmbedMolecule(randomSeed=self.options.seed)
        bond = next(x for x in mol.GetBonds() if x.HasProp(BEGIN))
        aids = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        return mol.GetConformer().measure(aids)


class Repeated(list):
    """
    List that repeat the first value on iteration.
    """

    def __init__(self, *args, repeat=1, **kwargs):
        """
        :param repeat int: the repeated times on iteration.
        """
        super().__init__(*args, **kwargs)
        self.repeat = repeat

    def __iter__(self):
        for _ in range(self.repeat):
            yield self[0]


class Mol(structure.Mol, logutils.Base):
    """
    Customized to embed polymer.
    """
    VEC = ['vx', 'vy', 'vz']

    def __init__(self,
                 mol,
                 mol_num=0,
                 moieties=None,
                 options=None,
                 delay=False,
                 **kwargs):
        """
        :param mol `Chem.Mol`: the molecule or polymer.
        :param mol_num int: the number of conformer.
        :param options 'argparse.Namespace': command-line options.
        :param delay bool: delay the initiation if True.
        """
        super().__init__(mol, polym=bool(moieties.mers), delay=delay)
        logutils.Base.__init__(self, **kwargs)
        self.moieties = moieties
        self.mol_num = mol_num
        self.options = options
        if not self.mol_num or delay:
            return
        self.EmbedMolecule()
        self.addConfRefs()

    def EmbedMolecule(self):
        """
        Embed the molecule with conformer.
        """
        with rdkitutils.capture_logging(self.logger):
            # ERROR UFFTYPER: Unrecognized charge state for atom: 0 (Mg+2)
            # WARNING UFFTYPER: Warning: hybridization set to SP3 for atom 0
            if not self.polym:
                super().EmbedMolecule(randomSeed=self.options.seed)
                return
            for moiety in self.moieties:
                moiety.EmbedMolecule(randomSeed=self.options.seed)

        Chem.GetSymmSSSR(self)
        conf = structure.Conf(self.GetNumAtoms())
        self.AddConformer(conf, assignId=True)

        bonds = [None]
        while bonds:
            bonds.extend(self.setConformer(bonds.pop()))

    def setConformer(self, pre, res=0):
        """
        Partially set the conformer of one residue (moiety).

        :param pre Chem.Bond: bond from previous to the current moiety.
        :param res int: the residue num of the current moiety.
        :return generator: the bonds between the current and next moieties
        """
        # Set the coordinates according to the moiety.
        if pre:
            res = pre.GetEndAtom().GetMonomerInfo().GetResidueNumber()
        serial = self.res[res][0].GetMonomerInfo().GetSerialNumber()
        xyzs = self.moieties[serial].GetConformer().getAligned(bond=pre)
        for atom in self.res[res]:
            xyz = xyzs[atom.GetIntProp(MAID)]
            self.GetConformer().SetAtomPosition(atom.GetIdx(), xyz)
        # Return the next bonds with target vector and coordinates recorded.
        for atom in self.res[res]:
            for nbr in atom.GetNeighbors():
                if nbr.GetMonomerInfo().GetResidueNumber() <= res:
                    continue
                bond = self.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                xyz = xyzs[atom.GetIntProp(MAID)]
                vec = xyzs[bond.GetIntProp(BEGIN)] - xyz
                vec *= self.moieties.getLength(
                    self.hash(bond)) / np.linalg.norm(vec)
                # Target bond vector
                for prop, val in zip(self.VEC, vec):
                    bond.SetDoubleProp(prop, val)
                # Target atom coordinates
                for prop, val in zip(symbols.XYZU, xyz + vec):
                    bond.SetDoubleProp(prop, val)
                yield bond

    @property
    @functools.cache
    def res(self):
        """
        The residue.

        :return dict: residue number -> the atoms
        """
        atoms = collections.defaultdict(list)
        for atom in self.GetAtoms():
            atoms[atom.GetMonomerInfo().GetResidueNumber()].append(atom)
        return atoms

    def hash(self, bond):
        """
        Get the hash value of a bond.

        :param bond 'Chem.Bond': the bond to get the hash value of.
        :return tuple: moiety serial number, the moiety atom index,
            the bonded moiety serial number, the bonded moiety atom index
        """
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        nums = [x.GetMonomerInfo().GetSerialNumber() for x in atoms]
        maids = [x.GetIntProp(MAID) for x in atoms]
        return tuple(y for x in sorted(x for x in zip(nums, maids)) for y in x)

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

    def write(self, filename):
        """
        Write the polymer and monomer into sdf files.

        :param filename str: The file path to write into
        """
        try:
            maids = [x.GetIntProp(MAID) for x in self.GetAtoms()]
        except KeyError:
            pass
        else:
            self.SetProps([MAID])
            self.SetProp(MAID, ' '.join(map(str, maids)))
        with Chem.SDWriter(filename) as fh:
            fh.write(self)

    @classmethod
    def read(cls, filename):
        """
        Read molecule from file path.

        :param filename str: the file path to read molecule from.
        :return 'Chem.Mol': The molecule with properties.
        """
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        mol = next(suppl)
        Chem.GetSymmSSSR(mol)
        try:
            maids = mol.GetProp(MAID).split()
        except KeyError:
            pass
        else:
            for atom, maid in zip(mol.GetAtoms(), maids):
                atom.SetProp(MAID, maid)
        return mol
