# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module builds polymers.
"""
import collections
import functools

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


class Conf(structutils.Conf):

    AXIS = (1, 0, 0)

    def getAligned(self, bond=None):
        """
        Get the conformer coordinates aligned by the bond.

        :param bond `Chem.rdchem.Bond`: the residue starts from this bond
        :return `np.ndarray`: the conformer aligned by the bond.
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

        :param cap `rdkit.Chem.rdchem.Atom`: the capping atom.
        :return `np.ndarray`: the transformed coordinates.
        """
        if not bond:
            self.translate(-self.centroid())
            return self.GetPositions()

        cap_idx = bond.GetIntProp(Moieties.END)
        cap = self.GetOwningMol().GetAtomWithIdx(cap_idx)
        centroid = self.centroid(aids=[x.GetIdx() for x in cap.GetNeighbors()])
        self.translate(-centroid)
        coords = self.GetAtomPosition(cap_idx)
        rotation, _ = scipy.spatial.transform.Rotation.align_vectors(
            self.AXIS, [-coords.x, -coords.y, -coords.z])
        self.rotate(rotation=rotation)
        return self.GetPositions()


class Moiety(cru.Moiety):
    """
    Moiety.
    """
    Conf = Conf
    RES_NAME = 'res_name'
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
                    case self.RES_NAME:
                        info.setResidueName(value)
                    case self.RES_NUM:
                        info.SetResidueNumber(value)
                    case self.SERIAL:
                        info.SetSerialNumber(value)
            atom.SetMonomerInfo(info)

    @property
    @functools.cache
    def head(self):
        """
        Get the capping atom of the head.

        :return `Chem.Atom`: the capping atom of the head.
        """
        return self.getCapping(self.HEAD_ID)

    @property
    @functools.cache
    def tail(self):
        """
        Get the capping atom of the tail.

        :return `Chem.Atom`: the capping atom of the tail.
        """
        return self.getCapping(self.TAIL_ID)

    def extend(self, mol):
        """
        Extend the molecule by forming bond between the tail and the input head.

        :param mol 'Moiety': the moiety to extend with.
        :return 'Moiety': the extended molecule.
        """
        pair = self.tail[:1] + mol.head[:1]
        if len(pair) == 1:
            # Add implicit hydrogen as one input moiety is empty.
            atom = pair[0].GetNeighbors()[0]
            implicit_h = atom.GetIntProp(symbols.IMPLICIT_H)
            atom.SetIntProp(symbols.IMPLICIT_H, implicit_h + 1)
            atom.SetBoolProp(structutils.GrownMol.POLYM_HT, True)
        # FIXME: support multiple tails bonded to the copies of molecules
        # Increase the residue number of the mol
        nums = [x.GetMonomerInfo().GetResidueNumber() for x in self.GetAtoms()]
        start = max(nums) if nums else 0
        for atom in mol.GetAtoms():
            info = atom.GetMonomerInfo()
            info.SetResidueNumber(info.GetResidueNumber() + start)
        # Combine the molecules and remove the capping atoms
        edcombo = Chem.EditableMol(Chem.CombineMols(self, mol))
        for cap_aid in sorted([x.GetIdx() for x in pair], reverse=True):
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

    def EmbedMolecule(self, *args, **kwargs):
        """
        Embed the conformer.
        """
        for atom in self.stars:
            # Get XYZ with wild cards as carbon atoms
            atom.SetAtomicNum(6)
        super().EmbedMolecule(*args, **kwargs)
        for atom in self.stars:
            atom.SetAtomicNum(0)
        conf = self.GetConformer()
        for dihe in self.getDihes([x.GetIdx() for x in self.head],
                                  [x.GetIdx() for x in self.tail]):
            conf.setDihedralDeg(dihe, 180)


class Moieties(list, logutils.Base):
    """
    Build s polymer from moieties.
    """
    BEGIN = 'begin'
    END = 'end'

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
        self.length = {}

    def run(self):
        """
        Set up.
        """
        # Moieties
        frags = structure.Mol.MolFromSmiles(self.cru).GetMolFrags(asMols=True)
        self.extend(
            Moiety(x, info=dict(serial=i)) for i, x in enumerate(frags))
        if not self.mols:
            # Build molecule from monomers
            for moiety in self:
                for atom in moiety.GetAtoms():
                    atom.SetIntProp(MAID, atom.GetIdx())
            # FIXME: Support input sequence (e.g., AABA) and moiety ratios
            seq = [np.random.choice(self.mers) for _ in range(self.cru_num)]
            seq = [x.copy(i) for i, x in enumerate(seq)]
            chain = self.inr.extend(self.build(seq))
            terminated = chain.extend(self.ter)
            self.log(f"Polymer SMILES: {Chem.MolToSmiles(terminated)}")
            self.mols.append(terminated)
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

        :return list: regular molecules or polymer.
        """
        return [x for x in self if x.role == cru.REGULAR]

    @methodtools.lru_cache()
    @property
    def mers(self):
        """
        Get the monomers.

        :return list: the monomer moieties.
        """
        return [x for x in self if x.role == cru.MONOMER]

    def build(self, sequence):
        """
        Create bonds between a sequence of monomers.

        :param sequence list: monomers to create a chain.
        :return 'Moiety': the chain built from monomers.
        """
        mol = Moiety(functools.reduce(Chem.CombineMols, sequence))
        # FIXME: Support head-head and tail-tail coupling
        pairs = list(zip(mol.getCapping()[:-1], mol.getCapping(0)[1:]))
        edcombo, maids = Chem.EditableMol(mol), {}
        # Form bonds between the pres and nexs
        for caps in pairs:
            bonded = tuple([x.GetNeighbors()[0].GetIdx() for x in caps])
            edcombo.AddBond(*bonded, order=Chem.rdchem.BondType.SINGLE)
            maids[bonded] = [x.GetIntProp(MAID) for x in caps]
        # Record capping atoms moiety atom ids on the formed bonds
        chain = edcombo.GetMol()
        for bonded, (begin, end) in maids.items():
            bond = chain.GetBondBetweenAtoms(*bonded)
            bond.SetIntProp(self.BEGIN, begin)
            bond.SetIntProp(self.END, end)
        # Remove the capping atoms
        aids = [y.GetIdx() for x in pairs for y in x]
        editable = Chem.EditableMol(chain)
        for aid in sorted(aids, reverse=True):
            editable.RemoveAtom(aid)
        return Moiety(editable.GetMol())

    @methodtools.lru_cache()
    @property
    def inr(self):
        """
        Get the initiator.

        :return list: the initiator moiety.
        """
        return next((x for x in self if x.role == cru.INITIATOR), Moiety())

    @methodtools.lru_cache()
    @property
    def ter(self):
        """
        Get the terminator.

        :return list: the terminator moiety.
        """
        return next((x for x in self if x.role == cru.TERMINATOR), Moiety())

    def getLength(self,
                  bond,
                  marker='marker',
                  wild=Chem.MolFromSmiles(symbols.STAR)):
        """
        Get the length of a bond.

        :param bond 'Chem.Bond': the bond to get the hash value of.
        :param marker str: the marker.
        :param wild 'rdkit.Chem.rdchem.Molecule': wild card molecule.
        :return float: the bond length in Angstroms.
        """
        key = self.hash(bond)
        if key in self.length:
            return self.length[key]
        # Mark atoms to bond
        moieties = [self[int(x)].copy() for x in key[::2]]
        for moiety, idx in zip(moieties, key[1::2]):
            moiety.GetAtomWithIdx(idx).SetBoolProp(marker, True)
        # Combine moieties and add the bond
        cmol = Chem.CombineMols(*moieties)
        bonded = [x.GetIdx() for x in cmol.GetAtoms() if x.HasProp(marker)]
        edcombo = Chem.EditableMol(cmol)
        edcombo.AddBond(*bonded, order=Chem.rdchem.BondType.SINGLE)
        mol = Chem.DeleteSubstructs(edcombo.GetMol(), wild)
        # Measure the bond length
        mol = structure.Mol(mol)
        with rdkitutils.capture_logging():
            mol.EmbedMolecule(useRandomCoords=True,
                              randomSeed=self.options.seed)
        bonded = [x.GetIdx() for x in mol.GetAtoms() if x.HasProp(marker)]
        lgth = Chem.rdMolTransforms.GetBondLength(mol.GetConformer(), *bonded)
        return self.length.setdefault(key, lgth)

    def hash(self, bond):
        """
        Get the hash value of a bond.

        :param bond 'Chem.Bond': the bond to get the hash value of.
        :return tuple: (the moiety name, the moiety atom index,
            the bonded moiety name, the bonded moiety atom index)
        """
        atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        names = [x.GetMonomerInfo().GetSerialNumber() for x in atoms]
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
    Regular molecule or a polymer built from moieties.
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
        :param mol `rdkit.Chem.rdchem.Mol`: the molecule or polymer
        :param mol_num int: the number of molecules of this type of polymer
        :param options 'argparse.Namespace': command-line options
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
        Embed the molecule with coordinates.
        """
        with rdkitutils.capture_logging(self.logger):
            # ERROR UFFTYPER: Unrecognized charge state for atom: 0 (Mg+2)
            # WARNING UFFTYPER: Warning: hybridization set to SP3 for atom 0
            kwargs = dict(useRandomCoords=True, randomSeed=self.options.seed)
            if not self.polym:
                super().EmbedMolecule(**kwargs)
                return
            for moiety in self.moieties:
                moiety.EmbedMolecule(**kwargs)

        Chem.GetSymmSSSR(self)
        conf = structure.Conf(self.GetNumAtoms())
        self.AddConformer(conf, assignId=True)

        bonds = [None]
        while bonds:
            bond = bonds.pop()
            bonds.extend(self.setConformer(bond))

    def setConformer(self, pre, res=0):
        """
        Partially set the conformer of one residue (moiety).

        :param bond Chem.rdchem.Bond: from previous to the current moiety.
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
                xyz = xyzs[atom.GetIntProp(MAID)]
                bond = self.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                vec = xyzs[bond.GetIntProp(Moieties.BEGIN)] - xyz
                vec *= self.moieties.getLength(bond) / np.linalg.norm(vec)
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
            res_num = atom.GetMonomerInfo().GetResidueNumber()
            atoms[res_num].append(atom)
        return atoms

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
                maids = [x.GetIntProp(MAID) for x in mol.GetAtoms()]
            except KeyError:
                fh.write(mol)
                return
            mol.SetProps([MAID])
            mol.SetProp(MAID, ' '.join(map(str, maids)))
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
            maids = mol.GetProp(MAID).split()
        except KeyError:
            return mol
        for atom, maid in zip(mol.GetAtoms(), maids):
            atom.SetProp(MAID, maid)
        return mol
