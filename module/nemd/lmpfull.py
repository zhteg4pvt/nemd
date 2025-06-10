# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses a LAMMPS data file in the atomic atom_style.

Atoms section: atom-ID molecule-ID atom-type q x y z
"""
import collections
import functools
import itertools

import methodtools
import numpy as np
import pandas as pd
from rdkit import Chem

from nemd import builtinsutils
from nemd import lmpatomic
from nemd import lmpfix
from nemd import lmpin
from nemd import numpyutils
from nemd import oplsua
from nemd import symbols

TYPE_ID = lmpatomic.TYPE_ID
ATOM1 = lmpatomic.ATOM1
ATOM2 = 'atom2'
ATOM3 = 'atom3'
ATOM4 = 'atom4'
ENE = 'ene'


class Mass(lmpatomic.Mass):
    """
    See parent.
    """

    @classmethod
    def fromAtoms(cls, atoms):
        """
        See parent.
        """
        masses = super().fromAtoms(atoms)
        masses.comment = masses.comment.str.cat(atoms.descr.values, sep=' ')
        masses.comment = masses.comment.str.cat(map(str, atoms.index), sep=' ')
        return masses


class PairCoeff(lmpatomic.Base):
    """
    The pair coefficients between non-bonded atoms.
    """
    NAME = 'Pair Coeffs'
    LABEL = Mass.LABEL
    COLUMNS = [ENE, 'dist']


class BondCoeff(PairCoeff):
    """
    The bond coefficients between bonded atoms.
    """
    NAME = 'Bond Coeffs'
    LABEL = 'bond types'


class AngleCoeff(BondCoeff):
    """
    The angle coefficients between bonded atoms.
    """
    NAME = 'Angle Coeffs'
    LABEL = 'angle types'
    COLUMNS = [ENE, 'deg']


class DihedralCoeff(AngleCoeff):
    """
    The dihedral coefficients between bonded atoms.
    """
    NAME = 'Dihedral Coeffs'
    LABEL = 'dihedral types'
    COLUMNS = ['k1', 'k2', 'k3', 'k4']


class ImproperCoeff(AngleCoeff):
    """
    The improper coefficients between bonded atoms in the system.
    """
    NAME = 'Improper Coeffs'
    LABEL = 'improper types'
    COLUMNS = ['k', 'd', 'n']


class Id(lmpatomic.Id):
    """
    See parent.
    """
    MOL_ID = 'mol_id'
    COLUMNS = [ATOM1, MOL_ID, TYPE_ID]

    @classmethod
    def fromAtoms(cls, atoms, idx=0):
        """
        See parent.

        :param idx int: the molecule id.
        """
        return cls([[x.GetIdx(), idx, x.GetIntProp(TYPE_ID)] for x in atoms])

    def to_numpy(self, *args, col_id=COLUMNS.index(MOL_ID), gid=0):
        """
        See parent.

        :param col_id int: the column index of the molecule id
        :param idx int: the molecule id.
        """
        array = super().to_numpy(*args)
        array[:, col_id] = gid
        return array


class Atom(lmpatomic.Atom):
    """
    See parent.
    """
    ID_COLS = [Id.MOL_ID]
    COLUMNS = Id.COLUMNS + ['Charge'] + symbols.XYZU
    FMT = '%i %i %i %.4f %.4f %.4f %.4f'


class Bond(lmpatomic.Id):
    """
    The bond information including the bond type and the atom ids.
    """
    NAME = 'Bonds'
    LABEL = 'bonds'
    ID_COLS = [ATOM1, ATOM2]
    COLUMNS = [TYPE_ID] + ID_COLS
    FMT = '%i'
    SLICE = slice(1, None)

    def __init__(self, *args, dtype=np.uint32, **kwargs):
        """
        :param dtype 'type': the data type of the Series
        """
        super().__init__(*args, dtype=dtype, **kwargs)

    @classmethod
    def fromAtoms(cls, atoms, ff):
        """
        Construct instance from atoms and force field block.

        :param atoms list: each sublist contains atoms of certain topology.
        :param ff `oplsua.[Bond|Angle|Dihedral|Improper]`: force field block.
        :return `cls`: the bond | angle | diehdral | improper information.
        """
        return cls([[ff.match(x)] + [y.GetIdx() for y in x] for x in atoms])

    def getPairs(self):
        """
        Get atom pairs from the topology connectivity.

        :return list of tuple: the atom pairs
        """
        head_tail = self[[self.ID_COLS[0], self.ID_COLS[-1]]].values
        return list(map(tuple, np.sort(head_tail, axis=1)))


class Angle(Bond):
    """
    The angle information including the angle type and the atom ids.
    """
    NAME = 'Angles'
    LABEL = 'angles'
    ID_COLS = [ATOM1, ATOM2, ATOM3]
    COLUMNS = [TYPE_ID] + ID_COLS

    @classmethod
    def fromAtoms(cls, atoms, ff, impropers):
        """
        See parent.
        """
        angles = super().fromAtoms(atoms, ff)
        # Drop angles due to the improper angles (see Mol.getImproperAtoms)
        angles.dropLowest(impropers.getAngles(), ff.ene.to_dict())
        return angles

    def dropLowest(self, angles, ene):
        """
        Drop the angle of the lowest energy in each row.

        :param angles nx3x3 np.ndarray: each sublist contains three angles.
        :param ene `pd.Series`: type ids -> the energy.
        """
        indices = [[self.row.index(tuple(y)) for y in x] for x in angles]
        index = [self.type_id.loc[x].map(ene).idxmin() for x in indices]
        self.drop(index=index, inplace=True)

    @methodtools.lru_cache()
    @property
    def row(self):
        """
        The mapping from the values of a row to the index.

        :return `oplsua.BondIndex` (sub-)class: the mapping object.
        """
        return oplsua.AngleIndex(self[self.ID_COLS].reset_index())


class Dihedral(Bond):
    """
    The dihedral angle information including the dihedral type and the atom ids.
    """
    NAME = 'Dihedrals'
    LABEL = 'dihedrals'
    ID_COLS = [ATOM1, ATOM2, ATOM3, ATOM4]
    COLUMNS = [TYPE_ID] + ID_COLS


class Improper(Dihedral):
    """
    The improper angle information including the improper type and the atom ids.
    """
    NAME = 'Impropers'
    LABEL = 'impropers'

    def getPairs(self):
        """
        See parent.
        """
        nvertices = self[self.ID_COLS].drop(columns=[ATOM3]).values
        pairs = (y for x in nvertices for y in itertools.combinations(x, 2))
        return [tuple(sorted(x)) for x in pairs]

    def getAngles(self, columns=(ATOM2, ATOM1, ATOM4)):
        """
        Get the angle from each topology connectivity.

        :param columns tuple: non-vertex atoms
        :return nx3x3 ndarray: each sublist contains three angles.
        """
        cols = [[x, ATOM3, y] for x, y in itertools.combinations(columns, 2)]
        return np.stack([self[x].values for x in cols], axis=1)


class Conf(lmpatomic.Conf):
    """
    Customized with id, topology, measurement, and internal coordinate.
    """

    @property
    def ids(self):
        """
        The ids of this conformer.

        :return `np.ndarray`: global, molecule, and type ids.
        """
        return self.GetOwningMol().ids.to_numpy(self.gids, gid=self.gid)

    @property
    def bonds(self):
        """
        Bonds.

        :return `np.ndarray`: bond ids and bonded atom ids.
        """
        return self.GetOwningMol().bonds.to_numpy(self.gids)

    @property
    def angles(self):
        """
        Angles.

        :return `np.ndarray`: angle ids and connected atom ids.
        """
        return self.GetOwningMol().angles.to_numpy(self.gids)

    @property
    def dihedrals(self):
        """
        Dihedral angles.

        :return `np.ndarray`: dihedral ids and connected atom ids.
        """
        return self.GetOwningMol().dihedrals.to_numpy(self.gids)

    @property
    def impropers(self):
        """
        Improper angles.

        :return `np.ndarray`: improper ids and connected atom ids.
        """
        return self.GetOwningMol().impropers.to_numpy(gids=self.gids)

    def setGeo(self, aids, val):
        """
        Set the bond length, angle degree, or dihedral angle.

        :param aids list: atom ids
        :param val float: the value to set.
        """
        if val is None:
            return
        match len(aids):
            case 2:
                Chem.rdMolTransforms.SetBondLength(self, *aids, val)
            case 3:
                Chem.rdMolTransforms.SetAngleDeg(self, *aids, val)
            case 4:
                Chem.rdMolTransforms.SetDihedralDeg(self, *aids, val)

    def measure(self, aids=None, name=None):
        """
        Measure the bond length, angle degree, or dihedral angle.

        :param aids list: atom ids
        :param name str: the measurement name.
        :return np.ndarray or float: the measurement.
        """
        if aids is None:
            match = self.GetOwningMol().getSubstructMatch()
            aids, name = match.tolist(), match.name
        value = super().measure(aids)
        if isinstance(value, float):
            return builtinsutils.Float(value, name=name)
        return value


class Mol(lmpatomic.Mol):
    """
    See parent.
    """
    Conf = Conf
    Atom = Atom
    Id = Id

    def setUp(self, *args, **kwargs):
        """
        See parent.
        """
        super().setUp(*args, **kwargs)
        self.setInternal()
        self.setSubstruct()
        self.updateAll()

    def type(self):
        """
        See parent.
        """
        self.ff.type(self)

    def setInternal(self):
        """
        Set the internal coordinates.
        """
        tpl = self.GetConformer()
        for tid, *ids in self.bonds.values:
            tpl.setGeo(list(map(int, ids)), self.ff.bonds.loc[tid].dist)
        for tid, *ids in self.angles.values:
            tpl.setGeo(list(map(int, ids)), self.ff.angles.loc[tid].deg)

    def setSubstruct(self):
        """
        Set substructure.
        """
        try:
            value = self.struct.options.substruct[1]
        except (TypeError, AttributeError):
            return
        self.GetConformer().setGeo(self.getSubstructMatch(), value)

    def getSubstructMatch(self, gid=False):
        """
        Get substructure match.

        :param gid bool: whether to return global atom ids.
        :return Series: the atom ids of the substructure match.
        """
        try:
            struct = self.struct.options.substruct[0]
        except (TypeError, AttributeError):
            return pd.Series([])
        struct = Chem.MolFromSmiles(struct)
        if not self.HasSubstructMatch(struct):
            return pd.Series([])
        ids = self.GetSubstructMatch(struct)
        if gid:
            ids = self.GetConformer().gids[list(ids)]
        match len(ids):
            case 1:
                name = 'coordinates (angstrom)'
            case 2:
                name = 'distance (angstrom)'
            case 3:
                name = 'angle (degree)'
            case 4:
                name = 'dihedral (degree)'
            case _:
                name = None
        return pd.Series(ids, name=name)

    def updateAll(self):
        """
        Update all conformers.
        """
        if not self.GetNumConformers():
            return
        xyz = self.GetConformer().GetPositions()
        for conf in self.GetConformers():
            conf.setPositions(xyz)

    @property
    @functools.cache
    def charges(self):
        """
        The charges.

        :return `np.ndarray`: the atomic charges.
        """
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        return self.ff.charges.loc[type_ids].values + self.nbr_charge

    @property
    @functools.cache
    def bonds(self):
        """
        The bonds.

        :return `Bond`: the bonds
        """
        atoms = [[x.GetBeginAtom(), x.GetEndAtom()] for x in self.GetBonds()]
        return Bond.fromAtoms(atoms, self.ff.bonds)

    @property
    @functools.cache
    def angles(self):
        """
        The angles.

        :return Angle: the angles.
        """
        return Angle.fromAtoms(self.getAngle(), self.ff.angles, self.impropers)

    def getAngle(self):
        """
        Get all three angle atoms from the input middle atom.

        :return generator: each sublist contains three atoms.
        """
        for atom in self.GetAtoms():
            for atom1, atom3 in itertools.combinations(atom.GetNeighbors(), 2):
                yield atom1, atom, atom3

    @property
    @functools.cache
    def dihedrals(self):
        """
        The dihedral angles.

        :return Dihedral: the dihedral types and atoms forming each dihedral.
        """
        return Dihedral.fromAtoms(self.getDehedral(), self.ff.dihedrals)

    def getDehedral(self, key=lambda x: x.GetBeginAtom().GetIdx()):
        """
        Get the dihedral atoms of this molecule.

        :param key func: to sort the bonds.
        :return generator: each sublist has four atoms forming a dihedral angle.
        """
        for bond in sorted(self.GetBonds(), key=key):
            second, third = bond.GetBeginAtom(), bond.GetEndAtom()
            for first in second.GetNeighbors():
                if first.GetIdx() != third.GetIdx():
                    for fourth in third.GetNeighbors():
                        if fourth.GetIdx() != second.GetIdx():
                            yield first, second, third, fourth

    @property
    @functools.cache
    def impropers(self):
        """
        The improper angles.

        :return Improper: the improper types and atoms forming each improper.
        """
        return Improper.fromAtoms(self.getImproper(), self.ff.impropers)

    def getImproper(self):
        """
        Set improper angles based on center atoms and neighbor symbols.

        1) sp2 sites and united atom CH groups (sp3 carbons) needs improper
        FIXME: missing improper for sp3 N with lone pair (e.g. ammonia)
        2) No rules for a center atom. (Charmm asks order for symmetricity)
        3) Number of internal geometry variables (3N_atom – 6) deletes one angle

        1) Improper Senario
        When the Weiner et al. (1984,1986) force field was developed, improper
        torsions were designated for specific sp2 sites, as well as for united
        atom CH groups - sp3 carbons with one implicit hydrogen.
        Ref: http://ambermd.org/Questions/improp.html

        2) Center Definition
        There are no rules for a center atom. You simply define two planes, each
        defined by three atoms. The angle is given by the angle between these
        two planes. (from hess)
        ref: https://gromacs.bioexcel.eu/t/the-atom-order-i-j-k-l-in-defining-an
        -improper-dihedral-in-gromacs-using-the-opls-aa-force-field/3658

        The CHARMM convention in the definition of improper torsion angles is to
        list the central atom in the first position, while no rule exists for how
        to order the other three atoms.
        ref: Symmetrization of the AMBER and CHARMM Force Fields, J. Comput. Chem.

        "A New Force Field for Molecular Mechanical Simulation of Nucleic Acids
        and Proteins" uses the third as the center.

        3) Angle Deletion
        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models

        :return generator: four atoms forming an improper angle.
        """
        # LAMMPS recommends the first to be the center, while the prm and
        # literature take the third as the center.

        # First or third acts exactly the same for planar scenario as both 0 deg
        # and 180 deg imply in plane. The chosen center defines the plane. The
        # third as center uses 120 deg as the equilibrium improper angle, which
        # the first as center uses 45 deg. We take the third as the center, and
        # there is no special treatment to the order of other atoms.
        for atom in self.GetAtoms():
            if atom.GetTotalDegree() != 3:
                continue
            match atom.GetSymbol():
                case symbols.CARBON:
                    # Planar Sp2 carbonyl carbon (R-COOH)
                    # tetrahedral Sp3 carbon with one implicit H (CHR1R2R3)
                    pass
                case symbols.NITROGEN:
                    if atom.GetHybridization(
                    ) != Chem.rdchem.HybridizationType.SP2:
                        continue
                    # Sp2 N in Amino Acid or Dimethylformamide
                case _:
                    continue
            atoms = list(atom.GetNeighbors())
            atoms.insert(2, atom)
            yield atoms

    @property
    def molecular_weight(self):
        """
        The molecular weight.

        :return float: the total molecular weight.
        """
        return self.ff.molecular_weight(self)

    @property
    @functools.cache
    def nbr_charge(self):
        """
        Balance the charge when residues are not neutral.

        :return `np.ndarray`: the charge of each atom due to connected neighbors.
        """
        # residual num: residual charge
        res_charge = collections.defaultdict(float)
        for atom in self.GetAtoms():
            res_num = atom.GetIntProp(symbols.RES_NUM)
            type_id = atom.GetIntProp(TYPE_ID)
            res_charge[res_num] += self.ff.charges.loc[type_id].q

        # residual num: largest atomic charge, connected atom in another residual
        res_atom = {}
        for bond in self.GetBonds():
            batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
            bres_num = batom.GetIntProp(symbols.RES_NUM)
            eres_num = eatom.GetIntProp(symbols.RES_NUM)
            if bres_num == eres_num:
                continue
            # Bonded atoms in different residuals
            for atom, natom in [[batom, eatom], [eatom, batom]]:
                res_num = atom.GetIntProp(symbols.RES_NUM)
                if not res_charge[res_num]:
                    continue
                charge = abs(self.ff.charges.loc[atom.GetIntProp(TYPE_ID)].q)
                if charge > res_atom.get(res_num, (0, 0))[0]:
                    res_atom[res_num] = (charge, natom.GetIdx())

        # The atom of the largest charge in a residual holds the total residual
        # charge so that connected atom in another residue can subtract it.

        nbr_charge = np.zeros(self.GetNumAtoms())
        for res_num, (_, idx) in res_atom.items():
            nbr_charge[idx] -= res_charge[res_num]
        return nbr_charge.reshape(-1, 1)


class In(lmpin.In):
    """
    Class to write out LAMMPS in script.
    """
    FULL = 'full'
    V_UNITS = lmpin.In.REAL
    V_ATOM_STYLE = FULL
    DEFAULT_CUT = symbols.DEFAULT_CUT

    def setup(self):
        """
        Write the setup section including unit, topology styles, and specials.
        """
        super().setup()
        self.fh.write(f"bond_style harmonic\n")
        self.fh.write(f"angle_style harmonic\n")
        self.fh.write("dihedral_style opls\n")
        self.fh.write("improper_style cvff\n")
        self.fh.write("special_bonds lj/coul 0 0 0.5\n")

    def pair(self):
        """
        Write pair style, coefficients, and mixing rules as well as k-space.
        """
        pair_style = 'lj/cut/coul/long' if self.hasCharge() else 'lj/cut'
        self.fh.write(f"{self.PAIR_STYLE} {pair_style} {self.DEFAULT_CUT}\n")
        self.fh.write(f"pair_modify mix geometric\n")
        if self.hasCharge():
            self.fh.write(f"kspace_style pppm 0.0001\n")

    def hasCharge(self):
        """
        Whether any atom has non-zero charge.

        :return bool: True if any atom has non-zero charge.
        """
        return True


class Struct(lmpatomic.Struct, In):
    """
    Customized for molecules with bonds.
    """
    Id = Id
    Atom = Atom
    Mol = Mol

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, options=options, **kwargs)
        In.__init__(self, options=options)
        self.atm_types = numpyutils.IntArray(self.ff.atoms.index.size)
        self.bnd_types = numpyutils.IntArray(self.ff.bonds.index.size)
        self.ang_types = numpyutils.IntArray(self.ff.angles.index.size)
        self.dihe_types = numpyutils.IntArray(self.ff.dihedrals.index.size)
        self.impr_types = numpyutils.IntArray(self.ff.impropers.index.size)

    def setTypeMap(self, mol):
        """
        Set the type map for atoms, bonds, angles, dihedrals, and impropers.

        :param mol `Mol`: add this molecule to the structure
        """
        self.atm_types[[x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]] = True
        self.bnd_types[mol.bonds.type_id] = True
        self.ang_types[mol.angles.type_id] = True
        self.dihe_types[mol.dihedrals.type_id] = True
        self.impr_types[mol.impropers.type_id] = True

    def writeData(self):
        """
        Write out a LAMMPS datafile or return the content.
        """
        with open(self.datafile, 'w') as self.hdl:
            self.hdl.write(f"{self.DESCR.format(style=self.V_ATOM_STYLE)}\n\n")
            # Topology counting
            self.atoms.writeCount(self.hdl)
            self.bonds.writeCount(self.hdl)
            self.angles.writeCount(self.hdl)
            self.dihedrals.writeCount(self.hdl)
            self.impropers.writeCount(self.hdl)
            self.hdl.write("\n")
            # Type counting
            self.masses.writeCount(self.hdl)
            self.bond_coeffs.writeCount(self.hdl)
            self.angle_coeffs.writeCount(self.hdl)
            self.dihedral_coeffs.writeCount(self.hdl)
            self.improper_coeffs.writeCount(self.hdl)
            self.hdl.write("\n")
            # Box boundary
            self.box.write(self.hdl)
            # Interaction coefficients
            self.masses.write(self.hdl)
            self.pair_coeffs.write(self.hdl)
            self.bond_coeffs.write(self.hdl)
            self.angle_coeffs.write(self.hdl)
            self.dihedral_coeffs.write(self.hdl)
            self.improper_coeffs.write(self.hdl)
            # Topology details
            self.atoms.write(self.hdl)
            self.bonds.write(self.hdl)
            self.angles.write(self.hdl)
            self.dihedrals.write(self.hdl)
            self.impropers.write(self.hdl)

    def getAtomic(self):
        """
        See parent.
        """
        return zip(self.ids.values, self.charges, self.GetPositions())

    @property
    @functools.cache
    def charges(self):
        """
        Atoms charges.

        :return `np.ndarray`: the charges of all atoms.
        """
        charges = [x.GetOwningMol().charges for x in self.conf]
        return np.concatenate(charges or [[]])

    @property
    @functools.cache
    def bonds(self):
        """
        Bonds.

        :return 'np.ndarray': bond types and bonded atom ids.
        """
        bonds = [y.bonds for x in self.mols for y in x.confs]
        return Bond.concatenate(bonds, self.bnd_types)

    @property
    @functools.cache
    def angles(self):
        """
        Angle.

        :return 'np.ndarray': angle types and connected atom ids.
        """
        angles = [y.angles for x in self.mols for y in x.confs]
        return Angle.concatenate(angles, self.ang_types)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Dihedral angles.

        :return 'np.ndarray': dihedral types and connected atom ids.
        """
        dihes = [y.dihedrals for x in self.mols for y in x.confs]
        return Dihedral.concatenate(dihes, self.dihe_types)

    @property
    @functools.cache
    def impropers(self):
        """
        Improper angles.

        :return 'np.ndarray': improper types and connected atom ids.
        """
        imprps = [y.impropers for x in self.mols for y in x.confs]
        return Improper.concatenate(imprps, self.impr_types)

    @property
    def masses(self):
        """
        Atom masses.

        :return `Mass`: mass of each type of atom.
        """
        return Mass.fromAtoms(self.ff.atoms.loc[self.atm_types.on])

    @property
    def pair_coeffs(self):
        """
        Non-bonded atom pair coefficients.

        :return `PairCoeff`: the interaction between non-bond atoms.
        """
        vdws = self.ff.vdws.loc[self.atm_types.on]
        return PairCoeff([[x.ene, x.dist] for x in vdws.itertuples()])

    @property
    def bond_coeffs(self):
        """
        Bond coefficients.

        :return `BondCoeff`: the interaction between bonded atoms.
        """
        bonds = self.ff.bonds.loc[self.bnd_types.on]
        return BondCoeff([[x.ene, x.dist] for x in bonds.itertuples()])

    @property
    def angle_coeffs(self):
        """
        Angle coefficients.

        :return `AngleCoeff`: the three-atom angle interaction coefficients
        """
        angles = self.ff.angles.loc[self.ang_types.on]
        return AngleCoeff([[x.ene, x.deg] for x in angles.itertuples()])

    @property
    def dihedral_coeffs(self):
        """
        Dihedral coefficients.

        :return `DihedralCoeff`: the four-atom torsion interaction coefficients
        """
        dihes = self.ff.dihedrals.loc[self.dihe_types.on]
        return DihedralCoeff([[x.k1, x.k2, x.k3, x.k4]
                              for x in dihes.itertuples()])

    @property
    def improper_coeffs(self):
        """
        Improper coefficients.

        :return `ImproperCoeff`: the four-atom improper interaction coefficients
        """
        imprps = self.ff.impropers.loc[self.impr_types.on]
        # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
        # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
        ks = imprps.deg.map(lambda x: 1 if x == 0. else -1)
        return ImproperCoeff(list(zip(*[imprps.ene, ks, imprps.n_parm])))

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return sum([x.molecular_weight * len(x.confs) for x in self.mols])

    def hasCharge(self):
        """
        Whether any atom has charge.

        :return `bool`: Whether any atom has charge
        """
        return self.charges.size and not np.isclose(self.charges, 0).any()

    def minimize(self, *args, geo=None, **kwargs):
        """
        See parent.
        """
        if geo is None:
            mol = next((x for x in self.mols if x.GetNumConformers()), None)
            gids = mol.getSubstructMatch(gid=True)
            if not gids.empty:
                geo = f"{gids.name.split()[0]} {' '.join(map(str, gids + 1))}"
        super().minimize(*args, geo=geo, **kwargs)

    def shake(self):
        """
        Write fix shake command to enforce constant bond length and angel values.
        """
        bonds = self.getRigid(self.bnd_types, self.ff.bonds)
        angles = self.getRigid(self.ang_types, self.ff.angles)
        super().shake(bonds=bonds, angles=angles)

    def getRigid(self, type_map, ff):
        """
        Get the rigid topology types.

        :param type_map `np.ndarray`: the bond or angle map.
        :param ff `oplsua.[Bond|Angle]`: force field block for hydrogen info.
        :return str: the rigid topology types.
        """
        type_ids = type_map.on[ff.has_h[type_map.on]]
        # type_map starts from 0, but lammps starts from 1
        return ' '.join(map(str, type_map.index(type_ids) + 1))

    def getWarnings(self):
        """
        Get warnings for the structure.

        :return generator of str: the warnings on structure checking.
        """
        net_charge = round(self.charges.sum().sum(), 4)
        if net_charge:
            yield f'The system has a net charge of {net_charge:.4f}'
        if self.box.empty:
            return
        min_span = self.box.span.min()
        if min_span < self.DEFAULT_CUT * 2:
            yield f'Box span ({min_span:.2f} {symbols.ANGSTROM}) < ' \
                  f'{self.DEFAULT_CUT * 2:.2f} {symbols.ANGSTROM} (cutoff x 2)'

    @property
    @functools.cache
    def ff(self):
        """
        See parent.
        """
        ff = self.options.force_field if self.options else symbols.OPLSUA_TIP3P
        return oplsua.Parser.get(*ff[1:])


class Reader(lmpatomic.Reader):
    """
    See parent.
    """
    Atom = Atom
    Mass = Mass
    NAMES = [
        lmpatomic.Mass, PairCoeff, BondCoeff, AngleCoeff, DihedralCoeff,
        ImproperCoeff, Atom, Bond, Angle, Dihedral, Improper
    ]
    NAMES = {x.NAME: x.LABEL for x in NAMES}

    @property
    @functools.cache
    def pair_coeffs(self):
        """
        Paser the pair coefficient section.

        :return `PairCoeff`: the pair coefficients between non-bonded atoms.
        """
        return self.fromLines(PairCoeff)

    @property
    @functools.cache
    def bond_coeffs(self):
        """
        Paser the bond coefficients.

        :return `BondCoeff`: the interaction between bonded atoms.
        """
        return self.fromLines(BondCoeff)

    @property
    @functools.cache
    def angle_coeffs(self):
        """
        Paser the angle coefficients.

        :return `AngleCoeff`: the three-atom angle interaction coefficients
        """
        return self.fromLines(AngleCoeff)

    @property
    @functools.cache
    def dihedral_coeffs(self):
        """
        Paser the dihedral coefficients.

        :return `DihedralCoeff`: the four-atom torsion interaction coefficients
        """
        return self.fromLines(DihedralCoeff)

    @property
    @functools.cache
    def improper_coeffs(self):
        """
        Paser the improper coefficients.

        :return `ImproperCoeff`: the four-atom improper interaction coefficients
        """
        return self.fromLines(ImproperCoeff)

    @property
    @functools.cache
    def bonds(self):
        """
        Parse the atom section for atom id and molecule id.

        :return `Bond`: bond id, type id, and bonded atom ids.
        """
        return self.fromLines(Bond)

    @property
    @functools.cache
    def angles(self):
        """
        Parse the angle section for angle id and constructing atoms.

        :return `Angle`: angle id, type id, and atom ids.
        """
        return self.fromLines(Angle)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.

        :return `Dihedral`: dihedral id, type id, and atom ids.
        """
        return self.fromLines(Dihedral)

    @property
    @functools.cache
    def impropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.

        :return `Improper`: improper id, type id, and atom ids.
        """
        return self.fromLines(Improper)

    @property
    @functools.cache
    def mols(self):
        """
        The atom ids grouped by molecules.

        :return dict: keys are molecule ids and values are atom global ids.
        """
        mols = collections.defaultdict(list)
        for gid, mol_id, in self.atoms.mol_id.items():
            mols[mol_id].append(gid)
        return mols

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.masses.mass[self.atoms.type_id].sum()

    def allClose(self, other, atol=1e-08, rtol=1e-05, equal_nan=True):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other float: the other data reader to compare against.
        :param atol float: The relative tolerance parameter.
        :param rtol float: The absolute tolerance parameter.
        :param equal_nan bool: If True, NaNs are considered close.
        :return bool: whether two data are close.
        """
        kwargs = dict(atol=atol, rtol=rtol, equal_nan=equal_nan)
        if not super().allClose(other, **kwargs):
            return False
        if not self.pair_coeffs.allClose(other.pair_coeffs, **kwargs):
            return False
        if not self.bond_coeffs.allClose(other.bond_coeffs, **kwargs):
            return False
        if not self.angle_coeffs.allClose(other.angle_coeffs, **kwargs):
            return False
        if not self.dihedral_coeffs.allClose(other.dihedral_coeffs, **kwargs):
            return False
        if not self.improper_coeffs.allClose(other.improper_coeffs, **kwargs):
            return False
        if not self.bonds.allClose(other.bonds, **kwargs):
            return False
        if not self.angles.allClose(other.angles, **kwargs):
            return False
        if not self.dihedrals.allClose(other.dihedrals, **kwargs):
            return False
        if not self.impropers.allClose(other.impropers, **kwargs):
            return False
        return True
