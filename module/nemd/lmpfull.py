# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses a LAMMPS data file in the full atom_style.

Atoms section: atom-ID molecule-ID atom-type q x y z
"""
import collections
import functools
import io
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem

from nemd import lammpsin
from nemd import lmpatomic
from nemd import numpyutils
from nemd import symbols

TYPE_ID = lmpatomic.TYPE_ID
ATOM1 = lmpatomic.ATOM1
ATOM2 = 'atom2'
ATOM3 = 'atom3'
ATOM4 = 'atom4'


class Mass(lmpatomic.Mass):
    """
    Decorate the parent class with additional comments.
    """

    CMT_FMT = "{descr} {symbol} {idx}"
    CMT_RE = r'(\w+)\s+\d+$'

    @classmethod
    def fromAtoms(cls, atoms):
        """
        Construct a mass instance from atoms.

        :param atoms `pd.DataFrame`: the atoms.
        :return `cls`: the mass instance.
        """
        mass = [x.mass for x in atoms.itertuples()]
        cmt = [
            f"{x.descr} {x.symbol} {x.Index + 1}" for x in atoms.itertuples()
        ]
        return cls({cls.MASS: mass, cls.COMMENT: cmt})


class BondCoeff(lmpatomic.PairCoeff):
    """
    The bond coefficients between bonded atoms in the system.
    """
    NAME = 'Bond Coeffs'
    LABEL = 'bond types'


class AngleCoeff(lmpatomic.Base):
    """
    The angle coefficients between bonded atoms in the system.
    """
    NAME = 'Angle Coeffs'
    COLUMNS = [lmpatomic.ENE, 'deg']
    LABEL = 'angle types'


class DihedralCoeff(AngleCoeff):
    """
    The dihedral coefficients between bonded atoms in the system.
    """
    NAME = 'Dihedral Coeffs'
    COLUMNS = ['k1', 'k2', 'k3', 'k4']
    LABEL = 'dihedral types'


class ImproperCoeff(AngleCoeff):
    """
    The improper coefficients between bonded atoms in the system.
    """
    NAME = 'Improper Coeffs'
    COLUMNS = ['k', 'd', 'n']
    LABEL = 'improper types'


class Charge(lmpatomic.XYZ):
    """
    The charge of every atom.
    """

    NAME = 'Charge'
    COLUMNS = ['charge']


class Atom(lmpatomic.Atom):
    """
    See parent class.
    """

    MOL_ID = 'mol_id'
    COLUMNS = [ATOM1, MOL_ID, TYPE_ID]


class AtomBlock(lmpatomic.AtomBlock):
    """
    See parent class.
    """
    ID_COLS = [Atom.MOL_ID]
    TYPE_COL = [TYPE_ID]
    COLUMNS = Atom.COLUMNS + Charge.COLUMNS + lmpatomic.XYZ.COLUMNS
    FMT = '%i %i %i %.4f %.4f %.4f %.4f'


class Bond(Atom):
    """
    The bond information including the bond type and the atom ids.
    """

    NAME = 'Bonds'
    ID_COLS = [ATOM1, ATOM2]
    COLUMNS = [TYPE_ID] + ID_COLS
    DEFAULT_DTYPE = np.uint32
    LABEL = 'bonds'
    FMT = '%i'

    def __init__(self,
                 data=None,
                 type_ids=None,
                 aids=None,
                 dtype=DEFAULT_DTYPE,
                 **kwargs):
        """
        :param data: ndarray, Iterable, dict, or DataFrame
        :type data: the content to create dataframe
        :param type_ids: type ids of the aids
        :type type_ids: list of int
        :param aids: each sublist contains atom ids matching with one type id
        :type aids: list of list
        :param dtype: 'the data type of the Series
        :type dtype: 'type'
        """
        if data is None and type_ids is not None and aids is not None:
            data = [[x] + y for x, y in zip(type_ids, aids)]
        if data is None:
            data = {x: pd.Series(dtype=dtype) for x in self.COLUMNS}
        super().__init__(data=data, **kwargs)

    def getPairs(self, step=1):
        """
        Get the atom pairs from each topology connectivity.

        :param step int: the step when slicing the atom ids
        :return list of tuple: the atom pairs
        """
        slices = slice(None, None, step)
        return [tuple(sorted(x[slices])) for x in self[self.ID_COLS].values]

    def getRigid(self, has_h):
        """
        Get the rigid topology types.

        :param has_h `ndarray`: whether each type has hydrogen involved
        :return `DataFrame`: the rigid topology types.
        """
        ids = self[TYPE_ID].unique()
        return pd.DataFrame({self.NAME: ids[has_h[ids]] if len(ids) else []})


class Angle(Bond):
    """
    The angle information including the angle type and the atom ids.
    """

    NAME = 'Angles'
    ID_COLS = [ATOM1, ATOM2, ATOM3]
    COLUMNS = [TYPE_ID] + ID_COLS
    LABEL = 'angles'
    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['id_map']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_map = None

    def getPairs(self, step=2):
        """
        Set parent.
        """
        return super(Angle, self).getPairs(step=step)

    def select(self, atom_ids):
        """
        Get the angles indexes from atom ids.

        :param atom_ids `numpy.ndarray`: each row is atom ids from one angle
        :return Angle: the selected angles matching the input atom ids.
        """
        if self.id_map is None:
            shape = 0 if self.empty else self[self.ID_COLS].max().max() + 1
            self.id_map = np.zeros([shape] * len(self.ID_COLS), dtype=int)
            col1, col2, col3 = tuple(np.transpose(self[self.ID_COLS].values))
            self.id_map[col1, col2, col3] = self.index
            self.id_map[col3, col2, col1] = self.index

        return self.loc[self.id_map[tuple(np.transpose(atom_ids))]]

    def getIndex(self, func):
        """
        Get the index of the angle with the lowest energy.

        :param key func: a function to get the angle energy from the type.
        :return int: the index of the angle with the lowest energy.
        """
        return min(self.index, key=lambda x: func(self.loc[x].type_id))


class Dihedral(Bond):
    """
    The dihedral angle information including the dihedral type and the atom ids.
    """

    NAME = 'Dihedrals'
    ID_COLS = [ATOM1, ATOM2, ATOM3, ATOM4]
    COLUMNS = [TYPE_ID] + ID_COLS
    LABEL = 'dihedrals'

    def getPairs(self, step=3):
        """
        Get the atom pairs from each topology connectivity.

        :param step: the step when slicing the atom ids
        :type step: int
        :return: the atom pairs
        :rtype: list of tuple
        """
        return super(Dihedral, self).getPairs(step=step)


class Improper(Dihedral):
    """
    The improper angle information including the improper type and the atom ids.
    """

    NAME = 'Impropers'
    LABEL = 'impropers'

    def getPairs(self):
        """
        Get the atom pairs from each topology connectivity.

        :param step: the step when slicing the atom ids
        :type step: int
        :return: the atom pairs
        :rtype: list of tuple
        """
        ids = [itertools.combinations(x, 2) for x in self[self.ID_COLS].values]
        return [tuple(sorted(y)) for x in ids for y in x]

    def getAngles(self):
        """
        Get the atom pairs from each topology connectivity.

        :return: each row contains three angles by one improper angle atoms.
        :rtype: ndarray
        """
        columns = [ATOM2, ATOM1, ATOM4]
        cols = [[x, ATOM3, y] for x, y in itertools.combinations(columns, 2)]
        return np.array([x for x in zip(*[self[x].values for x in cols])])


class Conformer(lmpatomic.Conformer):
    """
    The customized conformer class with additional methods for atom information,
    topology information, measurement, and internal coordinate manipulations.
    """

    @property
    def atoms(self):
        """
        Atoms in the conformer.

        :return 'numpy.ndarray': information such as global ids, molecule
            ids, atom type ids, charges, coordinates.
        """
        atoms = super().atoms
        atoms[:, 1] = self.gid
        return atoms

    @property
    def bonds(self):
        """
        Bonds in the conformer.

        :return `Bond`: information such as bond ids and bonded atom ids.
        """
        return self.GetOwningMol().bonds.to_numpy(id_map=self.id_map)

    @property
    def angles(self):
        """
        Angles in the conformer.

        :return `Angle`: information such as angle ids and connected atom ids.
        """
        return self.GetOwningMol().angles.to_numpy(id_map=self.id_map)

    @property
    def dihedrals(self):
        """
        Dihedral angles in the conformer.

        :return `Dihedral`: information such as dihedral ids and connected atom ids.
        """
        return self.GetOwningMol().dihedrals.to_numpy(id_map=self.id_map)

    @property
    def impropers(self):
        """
        Improper angles in the conformer.

        :return `Improper`: information such as improper ids and connected atom ids.
        """
        return self.GetOwningMol().impropers.to_numpy(id_map=self.id_map)

    def setBondLength(self, bonded, val):
        """
        Set bond length of the given dihedral.

        :param bonded tuple of int: the bonded atom indices.
        :param val: the bond distance.
        """
        Chem.rdMolTransforms.SetBondLength(self, *bonded, val)

    def setAngleDeg(self, aids, val):
        """
        Set bond length of the given dihedral.

        :param aids tuple of int: the atom indices in one angle.
        :param val: the angle degree.
        """
        Chem.rdMolTransforms.SetAngleDeg(self, *aids, val)

    def setDihedralDeg(self, dihe, val):
        """
        Set angle degree of the given dihedral.

        :param dihe tuple of int: the dihedral atom indices.
        :param val float: the angle degree.
        """
        Chem.rdMolTransforms.SetDihedralDeg(self, *dihe, val)

    def measure(self, *aids):
        """
        Measure the bond length, angle degree, or dihedral angle.

        :param aids list of int: the atoms defining the bond or angle.
        :return float or str: the measurement.
        """
        if not aids:
            aids = self.GetOwningMol().getSubstructMatch()
        if aids is None:
            return

        num = len(aids)
        match num:
            case 2:
                value = Chem.rdMolTransforms.GetBondLength(self, *aids)
            case 3:
                value = Chem.rdMolTransforms.GetAngleDeg(self, *aids)
            case 4:
                value = Chem.rdMolTransforms.GetDihedralDeg(self, *aids)
        return Float(value, num=num)


class Float(float):
    """
    A float class providing unit and name in string representation.
    """

    FMT = '{name}: {value:.2f} {unit}'
    NAME = {2: 'distance', 3: 'angle', 4: 'dihedral'}
    UNIT = {2: 'angstrom', 3: 'degree', 4: 'degree'}

    def __new__(self, value, num=None):
        """
        :param value float: the measured float value.
        :param num int: the number of atoms involved in the measurement.
        """
        return float.__new__(self, value)

    def __init__(self, value, num=None):
        """
        :param value float: the measured float value.
        :param num int: the number of atoms involved in the measurement.
        """
        float.__init__(value)
        self.num = num

    def __str__(self):
        """
        :return str: the string representation with name and unit.
        """
        if self.num is None:
            return super().__str__()
        return self.FMT.format(name=self.NAME[self.num],
                               value=self,
                               unit=self.UNIT[self.num])


class Mol(lmpatomic.Mol):
    """
    In addition to the parent, additional methods are added for internal
    coordinate manipulations, force field parsing, and substructure matching.
    """

    ConfClass = Conformer
    RES_NUM = symbols.RES_NUM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charges = None
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None
        if self.delay:
            return
        self.setCharges()
        self.setBonds()
        self.setDihedrals()
        self.setImpropers()
        self.setAngles()
        self.setInternal()
        self.setSubstructure()
        self.updateAll()

    def type(self):
        """
        Type atoms and set charges.
        """
        self.ff.type(self)

    def setInternal(self):
        """
        Set the internal coordinates by adjusting bonds and angles.
        """
        # Set the bond lengths of one conformer
        tpl = self.GetConformer()
        for type_id, *ids in self.bonds.values:
            tpl.setBondLength(list(map(int, ids)),
                              self.ff.bonds.loc[type_id].dist)
        # Set the angle degree of one conformer
        for type_id, *ids in self.angles.values:
            tpl.setAngleDeg(list(map(int, ids)),
                            self.ff.angles.loc[type_id].deg)

    def setSubstructure(self):
        """
        Set substructure.
        """
        aids = self.getSubstructMatch()
        if aids is None or self.struct.options.substruct[1] is None:
            return
        template = self.GetConformer()
        match len(aids):
            case 2:
                template.setBondLength(aids, self.struct.options.substruct[1])
            case 3:
                template.setAngleDeg(aids, self.struct.options.substruct[1])
            case 4:
                template.setDihedralDeg(aids, self.struct.options.substruct[1])

    def getSubstructMatch(self, gid=False):
        """
        Get substructure match.

        :param gid bool: whether to return global atom ids.
        :return Series: the atom ids of the substructure match.
        """
        if self.struct.options.substruct is None:
            return
        struct = Chem.MolFromSmiles(self.struct.options.substruct[0])
        if not self.HasSubstructMatch(struct):
            return
        ids = self.GetSubstructMatch(struct)
        if gid:
            ids = self.GetConformer().id_map[list(ids)]
        match len(ids):
            case 2:
                return pd.Series(ids, name='bond')
            case 3:
                return pd.Series(ids, name='angle')
            case 4:
                return pd.Series(ids, name='dihedral')

    def updateAll(self):
        """
        Update all conformers.
        """
        xyz = self.GetConformer().GetPositions()
        for conf in self.GetConformers():
            conf.setPositions(xyz)

    def setAtoms(self):
        """
        The atoms of the molecules.

        :return Atom: Atoms with type ids and charges
        """
        super().setAtoms()
        self.atoms.insert(1, Atom.MOL_ID, 0)
        self.atoms = Atom(self.atoms)

    def setCharges(self):
        """
        The charges of the molecules.

        :return list of float: the atomic charges.
        """
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        fchrg = [self.ff.charges.loc[x] for x in type_ids]
        nchrg = [self.nbr_charge[x.GetIdx()] for x in self.GetAtoms()]
        self.charges = [sum(x) for x in zip(fchrg, nchrg)]

    def setBonds(self):
        """
        The bonds of the molecule.
        """
        atoms = [[x.GetBeginAtom(), x.GetEndAtom()] for x in self.GetBonds()]
        type_ids = [self.ff.bonds.getMatched(x) for x in atoms]
        aids = [[y.GetIdx() for y in x] for x in atoms]
        self.bonds = Bond(type_ids=type_ids, aids=aids)

    def setAngles(self):
        """
        Angle force of the molecules after removal due to improper angles.

        e.g. NH3 if all three H-N-H angles are defined, you cannot control out
        of plane mode.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multi-pole and Gaussian
        Multi-xpole Models

        :return Angle: the angle types and atoms forming each angle.
        """
        angles = [y for x in self.GetAtoms() for y in self.getAngleAtoms(x)]
        type_ids = [self.ff.angles.getMatched(x) for x in angles]
        aids = [[y.GetIdx() for y in x] for x in angles]
        angles = Angle(type_ids=type_ids, aids=aids)
        matches = [angles.select(x) for x in self.impropers.getAngles()]
        index = [
            x.getIndex(lambda x: self.ff.angles.loc[x].ene) for x in matches
        ]
        self.angles = angles.drop(index=index)

    def getAngleAtoms(self, atom, unique=True):
        """
        Get all three angle atoms from the input middle atom. The first atom has
        a TYPE_ID smaller than the third.

        :param atom 'rdkit.Chem.rdchem.Atom': the middle atom
        :return list of list: each sublist contains three atoms.
        """
        nbrs = atom.GetNeighbors()
        if len(nbrs) < 2:
            return []
        angles = [[x, atom, y] for x, y in itertools.combinations(nbrs, 2)]
        return angles if unique else angles + [x[::-1] for x in angles]

    def setDihedrals(self):
        """
        Dihedral angles of the molecules.

        :return Dihedral: the dihedral types and atoms forming each dihedral.
        """
        dihes = [x for x in self.getDihAtoms()]
        type_ids = [self.ff.dihedrals.getMatched(x) for x in dihes]
        aids = [[y.GetIdx() for y in x] for x in dihes]
        self.dihedrals = Dihedral(type_ids=type_ids, aids=aids)

    def setImpropers(self):
        """
        Improper angles of the molecules.

        :return Improper: the improper types and atoms forming each improper.
        """
        imprps = [x for x in self.getImproperAtoms()]
        type_ids = [self.ff.impropers.getMatched(x) for x in imprps]
        aids = [[y.GetIdx() for y in x] for x in imprps]
        self.impropers = Improper(type_ids=type_ids, aids=aids)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight

    @property
    @functools.cache
    def nbr_charge(self):
        """
        Balance the charge when residues are not neutral.

        :return dict: the atom id and its charge due to connected neighbors.
        """
        # residual num: residual charge
        res_charge = collections.defaultdict(float)
        for atom in self.GetAtoms():
            res_num = atom.GetIntProp(self.RES_NUM)
            type_id = atom.GetIntProp(TYPE_ID)
            res_charge[res_num] += self.ff.charges.loc[type_id].q

        res_snacharge = {x: 0 for x, y in res_charge.items() if y}
        res_atom = {}
        for bond in self.GetBonds():
            batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
            bres_num = batom.GetIntProp(self.RES_NUM)
            eres_num = eatom.GetIntProp(self.RES_NUM)
            if bres_num == eres_num:
                continue
            # Bonded atoms in different residuals
            for atom, natom in [[batom, eatom], [eatom, batom]]:
                nres_num = natom.GetIntProp(self.RES_NUM)
                ncharge = res_charge[nres_num]
                if not ncharge:
                    continue
                # The natom lives in nres with total charge
                snatom_charge = abs(
                    self.ff.charges.loc[natom.GetIntProp(TYPE_ID)].q)
                if snatom_charge > res_snacharge[nres_num]:
                    res_atom[nres_num] = atom.GetIdx()
                    res_snacharge[nres_num] = snatom_charge

        nbr_charge = collections.defaultdict(float)
        for res, idx in res_atom.items():
            nbr_charge[idx] -= res_charge[res]
        return nbr_charge

    def getRigid(self):
        """
        The bond and angle are rigid during simulation.

        :return DataFrame, DataFrame: the type ids of the rigid bonds and angles
        """
        bnd_types = self.bonds.getRigid(self.ff.bonds.has_h)
        ang_types = self.angles.getRigid(self.ff.angles.has_h)
        return bnd_types, ang_types

    def getDihAtoms(self):
        """
        Get the dihedral atoms of this molecule.

        NOTE: Flipping the order the four dihedral atoms yields the same dihedral,
        and only one of them is returned.

        :return list of list: each sublist has four atom ids forming a dihedral angle.
        """
        atomss = [y for x in self.GetAtoms() for y in self.getDihedralAtoms(x)]
        # 1-2-3-4 and 4-3-2-1 are the same dihedral
        atomss_no_flip = []
        atom_idss = set()
        for atoms in atomss:
            atom_ids = tuple(x.GetIdx() for x in atoms)
            if atom_ids in atom_idss:
                continue
            atom_idss.add(atom_ids)
            atom_idss.add(atom_ids[::-1])
            atomss_no_flip.append(atoms)
        return atomss_no_flip

    def getDihedralAtoms(self, atom):
        """
        Get the dihedral atoms whose torsion bonded atoms contain this atom.

        :param atom 'rdkit.Chem.rdchem.Atom': the middle atom of the dihedral
        :return generator: four atoms forming a dihedral angle at a time
        """
        atomss = self.getAngleAtoms(atom, unique=False)
        for satom, matom, eatom in atomss:
            idx = matom.GetIdx()
            dihe_4ths = [y for x in self.getAngleAtoms(eatom) for y in x[::2]]
            for dihe_4th in dihe_4ths:
                if dihe_4th.GetIdx() == idx:
                    continue
                yield satom, matom, eatom, dihe_4th

    def getImproperAtoms(self):
        """
        Set improper angles based on center atoms and neighbor symbols.

        :param csymbols str: each Char is one possible center element

        In short:
        1) sp2 sites and united atom CH groups (sp3 carbons) needs improper
         (though I saw a reference using improper for sp3 N)
        2) No rules for a center atom. (Charmm asks order for symmetricity)
        3) Number of internal geometry variables (3N_atom – 6) deletes one angle

        The details are the following:

        When the Weiner et al. (1984,1986) force field was developed, improper
        torsions were designated for specific sp2 sites, as well as for united
        atom CH groups - sp3 carbons with one implicit hydrogen.
        Ref: http://ambermd.org/Questions/improp.html

        There are no rules for a center atom. You simply define two planes, each
        defined by three atoms. The angle is given by the angle between these
        two planes. (from hess)
        ref: https://gromacs.bioexcel.eu/t/the-atom-order-i-j-k-l-in-defining-an
        -improper-dihedral-in-gromacs-using-the-opls-aa-force-field/3658

        The CHARMM convention in the definition of improper torsion angles is to
        list the central atom in the first position, while no rule exists for how
        to order the other three atoms.
        ref: Symmetrization of the AMBER and CHARMM Force Fields, J. Comput. Chem.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models
        """
        # FIXME: LAMMPS recommends the first to be the center, while the prm
        # and literature order the third as the center.
        # My Implementation:
        # Use the center as the third according to "A New Force Field for
        # Molecular Mechanical Simulation of Nucleic Acids and Proteins"
        # No special treatment to the order of other atoms.

        # My Reasoning: first or third functions the same for planar
        # scenario as both 0 deg and 180 deg implies in plane. However,
        # center as first or third defines different planes, leading to
        # eiter ~45 deg or 120 deg as the equilibrium improper angle.
        # 120 deg sounds more plausible and thus the third is chosen to be
        # the center.

        atoms = []
        for atom in self.GetAtoms():
            if atom.GetTotalDegree() != 3:
                continue
            match atom.GetSymbol():
                case symbols.CARBON:
                    # Planar Sp2 carbonyl carbon (R-COOH)
                    # tetrahedral Sp3 carbon with one implicit H (CHR1R2R3)
                    atoms.append(atom)
                case symbols.NITROGEN:
                    if atom.GetHybridization(
                    ) == Chem.rdchem.HybridizationType.SP2:
                        # Sp2 N in Amino Acid or Dimethylformamide
                        atoms.append(atom)
        neighbors = [x.GetNeighbors() for x in atoms]
        return [[y[0], y[1], x, y[2]] for x, y in zip(atoms, neighbors)]

    def getBonds(self):
        """
        Bonds in the conformer.

        :return `Bond`: information such as bond ids and bonded atom ids.
        """
        if self.bonds.empty:
            return
        return np.concatenate([x.bonds for x in self.confs])

    def getAngles(self):
        """
        Angles in the conformer.

        :return `Angle`: information such as angle ids and connected atom ids.
        """
        if self.angles.empty:
            return
        return np.concatenate([x.angles for x in self.confs])

    def getDihedrals(self):
        """
        Dihedral angles in the conformer.

        :return `Dihedral`: information such as dihedral ids and connected atom ids.
        """
        if self.dihedrals.empty:
            return
        return np.concatenate([x.dihedrals for x in self.confs])

    def getImpropers(self):
        """
        Improper angles in the conformer.

        :return `Improper`: information such as improper ids and connected atom ids.
        """
        if self.impropers.empty:
            return
        return np.concatenate([x.impropers for x in self.confs])


class In(lammpsin.In):
    """
    Class to write out LAMMPS in script.
    """

    BOND_STYLE = 'bond_style'
    ANGLE_STYLE = 'angle_style'
    DIHEDRAL_STYLE = 'dihedral_style'
    IMPROPER_STYLE = 'improper_style'
    SPECIAL_BONDS = 'special_bonds'
    LJ_COUL = 'lj/coul'
    KSPACE_STYLE = 'kspace_style'

    PAIR_MODIFY = 'pair_modify'
    GEOMETRIC = 'geometric'
    ARITHMETIC = 'arithmetic'
    SIXTHPOWER = 'sixthpower'

    MIX = 'mix'
    PPPM = 'pppm'
    OPLS = 'opls'
    CVFF = 'cvff'

    HARMONIC = 'harmonic'

    V_UNITS = lammpsin.In.REAL
    V_ATOM_STYLE = lammpsin.In.FULL
    V_BOND_STYLE = HARMONIC
    V_ANGLE_STYLE = HARMONIC
    V_DIHEDRAL_STYLE = OPLS
    V_IMPROPER_STYLE = CVFF

    def setup(self):
        """
        Write the setup section including unit, topology styles, and specials.
        """
        super().setup()
        self.fh.write(f"{self.BOND_STYLE} {self.V_BOND_STYLE}\n")
        self.fh.write(f"{self.ANGLE_STYLE} {self.V_ANGLE_STYLE}\n")
        self.fh.write(f"{self.DIHEDRAL_STYLE} {self.V_DIHEDRAL_STYLE}\n")
        self.fh.write(f"{self.IMPROPER_STYLE} {self.V_IMPROPER_STYLE}\n")
        self.fh.write(f"{self.SPECIAL_BONDS} {self.LJ_COUL} 0 0 0.5\n")

    def pair(self):
        """
        Write pair style, coefficients, and mixing rules as well as k-space.
        """
        pair_style, cuts = self.LJ_CUT, self.DEFAULT_LJ_CUT
        if self.hasCharge():
            pair_style = self.LJ_CUT_COUL_LONG
            cuts = self.DEFAULT_COUL_CUT
        self.fh.write(f"{self.PAIR_STYLE} {pair_style} {cuts}\n")
        self.fh.write(f"{self.PAIR_MODIFY} {self.MIX} {self.GEOMETRIC}\n")
        if self.hasCharge():
            self.fh.write(f"{self.KSPACE_STYLE} {self.PPPM} 0.0001\n")

    def hasCharge(self):
        """
        Whether any atom has non-zero charge. This method should be overwritten
        when force field and structure are available.

        :return bool: True if any atom has non-zero charge.
        """

        return True


class Struct(lmpatomic.Struct, In):
    """
    The structure class with interface to LAMMPS data file and in script.
    """

    Atom = Atom

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, options=options, **kwargs)
        In.__init__(self, options=options)
        self.atm_types = None
        self.bnd_types = None
        self.ang_types = None
        self.dihe_types = None
        self.impr_types = None
        self.initTypeMap()

    def initTypeMap(self):
        """
        Initiate type map.
        """
        self.atm_types = numpyutils.IntArray(self.ff.atoms.index.size)
        self.bnd_types = numpyutils.IntArray(self.ff.bonds.index.size)
        self.ang_types = numpyutils.IntArray(self.ff.angles.index.size)
        self.dihe_types = numpyutils.IntArray(self.ff.dihedrals.index.size)
        self.impr_types = numpyutils.IntArray(self.ff.impropers.index.size)

    def setTypeMap(self, mol):
        """
        Set the type map for atoms, bonds, angles, dihedrals, and impropers.

        :param mol: add this molecule to the structure
        :type mol: Mol
        """
        atypes = [x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]
        self.atm_types[atypes] = True
        self.bnd_types[mol.bonds[TYPE_ID]] = True
        self.ang_types[mol.angles[TYPE_ID]] = True
        self.dihe_types[mol.dihedrals[TYPE_ID]] = True
        self.impr_types[mol.impropers[TYPE_ID]] = True

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
            self.atom_blk.write(self.hdl)
            self.bonds.write(self.hdl)
            self.angles.write(self.hdl)
            self.dihedrals.write(self.hdl)
            self.impropers.write(self.hdl)

    @property
    @functools.cache
    def atom_blk(self):
        """
        The total atomic information of all data types.

        :return `Atom`: information such as global ids, type ids, charges, and
            coordinates.
        """
        return AtomBlock(
            self.atoms.astype(np.float32).join(self.charges).join(self.xyz))

    @property
    @functools.cache
    def charges(self):
        """
        Atoms charges.

        :return `Charge`: the charges of all atoms.
        """
        charges = [x.GetOwningMol().charges for x in self.conformer]
        return Charge(np.concatenate(charges).reshape(-1, 1))

    @property
    @functools.cache
    def bonds(self):
        """
        Bonds in the structure.

        :return 'np.ndarray': bond types and bonded atom ids.
        """
        bonds = [x.getBonds() for x in self.mols]
        return Bond.concatenate(bonds, type_map=self.bnd_types)

    @property
    @functools.cache
    def angles(self):
        """
        Angle in the structure.

        :return 'np.ndarray': angle types and connected atom ids.
        """
        angles = [x.getAngles() for x in self.mols]
        return Angle.concatenate(angles, type_map=self.ang_types)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Dihedral angles in the structure.

        :return 'np.ndarray': dihedral types and connected atom ids.
        """
        dihes = [x.getDihedrals() for x in self.mols]
        return Dihedral.concatenate(dihes, type_map=self.dihe_types)

    @property
    @functools.cache
    def impropers(self):
        """
        Improper angles in the structure.

        :return 'np.ndarray': improper types and connected atom ids.
        """
        imprps = [x.getImpropers() for x in self.mols]
        return Improper.concatenate(imprps, type_map=self.impr_types)

    @property
    def masses(self):
        """
        Atom masses.

        :return `Mass`: mass of each type of atom.
        """
        return Mass.fromAtoms(self.ff.atoms.loc[self.atm_types.on])

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
        imprps = [[x.ene, 1 if x.deg == 0. else -1, x.n_parm]
                  for x in imprps.itertuples()]
        return ImproperCoeff(imprps)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return sum([x.mw * len(x.confs) for x in self.mols])

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        return not np.isclose(self.charges, 0, 0.001).any()

    @property
    @functools.cache
    def rest(self):
        """
        See the parent class for docstring.
        """
        if self.options.substruct is None or self.options.substruct[1] is None:
            return
        gids = self.mols[0].getSubstructMatch(gid=True)
        if gids is None:
            return None
        geo = f"{gids.name} {' '.join(map(str, gids + 1))}"
        return self.FIX_RESTRAIN.format(geo=geo, val=self.options.substruct[1])

    def shake(self):
        """
        Write fix shake command to enforce constant bond length and angel values.
        """
        if self.options.rigid_bond is None and self.options.rigid_angle is None:
            data = [x.getRigid() for x in self.mols]
            bonds, angles = list(map(list, zip(*data)))
            bonds = Bond.concat([x for x in bonds if not x.empty])
            angles = Angle.concat([x for x in angles if not x.empty])
            bond_types = self.bnd_types.index(bonds.values.flatten()) + 1
            angle_types = self.ang_types.index(angles.values.flatten()) + 1
            self.options.rigid_bond = ' '.join(map(str, bond_types))
            self.options.rigid_angle = ' '.join(map(str, angle_types))
        super().shake()

    def getWarnings(self):
        """
        Get warnings for the structure.

        :return generator of str: the warnings on structure checking.
        """
        net_charge = round(self.charges.sum().sum(), 4)
        if net_charge:
            yield f'The system has a net charge of {net_charge:.4f}'
        min_span = self.box.span.min()
        if min_span < self.DEFAULT_CUT * 2:
            yield f'The minimum box span ({min_span:.2f} {symbols.ANGSTROM})' \
                  f' is smaller than {self.DEFAULT_CUT * 2:.2f} ' \
                  f'{symbols.ANGSTROM} (Lennard-Jones Cutoff x 2) '


class Reader(lmpatomic.Reader):
    """
    See the parent class for docstring.
    """
    Atom = Atom
    Mass = Mass
    AtomBlock = AtomBlock
    BLOCK_CLASSES = [
        lmpatomic.Mass, lmpatomic.PairCoeff, BondCoeff, AngleCoeff,
        DihedralCoeff, ImproperCoeff, AtomBlock, Bond, Angle, Dihedral,
        Improper
    ]

    def __init__(self, data_file=None, contents=None, delay=False):
        """
        :param data_file str: data file with path
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
        self.contents = contents
        self.lines = None
        self.name = {}
        if delay:
            return
        self.read()
        self.index()

    @property
    @functools.cache
    def charges(self):
        """
        Parse the atom section.

        :return `Charge`: the atomic charges.
        """
        return self.atom_blk[Charge.COLUMNS]

    @property
    @functools.cache
    def pair_coeffs(self):
        """
        Paser the pair coefficient section.

        :return `PairCoeff`: the pair coefficients between non-bonded atoms.
        """
        return self.fromLines(lmpatomic.PairCoeff)

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

        :return `Bond`: the bond information such as id, type id, and bonded
            atom ids.
        """
        return self.fromLines(Bond)

    @property
    @functools.cache
    def angles(self):
        """
        Parse the angle section for angle id and constructing atoms.

        :return `Angle`: the angle information such as id, type id, and atom ids
            in the angle.
        """
        return self.fromLines(Angle)

    @property
    @functools.cache
    def dihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.

        :return `Dihedral`: the dihedral angle information such as id, type id,
            and atom ids in the dihedral angle.
        """
        return self.fromLines(Dihedral)

    @property
    @functools.cache
    def impropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.

        :return `Improper`: the improper angle information such as id, type id,
            and atom ids in the improper angle.
        """
        return self.fromLines(Improper)

    @property
    @functools.cache
    def mols(self):
        """
        The atom ids grouped by molecules.

        :return: keys are molecule ids and values are atom global ids.
        :rtype: dict
        """
        mols = collections.defaultdict(list)
        for gid, mid, in self.atoms.mol_id.items():
            mols[mid].append(gid)
        return dict(mols)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.masses.mass[self.atoms.type_id].sum()

    mw = molecular_weight

    def allClose(self, other, atol=1e-08, rtol=1e-05, equal_nan=True):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other: the other data reader to compare against.
        :type other: float
        :param atol: The relative tolerance parameter (see Notes).
        :type atol: float
        :param rtol: The absolute tolerance parameter (see Notes).
        :type rtol: float
        :param equal_nan: If True, NaNs are considered close.
        :type equal_nan: bool
        :return: whether two data are close.
        :rtype: bool
        """
        kwargs = dict(atol=atol, rtol=rtol, equal_nan=equal_nan)
        if not super().allClose(other, **kwargs):
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
