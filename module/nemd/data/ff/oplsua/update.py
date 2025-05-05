# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This script generates oplsua database.
"""
import functools
import io
import os

import chemparse

from nemd import builtinsutils
from nemd import np
from nemd import oplsua
from nemd import pd
from nemd import rdkitutils
from nemd import symbols

IDX = oplsua.IDX

SMILES = [
# Single Atom Particle
['[Li+]', [197], None, 'Li+ Lithium Ion'],
['[Na+]', [198], None, 'Na+ Sodium Ion'],
['[K+]', [199], None, 'K+ Potassium Ion'],
['[Rb+]', [200], None, 'Rb+ Rubidium Ion'],
['[Cs+]', [201], None, 'Cs+ Cesium Ion'],
['[Mg+2]', [202], None, 'Mg+2 Magnesium Ion'],
['[Ca+2]', [203], None, 'Ca+2 Calcium Ion'],
['[Sr+2]', [204], None, 'Sr+2 Strontium Ion'],
['[Ba+2]', [205], None, 'Ba+2 Barium Ion'],
['[F-]', [206], None, 'F- Fluoride Ion'],
['[Cl-]', [207], None, 'Cl- Chloride Ion'],
['[Br-]', [208], None, 'Br- Bromide Ion'],
['[He]', [209], None, 'Helium Atom'],
['[Ne]', [210], None, 'Neon Atom'],
['[Ar]', [211], None, 'Argon Atom'],
['[Kr]', [212], None, 'Krypton Atom'],
['[Xe]', [213], None, 'Xenon Atom'],
# Alkane
['C', [81], None, 'CH4 Methane'],
['CC', [82, 82], None, 'Ethane'],
['CCC', [83, 86, 83], None, 'Propane'],
['CCCC', [83, 86, 86, 83], None, 'n-Butane'],
['CC(C)C', [84, 88, 84, 84], None, 'Isobutane'],
['CC(C)(C)C', [85, 90, 85, 85, 85], None, 'Neopentane'],
# Alkene
['CC=CC', [84, 89, 89, 84], None, '2-Butene'],
# Aldehydes (with formyl group)
# Ketone
['CC(=O)C', [129, 127, 128, 129], None, 'Acetone'],
['CCC(=O)CC', [7, 130, 127, 128, 130, 7], None, 'Diethyl Ketone'],
# https://docs.lammps.org/Howto_tip3p.html
['O', [77], {77: 78}, symbols.WATER_TIP3P],
['O', [79], {79: 80}, symbols.WATER_SPC],
['O', [214], {214: 215}, symbols.WATER_SPCE],
# t-Butyl Ketone CC(C)CC(=O)C(C)(C)C described by Neopentane, Acetone, and Diethyl Ketone
# Alcohol
['CO', [106, 104], {104: 105}, 'Methanol'],
['CCO', [83, 107, 104], {104: 105}, 'Ethanol'],
['CC(C)O', [84, 108, 84, 104], {104: 105}, 'Isopropanol'],
# Carboxylic Acids
# "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
['O=CO', [134, 133, 135], {135: 136}, 'Carboxylic Acid'],
# "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
['CC(=O)O', [137, 133, 134, 135], {135: 136}, 'Ethanoic acid'],
# Large Molecules
['CN(C)C=O', [156, 148, 156, 153, 151], None, 'N,N-Dimethylformamide']
] # yapf: disable


class Base(oplsua.Base):
    """
    Base class of oplsua database.
    """

    def to_parquet(self, *args, index=False, **kwargs):
        """
        Save the pandas data to the parquet file.

        :param index: include the dataframe’s index(es) if True
        """
        super().to_parquet(self.parquet, *args, index=index, **kwargs)


class Smiles(Base):

    def __init__(self):
        super().__init__(SMILES[::-1], columns=['sml', 'mp', 'hs', 'dsc'])
        mol = [rdkitutils.MolFromSmiles(x) for x in self.sml]
        self['deg'] = [list(map(self.getDeg, x.GetAtoms())) for x in mol]
        hs = self.hs.dropna()
        self.loc[hs.index, 'hs'] = hs
        self.hs = self.hs.astype(str)

    @staticmethod
    def getDeg(atom):
        """
        Get the degree of the atom. (the hydrogen atoms on carbons are not
        counted towards in the degree in the united atom model)

        :param atom `rdkit.Chem.rdchem.Atom`: the atom to get degree of
        :return list: the degree of the atom
        """
        degree = atom.GetDegree()
        if atom.GetSymbol() != symbols.CARBON:
            degree += atom.GetNumImplicitHs()
        return degree


class Charge(Base):
    """
    The class to hold charge information.
    """
    COLS = [IDX, 'q']
    MARKER = 'Atomic Partial Charge Parameters'

    def __init__(self, lines):
        """
        :param lines list: the lines from force field file.
        """
        text = io.StringIO('\n'.join(self.getLines(lines)))
        data = pd.read_csv(text, sep=r'\s+', names=[self.name] + self.COLS)
        super().__init__(data)
        self.drop([self.name], axis=1, inplace=True)
        if IDX in self.columns:
            self.set_index(IDX, inplace=True)

    @classmethod
    def getLines(cls, lines):
        """
        Get the block lines.

        :param lines list: the lines from force field file.
        :return list: the block lines.
        """
        sidx = lines.index(cls.MARKER)
        lines = lines[sidx + 1:]
        lines = lines[next(x for x, y in enumerate(lines) if y):]
        return lines[:next(x for x, y in enumerate(lines) if not y)]


class Vdw(Charge):
    """
    The class to hold Van der Waals information.
    """
    COLS = [IDX, 'dist', 'ene']
    MARKER = 'Van der Waals Parameters'


class Atom(Vdw):
    """
    The class to hold VDW information.
    """
    COLS = [IDX, 'formula', 'descr', 'Z', 'atomic_weight', 'conn', 'symbol']
    MARKER = 'Atom Type Definitions'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        formula = self.formula.apply(chemparse.parse_formula)
        self.conn += formula.apply(lambda x: int(x.pop(symbols.HYDROGEN, 0)))
        self.symbol = formula.apply(lambda x: next(iter(x), symbols.HYDROGEN))


class Improper(Charge):
    """
    The class to hold improper torsional information.
    """
    ID_COLS = oplsua.Improper.ID_COLS
    COLS = ID_COLS + ['ene', 'deg', 'n_parm']
    MARKER = 'Improper Torsional Parameters'


class Bond(Improper):
    """
    The class to hold bond information.
    """
    ID_COLS = oplsua.Bond.ID_COLS
    COLS = ID_COLS + ['ene', 'dist']
    MARKER = 'Bond Stretching Parameters'

    @property
    def TMAP(self):
        """
        See TMAP.
        """
        # "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
        return {
            134: 2,
            133: 26,
            135: 23,
            136: 24,
            153: 72,
            148: 3,
            108: 107,
            127: 1,
            128: 2,
            129: 7,
            130: 9,
            85: 9,
            90: 64
        }

    @property
    def MAP(self):
        """
        See MAP.
        """
        # C-OH (Tyr) is used as HO-C=O, which needs CH2-COOH map as alpha-COOH bond
        return {(26, 86): (16, 17), (26, 88): (16, 17), (86, 107): (86, 86)}

    def to_npy(self):
        """
        Save the mapping information into a numpy file.
        """
        with open(self.npy, 'wb') as fh:
            np.save(fh, np.array([x for x in self.TMAP.items()]))
            keys = np.array([x for x in self.MAP.keys()])
            values = np.array([x for x in self.MAP.values()])
            np.save(fh, np.stack([keys, values]))


class Angle(Bond):
    """
    The class to hold angle information.
    """
    ID_COLS = oplsua.Angle.ID_COLS
    COLS = ID_COLS + ['ene', 'deg']
    MARKER = 'Angle Bending Parameters'

    @property
    def TMAP(self):
        return {
            134: 2,
            133: 17,
            135: 76,
            136: 24,
            148: 3,
            153: 72,
            108: 107,
            127: 1,
            129: 7,
            130: 9
        }

    @property
    def MAP(self):
        return {
            (84, 107, 84): (86, 88, 86),
            (84, 107, 86): (86, 88, 83),
            (86, 107, 86): (86, 88, 83)
        }


class Dihedral(Bond):
    """
    The class to hold dihedral information.
    """
    ID_COLS = oplsua.Dihedral.ID_COLS
    COLS = ID_COLS + ['k1', 'k2', 'k3', 'k4']
    MARKER = 'Torsional Parameters'

    @property
    def TMAP(self):
        return {
            134: 11,
            133: 26,
            135: 76,
            136: 24,
            148: 3,
            153: 72,
            108: 107,
            127: 1,
            130: 9,
            86: 9,
            88: 9,
            90: 9
        }

    @property
    def MAP(self):
        return {
            (26, 86): (1, 6),
            (26, 88): (1, 6),
            (88, 107): (6, 22),
            (86, 107): (6, 25),
            (9, 26): (9, 1),
            (9, 107): (9, 9)
        }

    def getLines(self, lines):
        """
        See parent.
        """
        return [self.format(x) for x in super().getLines(lines)]

    @staticmethod
    def format(line):
        """
        Format the dihedral coefficients to Lammps format: K1, K2, K3, K4 as in
        0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)] ... from LAMMPS dihedral_style opls
        https://docs.lammps.org/dihedral_opls.html

        In oplsua.prm, dihedral adopts [1 + cos(n*x-gama)] formula.
        When gama = 180, cos(x-gama) = cos(x - 180°) = cos(180° - x) = -cos(x)

        :return list of float: opls coefficients K1, K2, K3, K4
        """
        splitted = line.split()
        contants = splitted[5:]
        params = [0., 0., 0., 0.]
        constants = zip(contants[::3], contants[1::3], contants[2::3])
        for ene, deg, n_parm in constants:
            ene, deg, n_parm = float(ene), float(deg), int(n_parm)
            params[n_parm - 1] = ene * 2
            if not params[n_parm]:
                continue
            if (deg == 180.) ^ (not n_parm % 2):
                params[n_parm] *= -1
        return ' '.join(splitted[:5] + list(map(str, params)))


class Raw(builtinsutils.Object):
    """
    Oplsua raw text reader and database writer.
    """

    def __init__(self):
        self.smiles = Smiles()
        lines = self.getLines()
        self.charges = Charge(lines)
        self.vdws = Vdw(lines)
        self.atoms = Atom(lines)
        self.bonds = Bond(lines)
        self.angles = Angle(lines)
        self.dihedrals = Dihedral(lines)
        self.impropers = Improper(lines)

    @staticmethod
    @functools.cache
    def getLines(to_strip=f'{symbols.POUND}\n '):
        """
        The lines from the force field file.

        :param to_strip: the chars to strip off.
        :return list: the stripped lines from the force field file.
        """
        pathname = os.path.join(oplsua.DIRNAME, f"{oplsua.OPLSUA}.prm")
        with open(pathname, 'r') as fh:
            return [x.strip(to_strip) for x in fh.readlines()]

    def write(self):
        """
        Write to database.
        """
        self.smiles.to_parquet()
        self.charges.to_parquet()
        self.vdws.to_parquet()
        self.atoms.to_parquet()
        self.bonds.to_parquet()
        self.bonds.to_npy()
        self.angles.to_parquet()
        self.angles.to_npy()
        self.dihedrals.to_parquet()
        self.dihedrals.to_npy()
        self.impropers.to_parquet()


if __name__ == "__main__":
    Raw().write()
