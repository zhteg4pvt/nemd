# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This script generates oplsua typing table.
"""
from collections import namedtuple

from nemd import oplsua
from nemd import rdkitutils
from nemd import symbols

FIELDS = ['sml', 'mp', 'hs', 'dsc']
UA = namedtuple('UA', FIELDS)
# https://docs.lammps.org/Howto_tip3p.html
WATERS = [
    UA(sml='O', mp=[77], hs={77: 78}, dsc=symbols.WATER_TIP3P),
    UA(sml='O', mp=[79], hs={79: 80}, dsc=symbols.WATER_SPC),
    UA(sml='O', mp=[214], hs={214: 215}, dsc=symbols.WATER_SPCE)
]
# yapf: disable
SMILES = [
    # Single Atom Particle
    UA(sml='[Li+]', mp=[197], hs=None, dsc='Li+ Lithium Ion'),
    UA(sml='[Na+]', mp=[198], hs=None, dsc='Na+ Sodium Ion'),
    UA(sml='[K+]', mp=[199], hs=None, dsc='K+ Potassium Ion'),
    UA(sml='[Rb+]', mp=[200], hs=None, dsc='Rb+ Rubidium Ion'),
    UA(sml='[Cs+]', mp=[201], hs=None, dsc='Cs+ Cesium Ion'),
    UA(sml='[Mg+2]', mp=[202], hs=None, dsc='Mg+2 Magnesium Ion'),
    UA(sml='[Ca+2]', mp=[203], hs=None, dsc='Ca+2 Calcium Ion'),
    UA(sml='[Sr+2]', mp=[204], hs=None, dsc='Sr+2 Strontium Ion'),
    UA(sml='[Ba+2]', mp=[205], hs=None, dsc='Ba+2 Barium Ion'),
    UA(sml='[F-]', mp=[206], hs=None, dsc='F- Fluoride Ion'),
    UA(sml='[Cl-]', mp=[207], hs=None, dsc='Cl- Chloride Ion'),
    UA(sml='[Br-]', mp=[208], hs=None, dsc='Br- Bromide Ion'),
    UA(sml='[He]', mp=[209], hs=None, dsc='Helium Atom'),
    UA(sml='[Ne]', mp=[210], hs=None, dsc='Neon Atom'),
    UA(sml='[Ar]', mp=[211], hs=None, dsc='Argon Atom'),
    UA(sml='[Kr]', mp=[212], hs=None, dsc='Krypton Atom'),
    UA(sml='[Xe]', mp=[213], hs=None, dsc='Xenon Atom'),
    # Alkane
    UA(sml='C', mp=[81], hs=None, dsc='CH4 Methane'),
    UA(sml='CC', mp=[82, 82], hs=None, dsc='Ethane'),
    UA(sml='CCC', mp=[83, 86, 83], hs=None, dsc='Propane'),
    UA(sml='CCCC', mp=[83, 86, 86, 83], hs=None, dsc='n-Butane'),
    UA(sml='CC(C)C', mp=[84, 88, 84, 84], hs=None, dsc='Isobutane'),
    UA(sml='CC(C)(C)C', mp=[85, 90, 85, 85, 85], hs=None, dsc='Neopentane'),
    # Alkene
    UA(sml='CC=CC', mp=[84, 89, 89, 84], hs=None, dsc='2-Butene'),
    # Aldehydes (with formyl group)
    # Ketone
    UA(sml='CC(=O)C', mp=[129, 127, 128, 129], hs=None, dsc='Acetone'),
    UA(sml='CCC(=O)CC', mp=[7, 130, 127, 128, 130, 7], hs=None, dsc='Diethyl Ketone'),
    *WATERS,
    # t-Butyl Ketone CC(C)CC(=O)C(C)(C)C described by Neopentane, Acetone, and Diethyl Ketone
    # Alcohol
    UA(sml='CO', mp=[106, 104], hs={104: 105}, dsc='Methanol'),
    UA(sml='CCO', mp=[83, 107, 104], hs={104: 105}, dsc='Ethanol'),
    UA(sml='CC(C)O', mp=[84, 108, 84, 104], hs={104: 105}, dsc='Isopropanol'),
    # Carboxylic Acids
    # "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
    UA(sml='O=CO', mp=[134, 133, 135], hs={135: 136}, dsc='Carboxylic Acid'),
    # "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
    UA(sml='CC(=O)O', mp=[137, 133, 134, 135], hs={135: 136}, dsc='Ethanoic acid'),
    # Large Molecules
    UA(sml='CN(C)C=O', mp=[156, 148, 156, 153, 151], hs=None, dsc='N,N-Dimethylformamide')
]

def shift(tmap):
    return {x-1: y-1 for x, y in tmap.items()}

def subtract(cmap):
    return {tuple([x - 1 for x in x]): [x - 1 for x in y] for x, y in cmap.items()}


class Smiles(oplsua.Smiles):

    @classmethod
    def read(cls):
        smiles = cls([x._asdict().values() for x in SMILES], columns=FIELDS)
        mol = [rdkitutils.MolFromSmiles(x) for x in smiles.sml]
        smiles['deg'] = [
            list(map(cls.getDeg, x.GetAtoms())) for x in mol
        ]
        smiles['mp'] = smiles['mp'].map(lambda x: [x - 1 for x in x])
        smiles['hs'] = smiles['hs'].map(lambda x: x if x is None else {
            x - 1: y - 1
            for x, y in x.items()
        })
        smiles['hs'] = smiles['hs'].astype(str)
        return smiles.iloc[::-1]

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


class Bond(oplsua.Bond):

    @property
    def TMAP(self):
        # "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
        return shift({
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
        })

    @property
    def MAP(self):
        # C-OH (Tyr) is used as HO-C=O, which needs CH2-COOH map as alpha-COOH bond
        return subtract({(26, 86): (16, 17), (26, 88): (16, 17), (86, 107): (86, 86)})


class Angle(oplsua.Angle):
    @property
    def TMAP(self):
        return shift({
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
        })

    @property
    def MAP(self):
        return subtract({
            (84, 107, 84): (86, 88, 86),
            (84, 107, 86): (86, 88, 83),
            (86, 107, 86): (86, 88, 83)
        })


class Dihedral(oplsua.Dihedral):

    @property
    def TMAP(self):
        return shift({
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
        })

    @property
    def MAP(self):
        return subtract({
            (26, 86): (1, 6),
            (26, 88): (1, 6),
            (88, 107): (6, 22),
            (86, 107): (6, 25),
            (9, 26): (9, 1),
            (9, 107): (9, 9)
        })


class Writer:

    def run(self):
        self.typer()
        self.atoms()
        self.vdws()
        self.charges()
        self.bonds()
        self.angles()
        self.dihedrals()
        self.impropers()

    def typer(self):
        smiles = Smiles.read()
        smiles.to_parquet()

    def atoms(self):
        #atom          1    C     "C Peptide Amide"              6    12.011    3
        atoms = oplsua.Atom.read()
        atoms.to_parquet()

    def vdws(self):
        # 'vdw         213               2.5560     0.4330'
        vdws = oplsua.Vdw.read()
        vdws.to_parquet()

    def charges(self):
        # 'charge      213               0.0000'
        charges = oplsua.Charge.read()
        charges.to_parquet()

    def bonds(self):
        # 'bond        104  107          386.00     1.4250'
        bonds =Bond.read()
        bonds.to_parquet()
        bonds.save()

    def angles(self):
        # 'angle        83  107  104      80.00     109.50'
        anlges = Angle.read()
        anlges.to_parquet()
        anlges.save()

    def dihedrals(self):
        # torsion       2    1    3    4            0.650    0.0  1      2.500  180.0  2
        dihedrals = Dihedral.read()
        dihedrals.to_parquet()
        dihedrals.save()

    def impropers(self):
        # imptors       5    3    1    2           10.500  180.0  2
        impropers =oplsua.Improper.read()
        impropers.to_parquet()


if __name__ == "__main__":
    Writer().run()