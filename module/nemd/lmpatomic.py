# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses a LAMMPS data file in the atomic atom_style.

Atoms section: atom-ID atom-type x y z
"""
import functools
import re

import methodtools
import numpy as np
import pandas as pd

from nemd import envutils
from nemd import jobutils
from nemd import lmpin
from nemd import numpyutils
from nemd import pbc
from nemd import structure
from nemd import sw
from nemd import symbols
from nemd import table

TYPE_ID = symbols.TYPE_ID
ATOM1 = 'atom1'


class Base(pbc.Base):
    """
    The base class for the coefficient and topology.
    """

    def writeCount(self, fh):
        """
        Write the count with the label appended.

        :param hdl `_io.TextIOWrapper`: write to this handler
        """
        fh.write(f'{self.shape[0]} {self.LABEL}\n')

    @classmethod
    def fromLines(cls, *args, index_col=0, **kwargs):
        """
        See parent.
        """
        return super().fromLines(*args, index_col=index_col, **kwargs)


class Mass(Base):
    """
    The masses class for each type of atoms.
    """
    NAME = 'Masses'
    LABEL = 'atom types'
    COLUMNS = ['mass', 'comment']

    @classmethod
    def fromAtoms(cls, atoms):
        """
        Construct a mass instance from atoms.

        :param atoms `pd.DataFrame`: the atoms.
        :return `cls`: the mass instance.
        """
        return cls([[x.atomic_weight, x.symbol] for x in atoms.itertuples()])

    @methodtools.lru_cache()
    @property
    def element(self, rex=r'^(\w+)'):
        """
        The element of each atom type.

        :param rex str: the regular expression to extract the symbol
        :return 'np.ndarray': the element of the atom types.
        """
        return self.comment.str.extract(rex).values.flatten()

    def write(self, *args, **kwargs):
        """
        See parent.
        """
        self.comment = ' ' + self.comment + ' '
        super().write(*args, **kwargs)  # e.g., 1 28.0860 # Si #
        self.comment = self.comment.str.strip()

    @classmethod
    def fromLines(cls, *args, **kwargs):
        """
        See parent.
        """
        mass = super().fromLines(*args, **kwargs)
        mass.comment = mass.comment.str.strip()
        return mass


class Id(Base):
    """
    The atomic information of the int type.
    """
    NAME = 'Atoms'
    LABEL = 'atoms'
    TYPE_COL = [TYPE_ID]
    COLUMNS = [ATOM1, TYPE_ID]
    SLICE = 0
    _metadata = ['cached']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached = None

    @classmethod
    def fromAtoms(cls, atoms):
        """
        Construct an instance from atoms.

        :param atoms `iterator` of 'Chem.rdchem.Atom': the atoms.
        :return `cls`: the instance.
        """
        return cls([[x.GetIdx(), x.GetIntProp(TYPE_ID)] for x in atoms])

    def to_numpy(self, gids):
        """
        Covert the DataFrame to a NumPy array with mapping.

        :param gids 'np.ndarray': map the atom ids to global ids.
        :return 'np.ndarray': the numpy array
        """
        if self.cached is None:
            self.cached = self.values
        array = self.cached.copy()
        array[:, self.SLICE] = gids[array[:, self.SLICE]]
        return array

    @classmethod
    def concatenate(cls, arrays, type_map):
        """
        Join a sequence of arrays along an existing axis with mapping.

        :param arrays sequence of array_like: the arrays to concatenate.
        :param type_map 'IntArray': map the type ids as consecutive integers
        :return 'pd.DataFrame': pandas DataFrame from the concatenated array.
        """
        if not arrays:
            return cls()
        array = np.concatenate(arrays)
        index = cls.COLUMNS.index(TYPE_ID)
        array[:, index] = type_map.index(array[:, index])
        return cls(array)


class Atom(Id):
    """
    The total atomic information of all data types.
    """
    COLUMNS = Id.COLUMNS + symbols.XYZU
    FMT = '%i %i %.4f %.4f %.4f'

    def write(self, *args, index_column=ATOM1, **kwargs):
        """
        See parent.
        """
        super().write(*args, index_column=index_column, **kwargs)

    @classmethod
    def fromData(cls, ids, *args):
        """
        Construct atom from ids and non-integer data types.

        :param ids 'Id': the ids (integer data types).
        """
        atoms = cls(np.concatenate((ids.values, *args), axis=1))
        atoms[ids.COLUMNS] = atoms[ids.COLUMNS].astype(np.uint32)
        return atoms


class Conf(structure.Conf):
    """
    The conformer class for global ids and types.
    """

    @property
    def ids(self):
        """
        The ids of this conformer.

        :return `np.ndarray`: global and type ids.
        """
        return self.GetOwningMol().ids.to_numpy(self.gids)


class Mol(structure.Mol):
    """
    The molecule class for atom typing and int properties.
    """
    Conf = Conf
    Id = Id

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)
        self.type()

    def type(self):
        """
        Type atoms.
        """
        for atom in self.GetAtoms():
            atom.SetIntProp(TYPE_ID, atom.GetAtomicNum())

    @functools.cached_property
    def ids(self):
        """
        The ids of the molecules.

        :return `Id`: the atom ids.
        """
        return self.Id.fromAtoms(self.GetAtoms())


class Struct(structure.Struct):
    """
    The atomic structure.
    """
    Id = Id
    Atom = Atom
    Mol = Mol
    Script = lmpin.Script
    DESCR = 'LAMMPS Description # {style}\n'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Atomic number of Og element
        self.atm_types = numpyutils.IntArray(shape=119)
        self.outfile = f"{self.options.JOBNAME}.data" if self.options else None

    def setUp(self, *arg, **kwargs):
        """
        See parent.
        """
        super().setUp(*arg, **kwargs)
        for mol in self.mols:
            self.setTypeMap(mol)

    def setTypeMap(self, mol):
        """
        Set the type map for atom.
        """
        self.atm_types[[x.GetAtomicNum() for x in mol.GetAtoms()]] = True

    def write(self):
        """
        Write out the LAMMPS datafile and script.
        """
        self.writeData()
        self.log(f'Data file written into {self.outfile}')
        jobutils.Job.reg(self.outfile)
        self.script.write()
        self.log(f'In script written into {self.script.outfile}')
        jobutils.Job.reg(self.script.outfile, file=True)

    def writeData(self):
        """
        Write out a LAMMPS datafile.
        """
        with open(self.outfile, 'w') as self.hdl:
            self.hdl.write(self.DESCR.format(style=self.script.V_ATOM_STYLE))
            self.hdl.write("\n")
            self.atoms.writeCount(self.hdl)
            self.hdl.write("\n")
            self.masses.writeCount(self.hdl)
            self.hdl.write("\n")
            self.box.write(self.hdl)
            self.masses.write(self.hdl)
            self.atoms.write(self.hdl)

    @property
    def atoms(self):
        """
        Atoms.

        :return `Atom`: atoms.
        """
        return self.Atom.fromData(self.ids, self.GetPositions())

    @functools.cached_property
    def ids(self):
        """
        The ids of the atoms.

        :return `Id`: information such as global ids and type ids.
        """
        return self.Id.concatenate([x.ids for x in self.conf], self.atm_types)

    def GetPositions(self):
        """
        The xyz coordinates of atoms.

        :return 'np.ndarray': the coordinates.
        """
        return np.concatenate([x.GetPositions() for x in self.conf],
                              dtype=np.float32)

    @property
    def masses(self, atoms=table.TABLE.reset_index()):
        """
        Atom masses.

        :param atoms `pd.DataFrame`: atoms from the periodic table.
        :return `Mass`: mass of each type of atom.
        """
        return Mass.fromAtoms(atoms.iloc[self.atm_types.on])

    @functools.cached_property
    def box(self):
        """
        Get the pbc box.

        :return `pbc.Box`: the box.
        """
        if self.mols and self.mols[0].vecs:
            return pbc.Box.fromParams(*self.mols[0].vecs)
        return pbc.Box()

    @functools.cached_property
    def script(self):
        """
        Get the LAMMPS in-script writer.

        :return `Script`: the in-script.
        """
        return self.Script(struct=self)

    @functools.cached_property
    def ff(self):
        """
        Force field object.

        :return str: the force field file or parser.
        """
        return sw.get_file(*self.masses.element.tolist())


class Reader:
    """
    LAMMPS Data file reader.

    https://docs.lammps.org/read_data.html#format-of-a-data-file
    """
    Id = Id
    Mass = Mass
    Atom = Atom
    NAMES = {x.NAME: x.LABEL for x in [Mass, Atom]}
    FLOAT_RE = r"[+-]?[\d\.]+"

    def __init__(self, data_file=None):
        """
        :param data_file str: data file with path
        """
        self.data_file = data_file

    @functools.cached_property
    def lines(self):
        """
        Return the lines.
        """
        with open(self.data_file, 'r') as df_fh:
            raw = df_fh.readlines()
        lines = {i: self.name_re.match(x) for i, x in enumerate(raw)}
        # The block name occupies one lien and there is one empty line below
        lines = {x.group(): i + 2 for i, x in lines.items() if x}
        header = raw[:min(lines.values())]
        # 'atoms': 1620, 'bonds': 1593, 'angles': 1566 ...
        # 'atom types': 7, 'bond types': 6, 'angle types': 5 ...
        matches = [self.count_re.match(x) for x in header]
        counts = {x.group(2): int(x.group(1)) for x in matches if x}
        lines = {x: raw[y:y + counts[self.NAMES[x]]] for x, y in lines.items()}
        # 'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], ..
        lines[pbc.Box.NAME] = [x for x in header if self.box_re.match(x)]
        lines[pbc.Box.NAME] += [x for x in header if self.tilt_re.match(x)]
        return lines

    @functools.cached_property
    def name_re(self):
        """
        The regular expression of any names. (e.g. 'Masses', 'Atoms')

        :return 're.pattern': the name regular expression
        """
        return re.compile(f"^{'|'.join([x for x in self.NAMES])}$")

    @functools.cached_property
    def count_re(self):
        """
        The regular expression of any counts. (e.g. 'atom types', 'atoms')

        :return 're.pattern': the count regular expression
        """
        labels = [x for x in self.NAMES.values()]
        return re.compile(rf"^([0-9]+)\s+({'|'.join(labels)})$")

    @functools.cached_property
    def box_re(self):
        """
        The regular expression of any box lines. (e.g. 'xlo xhi', 'ylo yhi')

        :return 're.pattern': the count regular expression
        """
        values = pbc.Box.getLabels().values()
        labels = '|'.join([f'{x}{symbols.SPACE}{y}' for x, y in zip(*values)])
        return re.compile(rf"^{self.FLOAT_RE}\s{self.FLOAT_RE}\s({labels}).*$")

    @functools.cached_property
    def tilt_re(self):
        """
        The regular expression of tge tilt line. (e.g. 'xy xz yz')

        :return 're.pattern': the count regular expression
        """
        rex = r'\s'.join([self.FLOAT_RE] * 3 + pbc.Box.TILT_LABEL)
        return re.compile(rf"^{rex}.*$")

    @functools.cached_property
    def box(self):
        """
        Parse the box section.

        :return `Box`: the box
        """
        return self.fromLines(pbc.Box)

    @functools.cached_property
    def masses(self):
        """
        Parse the mass section for masses and elements.

        :return `Mass`: the masses of atoms.
        """
        return self.fromLines(self.Mass)

    @functools.cached_property
    def elements(self, name='element'):
        """
        The elements of all atoms.

        :param name str: the name of the element column.
        :return `pd.DataFrame`: the element dataframe with atom ids
        """
        data = self.masses.element[self.atoms.type_id]
        return pd.DataFrame(data, index=self.atoms.index, columns=[name])

    @functools.cached_property
    def atoms(self):
        """
        The atom section (the atom block of the int data type).

        :return `Atom`: the atom information such as atom id, molecule id,
            type id, charge, position, etc.
        """
        return self.fromLines(self.Atom).reset_index(names=[ATOM1])

    def fromLines(self, BlockClass):
        """
        Parse a block of lines.

        :param BlockClass: the class to handle a block.
        :return BlockClass: the parsed block.
        """
        return BlockClass.fromLines(self.lines.get(BlockClass.NAME, []))

    def allClose(self, other, atol=1e-08, rtol=1e-05, equal_nan=True):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other 'Reader': the other data reader to compare against.
        :param atol `float`: The relative tolerance parameter (see Notes).
        :param rtol `float`: The absolute tolerance parameter (see Notes).
        :param equal_nan `bool`: If True, NaNs are considered close.
        :return `bool`: whether two data are close.
        """
        kwargs = dict(atol=atol, rtol=rtol, equal_nan=equal_nan)
        if not self.box.allClose(other.box, **kwargs):
            return False
        if not self.masses.allClose(other.masses, **kwargs):
            return False
        if not self.atoms.allClose(other.atoms, **kwargs):
            return False
        return True

    @classmethod
    def getStyle(cls,
                 pathname,
                 rex=re.compile(Struct.DESCR.replace('{style}', '(.*)$'))):
        """
        Get the lammps data file style.

        :param pathname str: the lammps data file with path.
        :param rex 're.Pattern': the pattern to search the atom block style.
        :return str: the style
        """
        with open(pathname, 'r') as fh:
            match = rex.match(fh.readline())
        if match:
            return match.group(1)

    @property
    def weights(self):
        """
        The masses of every atom.

        :return np.ndarray: the masses of every atom.
        """
        return self.masses.mass[self.atoms.type_id].to_numpy()

    @classmethod
    def fromTest(cls, *args):
        """
        Return a reader instance from test data.

        :return reader: the reader objet.
        """
        return cls(envutils.test_data(*args))
