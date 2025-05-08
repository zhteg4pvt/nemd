# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses a LAMMPS data file in the atomic atom_style.

Atoms section: atom-ID atom-type x y z
"""
import base64
import functools
import re

import methodtools
import numpy as np
import pandas as pd

from nemd import forcefield
from nemd import numpyutils
from nemd import pbc
from nemd import structure
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
        Set and cache the element of the atom types.

        :param rex str: the regular expression to extract the symbol
        :return 'numpy.ndarray': the element of the atom types.
        """
        return self.comment.str.extract(rex).values.flatten()

    def write(self, *args, **kwargs):
        """
        See parent.
        """
        self.comment = ' ' + self.comment + ' '
        # e.g., 1 28.0860 # Si #
        super().write(*args, **kwargs)
        self.comment = self.comment.str.strip()

    @classmethod
    def fromLines(cls, *args, **kwargs):
        """
        See parent.
        """
        mass = super().fromLines(*args, **kwargs)
        mass.comment = mass.comment.str.strip()
        return mass


class XYZ(Base):
    """
    The xyz coordinates of the atoms.
    """
    NAME = 'XYZ'
    COLUMNS = symbols.XYZU

    @classmethod
    def concatenate(cls, arrays, type_map=None):
        """
        Join a sequence of arrays along an existing axis.

        :param arrays sequence of array_like: the arrays to concatenate.
        :param type_map 'IntArray': map the type ids as consecutive integers
        :return 'pd.DataFrame': pandas DataFrame from the concatenated array.
        """
        arrays = [x for x in arrays if x is not None]
        if not arrays:
            return cls()
        array = np.concatenate(arrays)
        if type_map is None:
            return cls(array)
        index = cls.COLUMNS.index(TYPE_ID)
        array[:, index] = type_map.index(array[:, index])
        return cls(array)


class Atom(XYZ):
    """
    The atomic information of the int type.
    """
    NAME = 'Atoms'
    LABEL = 'atoms'
    TYPE_COL = [TYPE_ID]
    COLUMNS = [ATOM1, TYPE_ID]
    SLICE = 0

    @classmethod
    def fromAtoms(cls, atoms):
        """
        Construct an instance from atoms.

        :param atoms `iterator` of 'Chem.rdchem.Atom': the atoms.
        :return `cls`: the instance.
        """
        return cls([[x.GetIdx(), x.GetIntProp(TYPE_ID)] for x in atoms])

    def to_numpy(self, id_map=None):
        """
        Covert the DataFrame to a NumPy array.

        :param id_map 'np.ndarray': map the atom ids to global ids.
        :return 'numpy.ndarray': the numpy array
        """
        array = self.values.copy()
        array[:, self.SLICE] = id_map[array[:, self.SLICE]]
        return array

    @classmethod
    def concat(cls, objs, ignore_index=True, **kwargs):
        """
        Concatenate the instances and re-index the row from 1.

        :param objs `list`: the (sub-)class instances to concatenate.
        :return (sub-)class instances: the concatenated data
        """
        if not objs:
            return cls()
        return pd.concat(objs, ignore_index=ignore_index, **kwargs)


class AtomBlock(Atom):
    """
    The total atomic information of all data types.
    """
    COLUMNS = Atom.COLUMNS + XYZ.COLUMNS
    FMT = '%i %i %.4f %.4f %.4f'

    def write(self, *args, index_column=ATOM1, **kwargs):
        """
        See the parent class.
        """
        super().write(*args, index_column=index_column, **kwargs)


class Conformer(structure.Conformer):
    """
    The conformer class for global ids and types.
    """

    @property
    def atoms(self):
        """
        Atoms for this conformer.

        :return `'numpy.ndarray'`: information such as global ids, molecule ids.
        """
        return self.GetOwningMol().atoms.to_numpy(id_map=self.id_map)


class Mol(structure.Mol):
    """
    The molecule class for atom typing and int properties.
    """
    ConfClass = Conformer
    AtomClass = Atom

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.delay:
            return
        self.type()

    def type(self):
        """
        Type atoms.
        """
        for atom in self.GetAtoms():
            atom.SetIntProp(TYPE_ID, atom.GetAtomicNum())

    @property
    @functools.cache
    def atoms(self):
        """
        The atoms of the molecules.

        :return `Atom`: the atoms.
        """
        return self.AtomClass.fromAtoms(self.GetAtoms())

    @property
    @functools.cache
    def ff(self):
        """
        Force field parser for atoms, charges, and other parameters.

        :return `oplsua.Parser`: the force field parser.
        """
        return self.struct.ff if self.struct else None


class Struct(structure.Struct):
    """
    The atomic structure.
    """
    Atom = Atom
    MolClass = Mol
    DESCR = 'LAMMPS Description # {style}'

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, **kwargs)
        self.options = options
        # Atomic number of Og element
        self.atm_types = numpyutils.IntArray(shape=119)
        self.box = None

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

    @property
    @functools.cache
    def atom_blk(self):
        """
        The total atomic information of all data types.

        :return `Atom`: global ids, type ids, and coordinates.
        """
        return AtomBlock(self.atoms.astype(float).join(self.xyz))

    @property
    @functools.cache
    def atoms(self):
        """
        The atomic information of the int data type such as ids and types.

        :return `Atom`: information such as global ids and type ids.
        """
        atoms = [x.atoms for x in self.conformer]
        return self.Atom.concatenate(atoms, type_map=self.atm_types)

    @property
    def xyz(self):
        """
        The atom coordinates.

        :return `XYZ`: the atom coordinates.
        """
        return XYZ(np.concatenate([x.GetPositions() for x in self.mols]))

    @property
    def masses(self):
        """
        Atom masses.

        :return `Mass`: mass of each type of atom.
        """
        atoms = table.TABLE.iloc[self.atm_types.on].reset_index()
        return Mass.fromAtoms(atoms)

    @property
    @functools.cache
    def ff(self):
        """
        Force field object by name and arguments.

        :return str or `oplsua.Parser`: the force field file or parser.
        """
        if self.options is None:
            return
        return forcefield.get(*self.options.force_field, struct=self)


class Reader:
    """
    LAMMPS Data file reader.

    https://docs.lammps.org/read_data.html#format-of-a-data-file
    """
    Atom = Atom
    Mass = Mass
    AtomBlock = AtomBlock
    BLOCK_CLASSES = [Mass, AtomBlock]
    DESCR_RE = Struct.DESCR.replace('{style}', '(.*)$')

    def __init__(self, data_file=None, contents=None, delay=False):
        """
        :param data_file str: data file with path
        :param contents `bytes`: parse the contents if data_file not provided.
        :param delay bool: delay the reading and indexing.
        """
        self.data_file = data_file
        self.contents = contents
        self.lines = None
        self.name = {}
        if delay:
            return
        self.read()
        self.index()

    def read(self):
        """
        Read the data file or content into lines.
        """
        if self.data_file:
            with open(self.data_file, 'r') as df_fh:
                self.lines = df_fh.readlines()
                return

        content_type, content_string = self.contents.split(b',')
        decoded = base64.b64decode(content_string)
        self.lines = decoded.decode("utf-8").splitlines()

    def index(self):
        """
        Index the lines by block markers, and Parse the descr section for
        topo counts and type counts.
        """

        names = {}
        for idx, line in enumerate(self.lines):
            match = self.name_re.match(line)
            if not match:
                continue
            # The block name occupies one lien and there is one empty line below
            names[match.group()] = idx + 2

        counts = {}
        for line in self.lines[:min(names.values())]:
            match = self.count_re.match(line)
            if not match:
                continue
            # 'atoms': 1620, 'bonds': 1593, 'angles': 1566 ...
            # 'atom types': 7, 'bond types': 6, 'angle types': 5 ...
            counts[match.group(1)] = int(line.split(match.group(1))[0])

        for block_class in self.BLOCK_CLASSES:
            if block_class.NAME not in names:
                continue
            idx = names[block_class.NAME]
            count = counts[block_class.LABEL]
            self.name[block_class.NAME] = slice(idx, idx + count)

        lines = self.lines[:min(names.values())]
        # 'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], ..
        box_lines = [i for i, x in enumerate(lines) if self.box_re.match(x)]
        self.name[pbc.Box.LABEL] = slice(min(box_lines), max(box_lines) + 1)

    @property
    @functools.cache
    def name_re(self):
        """
        The regular expression of any names. (e.g. 'Masses', 'Atoms')

        :return 're.pattern': the name regular expression
        """
        names = [x.NAME for x in self.BLOCK_CLASSES]
        return re.compile(f"^{'|'.join(names)}$")

    @property
    @functools.cache
    def count_re(self):
        """
        The regular expression of any counts. (e.g. 'atom types', 'atoms')

        :return 're.pattern': the count regular expression
        """
        labels = [x.LABEL for x in self.BLOCK_CLASSES]
        return re.compile(rf"^[0-9]+\s+({'|'.join(labels)})$")

    @property
    @functools.cache
    def box_re(self, float_re=r"[+-]?[\d\.]+"):
        """
        The regular expression of any box lines. (e.g. 'xlo xhi', 'ylo yhi')

        :return 're.pattern': the count regular expression
        """
        # FIXME: read tilt factors of the triclinic box
        values = pbc.Box.getLabels().values()
        labels = '|'.join([f'{x}{symbols.SPACE}{y}' for x, y in zip(*values)])
        return re.compile(rf"^{float_re}\s+{float_re}\s+({labels}).*$")

    @property
    @functools.cache
    def box(self):
        """
        Parse the box section.

        :return `Box`: the box
        """
        return self.fromLines(pbc.Box)

    @property
    @functools.cache
    def masses(self):
        """
        Parse the mass section for masses and elements.

        :return `Mass`: the masses of atoms.
        """
        return self.fromLines(self.Mass)

    @property
    @functools.cache
    def elements(self, name='element'):
        """
        The elements of all atoms.

        :param name str: the name of the element column.
        :return `pd.DataFrame`: the element dataframe with atom ids
        """
        data = self.masses.element[self.atoms.type_id]
        return pd.DataFrame(data, index=self.atoms.index, columns=[name])

    @property
    @functools.cache
    def atoms(self):
        """
        The atom section (the atom block of the int data type).

        :return `Atom`: the atom information such as atom id, molecule id,
            type id, charge, position, etc.
        """
        return self.atom_blk[self.Atom.COLUMNS]

    @property
    @functools.cache
    def atom_blk(self):
        """
        The atom block.

        :return `AtomBlock`: the atom information such as atom id, molecule id,
            type id, charge, position, etc.
        """
        return self.fromLines(self.AtomBlock).reset_index(names=[ATOM1])

    @property
    @functools.cache
    def xyz(self):
        """
        Parse the atom section.

        :return `XYZ`: the atom coordinates.
        """
        return self.atom_blk[XYZ.COLUMNS]

    def fromLines(self, BlockClass):
        """
        Parse a block of lines from the datafile.

        :param BlockClass: the class to handle a block.
        :return BlockClass: the parsed block.
        """
        name = BlockClass.NAME if BlockClass.NAME else BlockClass.LABEL
        if name not in self.name:
            return BlockClass.fromLines([])
        lines = self.lines[self.name[name]]
        return BlockClass.fromLines(lines)

    def allClose(self, other, atol=1e-08, rtol=1e-05, equal_nan=True):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other `float`: the other data reader to compare against.
        :param atol `float`: The relative tolerance parameter (see Notes).
        :param rtol `float`: The absolute tolerance parameter (see Notes).
        :param equal_nan `bool`: If True, NaNs are considered close.
        :return `bool`: whether two data are close.
        """
        if not isinstance(other, Reader):
            return False
        kwargs = dict(atol=atol, rtol=rtol, equal_nan=equal_nan)
        if not self.box.allClose(other.box, **kwargs):
            return False
        if not self.masses.allClose(other.masses, **kwargs):
            return False
        if not self.pair_coeffs.allClose(other.pair_coeffs, **kwargs):
            return False
        if not self.atom_blk.allClose(other.atom_blk, **kwargs):
            return False
        return True

    @classmethod
    def getStyle(cls, pathname):
        """
        Get the lammps data file style.

        :param pathname str: the lammps data file with path.
        :return str: the style
        """
        with open(pathname, 'r') as fh:
            match = re.match(cls.DESCR_RE, fh.readline())
            if match:
                return match.group(1)
