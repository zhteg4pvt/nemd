# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses a LAMMPS data file in the atomic atom_style.

Atoms section: atom-ID atom-type x y z
"""
import base64
import functools
import io
import re

import numpy as np
import pandas as pd

from nemd import box
from nemd import forcefield
from nemd import numpyutils
from nemd import structure
from nemd import symbols
from nemd import table

TYPE_ID = symbols.TYPE_ID
ATOM1 = 'atom1'
ENE = 'ene'


class Base(box.Base):
    """
    The base class for the coefficient and topology.
    """

    def writeCount(self, fh):
        """
        Write the count with the label appended.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler
        """
        fh.write(f'{self.shape[0]} {self.LABEL}\n')

    @classmethod
    def fromLines(cls, *args, index_col=0, **kwargs):
        """
        Construct a mass instance from a list of lines.

        :param index_col int: Column(s) to use as row label(s)
        :return 'Mass' or subclass instance: the mass.
        """
        return super().fromLines(*args, index_col=index_col, **kwargs)


class Mass(Base):
    """
    The masses class for each type of atoms.
    """

    NAME = 'Masses'
    MASS = 'mass'
    COMMENT = 'comment'
    COLUMN_LABELS = [MASS, COMMENT]
    CMT_RE = '^(\w+)$'
    LABEL = 'atom types'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._element = None

    @classmethod
    def fromAtoms(cls, atoms):
        """
        Construct a mass instance from atoms.

        :param atoms `pd.DataFrame`: the atoms.
        :return `cls`: the mass instance.
        """
        return cls([[x.atomic_weight, x.Index] for x in atoms.itertuples()])

    @property
    def element(self):
        """
        Set and cache the element of the atom types.

        :return: the element of the atom types.
        :rtype: 'numpy.ndarray'
        """
        if self._element is not None:
            return self._element
        self._element = self.comment.str.extract(self.CMT_RE).values.flatten()
        return self._element

    def write(self, *args, **kwargs):
        """
        In addition to the parent class, add a space before and after the
        comment (e.g., 1 28.0860 # Si #).
        """
        self[self.COMMENT] = ' ' + self[self.COMMENT] + ' '
        super().write(*args, **kwargs)
        self[self.COMMENT] = self[self.COMMENT].str.strip()

    @classmethod
    def fromLines(cls, *args, **kwargs):
        """
        In addition to the parent class, strip the space wrapping the comment.

        :return 'Mass': the mass.
        """
        mass = super().fromLines(*args, **kwargs)
        mass.comment = mass.comment.str.strip()
        return mass


class PairCoeff(Base):
    """
    The pair coefficients between non-bonded atoms in the system.
    """

    NAME = 'Pair Coeffs'
    COLUMN_LABELS = [ENE, 'dist']
    LABEL = 'atom types'

    @classmethod
    def fromLines(cls, *args, index_col=0, **kwargs):
        """
        Construct a mass instance from a list of lines.

        :param index_col int: Column(s) to use as row label(s)
        :return 'Mass' or subclass instance: the mass.
        """
        return super().fromLines(*args, index_col=index_col, **kwargs)


class XYZ(PairCoeff):
    """
    The xyz coordinates of the atoms.
    """

    NAME = 'XYZ'
    COLUMN_LABELS = symbols.XYZU
    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['_cached']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = None

    def to_numpy(self, copy=True, id_map=None, index=None):
        """
        Covert the DataFrame to a NumPy array.

        :param copy bool:the returned is not a view on another array if True
        :param id_map 'np.ndarray': map the atom ids to global ids.
        :return 'numpy.ndarray': the numpy array
        """
        if self._cached is None:
            self._cached = super().values
        if not self._cached.size:
            return
        array = self._cached.copy() if copy else self._cached
        if not array.size or id_map is None:
            return array
        if index is None:
            array[:, 1:] = id_map[array[:, 1:]]
            return array
        array[:, 0] = id_map[array[:, 0]]
        return array

    @classmethod
    def concatenate(cls, arrays, type_map=None):
        """
        Join a sequence of arrays along an existing axis.

        :param arrays sequence of array_like: the arrays to concatenate.
        :param arrays 'IntArray': map the type ids as consecutive integers
        :return 'pd.DataFrame': pandas DataFrame from the concatenated array.
        """
        arrays = [x for x in arrays if x is not None]
        if not arrays:
            return cls()
        array = np.concatenate(arrays)
        if type_map is None:
            return cls(array)
        index = cls.COLUMN_LABELS.index(TYPE_ID)
        array[:, index] = type_map.index(array[:, index])
        return cls(array)


class Atom(XYZ):
    """
    The atomic information of the int data type.
    """

    NAME = 'Atoms'
    TYPE_COL = [TYPE_ID]
    COLUMN_LABELS = [ATOM1, TYPE_ID]
    LABEL = 'atoms'

    @classmethod
    def concat(cls, objs, ignore_index=True, **kwargs):
        """
        Concatenate the instances and re-index the row from 1.

        :param objs: the instances to concatenate.
        :type objs: list of (sub-)class instances.
        :return: the concatenated data
        :rtype: (sub-)class instances
        """
        if not len(objs):
            return cls()
        return pd.concat(objs, ignore_index=ignore_index, **kwargs)


class AtomBlock(Atom):
    """
    The total atomic information of all data types.
    """

    COLUMN_LABELS = Atom.COLUMN_LABELS + XYZ.COLUMN_LABELS
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
        return self.GetOwningMol().atoms.to_numpy(id_map=self.id_map, index=0)


class Mol(structure.Mol):
    """
    The molecule class for atom typing and int properties.
    """

    ConfClass = Conformer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atoms = None
        if self.delay:
            return
        self.type()
        self.setAtoms()

    def type(self):
        """
        Type atoms and set charges.
        """
        for atom in self.GetAtoms():
            atom.SetIntProp(TYPE_ID, atom.GetAtomicNum())

    def setAtoms(self):
        """
        The atoms of the molecules.

        :return 'Atom': Atoms with type ids and charges
        """
        type_ids = [x.GetIntProp(TYPE_ID) for x in self.GetAtoms()]
        aids = [x.GetIdx() for x in self.GetAtoms()]
        self.atoms = Atom({ATOM1: aids, TYPE_ID: type_ids})

    def GetPositions(self):
        return np.concatenate([x.GetPositions() for x in self.confs],
                              dtype=np.float32)

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
    The structure class with interface to LAMMPS data file.
    """

    Atom = Atom
    MolClass = Mol
    DESCR = 'LAMMPS Description # {style}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Atomic number of Og element
        self.atm_types = numpyutils.IntArray(118)

    def addMol(self, mol):
        """
        Add a molecule to the structure.

        :param mol: add this molecule to the structure
        :type mol: Mol
        :return: the added molecule
        :rtype: Mol
        """
        mol = super().addMol(mol)
        self.setTypeMap(mol)
        return mol

    def writeData(self, nofile=False):
        """
        Write out a LAMMPS datafile or return the content.

        :param nofile bool: return the content as a string if True.
        """
        with io.StringIO() if nofile else open(self.datafile, 'w') as self.hdl:
            self.hdl.write(f"{self.DESCR.format(style=self.V_ATOM_STYLE)}\n\n")
            self.atoms.writeCount(self.hdl)
            self.hdl.write("\n")
            self.masses.writeCount(self.hdl)
            self.hdl.write("\n")
            self.box.write(self.hdl)
            self.masses.write(self.hdl)
            self.atom_blk.write(self.hdl)
            return self.getContents() if nofile else None

    def setTypeMap(self, mol):
        """
        Set the type map for atom.
        """
        atomic_num = [x.GetAtomicNum() for x in mol.GetAtoms()]
        self.atm_types[atomic_num] = True

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

    def simulation(self, *arg, **kwargs):
        """
        Write command to further equilibrate the system with structure
        information considered.
        """
        super().simulation(*arg, atom_total=self.atom_total, **kwargs)

    def getContents(self):
        """
        Return datafile contents in base64 encoding.

        :return `bytes`: the contents of the data file in base64 encoding.
        """
        self.hdl.seek(0)
        contents = base64.b64encode(self.hdl.read().encode("utf-8"))
        return b','.join([b'lammps_datafile', contents])

    @property
    @functools.cache
    def atom_blk(self):
        """
        The total atomic information of all data types.

        :return `Atom`: information such as global ids, type ids, and coordinates.
        """
        return AtomBlock(self.atoms.astype(np.float32).join(self.xyz))

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
        return XYZ(np.concatenate([x.GetPositions() for x in self.molecules]))

    @property
    def masses(self):
        """
        Atom masses.

        :return `Mass`: mass of each type of atom.
        """
        return Mass.fromAtoms(table.TABLE.iloc[self.atm_types.values])

    @property
    def pair_coeffs(self):
        """
        Non-bonded atom pair coefficients.

        :return `PairCoeff`: the interaction between non-bond atoms.
        """
        vdws = self.ff.vdws.loc[self.atm_types.values]
        return PairCoeff([[x.ene, x.dist] for x in vdws.itertuples()])


class Reader:
    """
    LAMMPS Data file reader.

    https://docs.lammps.org/read_data.html#format-of-a-data-file
    """
    Atom = Atom
    Mass = Mass
    AtomBlock = AtomBlock
    BLOCK_CLASSES = [Mass, AtomBlock]
    BLOCK_NAMES = [x.NAME for x in BLOCK_CLASSES]
    BLOCK_LABELS = [x.LABEL for x in BLOCK_CLASSES]
    NAME_RE = re.compile(f"^{'|'.join(BLOCK_NAMES)}$")
    COUNT_RE = re.compile(f"^[0-9]+\s+({'|'.join(BLOCK_LABELS)})$")
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
            match = self.NAME_RE.match(line)
            if not match:
                continue
            # The block name occupies one lien and there is one empty line below
            names[match.group()] = idx + 2

        counts = {}
        for line in self.lines[:min(names.values())]:
            match = self.COUNT_RE.match(line)
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
        box_lines = [i for i, x in enumerate(lines) if box.Box.RE.match(x)]
        self.name[box.Box.LABEL] = slice(min(box_lines), max(box_lines) + 1)

    @property
    @functools.cache
    def box(self):
        """
        Parse the box section.

        :return `Box`: the box
        """
        return self.fromLines(box.Box)

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
    def atoms(self):
        """
        The atom section (the atom block of the int data type).

        :return `Atom`: the atom information such as atom id, molecule id,
            type id, charge, position, etc.
        """
        return self.atom_blk[self.Atom.COLUMN_LABELS]

    @property
    @functools.cache
    def atom_blk(self):
        """
        The atom block..

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
        return self.atom_blk[XYZ.COLUMN_LABELS]

    @property
    @functools.cache
    def pair_coeffs(self):
        """
        Paser the pair coefficient section.

        :return `PairCoeff`: the pair coefficients between non-bonded atoms.
        """
        return self.fromLines(PairCoeff)

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