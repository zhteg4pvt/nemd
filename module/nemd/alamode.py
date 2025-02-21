# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
ALAMODE utilities.

https://github.com/ttadano/alamode
"""
import collections
import functools
import itertools

import pandas as pd

from nemd import constants
from nemd import process
from nemd import stillinger
from nemd import symbols
from nemd import table
from nemd import xtal

INDENT = "  "
SUGGEST = symbols.SUGGEST
OPTIMIZE = symbols.OPTIMIZE
PHONONS = symbols.PHONONS


class Struct(stillinger.Struct):
    """
    Structure class with force reading.
    """

    CUSTOM_EXT = process.Lmp.EXT

    def traj(self, force=True, sort=False, fmt="float '%20.15f'"):
        """
        See parent for docs.
        """
        super().traj(force=force, sort=sort, fmt=fmt)


def exe(obj, **kwargs):
    """
    Choose class, build command, execute subprocess, and return output files.

    :param obj str, Crystal, or Struct: based on which the Runner is chosen
    :return list: the output files
    """
    if isinstance(obj, Struct):
        Runner = process.Lmp
    elif isinstance(obj, Crystal):
        Runner = process.Alamode
    elif obj in [process.Tools.DISPLACE, process.Tools.EXTRACT]:
        Runner = process.Tools
    runner = Runner(obj, **kwargs)
    runner.run()
    return runner.outfiles


class Crystal(xtal.Crystal):

    IN = '.in'

    def __init__(self, *args, mode=SUGGEST, **kwargs):
        """
        :param mode str: the mode of the calculation
        """
        super().__init__(*args, **kwargs)
        self.mode = mode
        if self.options is None:
            return
        self.inscript = f"{self.options.JOBNAME}{self.IN}"

    def write(self):
        """
        Write out the alamode input script.
        """
        with open(self.inscript, 'w') as fh:
            self.general.write(fh)
            self.optimize.write(fh)
            self.interaction.write(fh)
            self.cutoff.write(fh)
            self.cell.write(fh)
            self.position.write(fh)
            self.kpoint.write(fh)

    @property
    def general(self):
        return General(self)

    @property
    def optimize(self):
        """
        Set the &optimize-filed.
        """
        return Optimize(self)

    @property
    def interaction(self):
        """
        Set the &interaction-field.
        """
        return Interaction(self)

    @property
    def cutoff(self):
        """
        Set the cutoff for neighbor searching.
        """
        return Cutoff(self)

    @property
    def kpoint(self):
        """
        Set the &kpoint-field
        """
        return Kpoint(self)

    @property
    def cell(self):
        """
        Set the &cell-field.
        """
        return Cell(self)

    @property
    def position(self):
        return Position(self)


class Base(dict):
    """
    Class to store key value pairs by dot notation.
    """

    NAME = 'name'
    MODES = None
    SEP = ' = '
    FLOAT_FORMAT = None

    def __init__(self, crystal):
        """
        :param crystal Crystal: the crystal structure
        """
        super(Base, self).__setattr__('crystal', crystal)
        if self.MODES and self.crystal.mode not in self.MODES:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the block content.
        """
        pass

    def __setattr__(self, key, value):
        """
        Set the key = value by dot notation.

        :param key str: the key of the key / value pair
        :param value object convertible to str: the value to be saved.
        """
        self[key] = value

    def __getattr__(self, key):
        """
        Get the attribute from the class and the stored data.

        :param key str: the key to retrieve the value.
        :return any: the retrieved value.
        """
        try:
            return super().__getattr__(key)
        except AttributeError:
            return self[key]

    def write(self, fh):
        """
        Write the key value pairs to the file handle.

        :param fh '_io.TextIOWrapper': the file handle to write to.
        """
        if not self:
            return

        fh.write(f"{symbols.AND}{self.NAME}\n")
        for key, val in self.items():
            fh.write(f"{INDENT}{key}{self.SEP}{self.format(val)}\n")
        fh.write(f"{symbols.FORWARDSLASH}\n\n")

    def format(self, value):
        """
        Format the value to be written out.

        :param value object convertible to str: the value to be written out.
        :return str: the str representation of the value.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, float) and self.FLOAT_FORMAT:
            return format(value, self.FLOAT_FORMAT)
        if isinstance(value, collections.abc.Iterable):
            if not self.FLOAT_FORMAT:
                return symbols.SPACE.join(map(str, value))
            val = [
                format(value, self.FLOAT_FORMAT) if isinstance(x, float) else x
                for x in value
            ]
            return symbols.SPACE.join(map(str, val))
        return value


class General(Base):

    NAME = 'general'

    def setUp(self):
        """
        Set up the key / value pairs for the &general-field.
        """
        self.PREFIX = self.crystal.options.JOBNAME
        self.MODE = self.crystal.mode
        species = self.crystal.chemical_composition.keys()
        self.NKD = len(species)  # Atom specie number
        self.KD = species  # The atom specie names
        if self.crystal.mode in [SUGGEST, OPTIMIZE]:
            # The total atoms in the supercell
            self.NAT = len(self.crystal.super_cell.atoms)
            return
        # The masses of the specie
        self.MASS = [table.TABLE.loc[x].atomic_weight for x in species]
        # File containing force constants generated by the program alm
        self.FCSXML = f"{self.crystal.options.JOBNAME}{symbols.XML_EXT}"


class Optimize(Base):

    NAME = 'optimize'
    MODES = [OPTIMIZE]

    def setUp(self):
        """
        Set up the key / value pairs for the &optimize-field.
        """
        # The displacement-force data set
        self.DFSET = f"{self.crystal.options.JOBNAME}{symbols.DFSET_EXT}"


class Interaction(Base):

    NAME = 'interaction'
    MODES = [SUGGEST, OPTIMIZE]

    def setUp(self):
        """
        Set up the key / value pairs for the &interaction-field.
        """
        # The order of force constant: 1) harmonic; 2) cubic; ..
        self.NORDER = 1


class Cutoff(Base):

    NAME = 'cutoff'
    SEP = symbols.SPACE
    MODES = [SUGGEST, OPTIMIZE]

    def setUp(self):
        """
        Set up the key / value pairs for the &cutoff-field.
        """
        species = self.crystal.chemical_composition.keys()
        pairs = itertools.combinations_with_replacement(species, 2)
        self.update({f"{x}-{y}": 7.3 for x, y in pairs})


class Cell:

    NAME = 'cell'
    MODES = None

    def __init__(self, crystal):
        """
        :param crystal Crystal: the crystal to get in script from.
        """
        self.crystal = crystal

    @property
    @functools.cache
    def data(self):
        """
        The cell parameters.

        :return 'pd.DataFrame': cell vectors (with a scale factor).
        """
        vecs = self.crystal.primitive().lattice_vectors if self.crystal.mode == PHONONS \
            else self.crystal.super_cell.scaled_lattice_vectors
        vecs = [x * constants.ANG_TO_BOHR for x in vecs]
        data = pd.DataFrame(vecs)
        data.index.name = 1  # scale factor
        return data

    def write(self, fh, header=False, float_format='%0.8f', index=False):
        """
        Write the data to the file handle.

        :param fh '_io.TextIOWrapper': the file handler to write to.
        :param header bool: whether the header is written out.
        :param float_format str: the format of the float values.
        :param index bool: whether the index is written out.
        """
        if self.MODES and self.crystal.mode not in self.MODES:
            return
        fh.write(f"{symbols.AND}{self.NAME}\n")
        if self.data.index.name is not None:
            fh.write(f"{INDENT}{self.data.index.name}\n")
        lines = self.data.to_string(header=header,
                                    float_format=float_format,
                                    index=index)
        for line in lines.split('\n'):
            fh.write(f"{INDENT}{line}\n")
        fh.write(f"{symbols.FORWARDSLASH}\n\n")


class Position(Cell):

    NAME = 'position'
    MODES = [SUGGEST, OPTIMIZE]

    @property
    @functools.cache
    def data(self):
        """
        The position parameters.

        :return 'pd.DataFrame': atom positions (fractional).
        """
        supercell = self.crystal.super_cell
        data = [x.coords_fractional for x in supercell.atoms]
        index = [x.element for x in supercell.atoms]
        data = pd.DataFrame(data, index=index) / supercell.dimensions
        data.sort_values(by=list(data.columns), inplace=True)
        species = self.crystal.chemical_composition.keys()
        id_map = {x: i for i, x in enumerate(species, start=1)}
        data.rename(index=id_map, inplace=True)
        return data.reset_index()


class Kpoint(Cell):

    NAME = 'kpoint'
    MODES = [PHONONS]

    # https://en.wikipedia.org/wiki/Brillouin_zone
    # https://www.businessballs.com/glossaries-and-terminology/greek-alphabet/
    # Center of the Brillouin zone
    GAMMA = pd.Series([0, 0, 0], name='G')
    # Simple cube
    CHI = pd.Series([0.5, 0.5, 0.0], name='X')  # Center of a square face
    RHO = pd.Series([0.5, 0.5, 1], name='R')  # Corner point
    LAMBDA = pd.Series([0.5, 0.5, 0.5], name='L')  # Center of a hexagonal face
    GRID = 51

    @property
    @functools.cache
    def data(self):
        """
        The kspace points.

        :return 'pd.DataFrame': kspace points (with mode).
        """
        lines = [[self.GAMMA, self.CHI], [self.RHO, self.GAMMA, self.LAMBDA]]
        pairs = [[y, z] for x in lines for y, z in zip(x[:-1], x[1:])]
        start = pd.DataFrame([x[0] for x in pairs]).reset_index()
        end = pd.DataFrame([x[1] for x in pairs]).reset_index()
        data = pd.concat([start, end], axis=1)
        data.insert(len(data.columns), 'grid', self.GRID)
        data.index.name = 1  # line mode
        return data
