# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module parsers a Lammps log file and extracts the thermodynamic data.
"""
import io

import pandas as pd
import scipy

from nemd import lammpsin
from nemd import symbols

LJ = lammpsin.In.LJ
METAL = lammpsin.In.METAL
REAL = lammpsin.In.REAL


class Thermo(pd.DataFrame):
    """
    Backend thermodynamic data with time in ps, column renaming, start index
    """

    FS = 'fs'
    PS = 'ps'
    N_STEP = 'n'
    KELVIN = 'K'
    ATMOSPHERES = 'atmospheres'
    BARS = 'bars'
    KCAL_MOL = 'kcal/mol'
    EV = 'eV'
    ANGSTROMS = 'Angstroms'
    ANGSTROMS_CUBED = 'Angstroms^3'
    TIME = 'time'
    STEP = 'Step'
    TEMP = 'Temp'
    E_PAIR = 'E_pair'
    E_MOL = 'E_mol'
    TOTENG = 'TotEng'
    PRESS = 'Press'
    VOLUME = 'Volume'
    TIME_UNITS = {REAL: FS, METAL: PS}
    STEP_UNITS = {REAL: N_STEP, METAL: N_STEP}
    TEMP_UNITS = {REAL: KELVIN, METAL: KELVIN}
    ENG_UNITS = {REAL: KCAL_MOL, METAL: EV}
    PRESS_UNITS = {REAL: ATMOSPHERES, METAL: BARS}
    VOLUME_UNITS = {REAL: ANGSTROMS_CUBED, METAL: ANGSTROMS_CUBED}
    THERMO_UNITS = {
        TIME: TIME_UNITS,
        STEP: STEP_UNITS,
        TEMP: TEMP_UNITS,
        E_PAIR: ENG_UNITS,
        E_MOL: ENG_UNITS,
        TOTENG: ENG_UNITS,
        PRESS: PRESS_UNITS,
        VOLUME: VOLUME_UNITS
    }

    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['idx']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, unit=REAL, fac=1, options=None, **kwargs):
        """
        :param unit str: the unit of the log file
        :param fac float: the conversion factor from the step (n) to time (ps)
        :param options `namedtuple`: command line options
        """
        super().__init__(*args, **kwargs)
        self.idx = options.last_pct.getSidx(self) if options else 0
        name = f"{symbols.TIME_LB} ({self.idx})"
        self.index = pd.Index(self[self.STEP] * fac, name=name)
        cmap = {x: f"{x} ({self.THERMO_UNITS[x][unit]})" for x in self.columns}
        self.rename(columns=cmap, inplace=True)

    @property
    def range(self):
        """
        return the range of the selected time.

        :return list of floats: the start and end of the selected time range.
        """
        return self.index[self.idx], self.index[-1]


class Log(lammpsin.In):
    """
    Class to parse LAMMPS log file and extract data.
    """

    DEFAULT_TIMESTEP = {LJ: 0.005, REAL: 1., METAL: 0.001}

    def __init__(self, filename, options=None, delay=False):
        """
        :param filenamestr: LAMMPS log file name
        :param options namedtuple: command line options
        """
        self.filename = filename
        self.options = options
        self.timestep = None
        # FIXME: read the unit from the log file
        self.unit = self.REAL
        self.idx = 0
        self.thermo = pd.DataFrame()
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up by reading and parsing the log file.
        """
        self.read()
        self.setThermo()

    def read(self):
        """
        Read the LAMMPS log file to extract the thermodynamic data.
        """
        with open(self.filename) as fh:
            blk = []
            while line := fh.readline():
                if line.startswith('Loop time of'):
                    # Finishing up previous thermo block
                    data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
                    self.thermo = pd.concat((self.thermo, data))
                    blk = []
                elif blk:
                    # Inside thermo block: skip lines from fix rigid outputs
                    if not line.startswith(('SHAKE', 'Bond', 'Angle')):
                        blk.append(line)
                elif line.startswith('Per MPI rank memory allocation'):
                    # Start a new block
                    blk = [fh.readline()]
                # Other information outside the thermo block
                elif line.startswith(self.UNITS):
                    self.unit = line.strip(self.UNITS).strip()
                elif line.startswith(self.TIMESTEP):
                    self.timestep = float(line.strip(self.TIMESTEP).strip())
        if blk:
            # Finishing up the last running thermo block
            data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
            self.thermo = pd.concat((self.thermo, data))
        if self.timestep is None:
            self.timestep = self.DEFAULT_TIMESTEP[self.unit]

    def setThermo(self):
        """
        Set the thermodynamic data.
        """
        fac = self.timestep * self.time_unit(self.unit) / scipy.constants.pico
        self.thermo = Thermo(self.thermo,
                             fac=fac,
                             unit=self.unit,
                             options=self.options)
