# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module parsers a Lammps log file and extracts the thermodynamic data.
"""
import io

import pandas as pd

from nemd import lmpin
from nemd import symbols


class Log(lmpin.Base):
    """
    Class to parse LAMMPS log file and extract data.
    """

    def __init__(self, filename, delay=False, unit=lmpin.Base.REAL, **kwargs):
        """
        :param filename str: LAMMPS log file name.

        FIXME: read the unit and timestep from the log file.
        """
        super().__init__(unit=unit, **kwargs)
        self.filename = filename
        self.thermo = None
        self.timestep = self.getTimestep(backend=True)
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up by reading and parsing the log file.
        """
        self.read()
        self.setThermo()

    def read(self, to_skip=('SHAKE', 'Bond', 'Angle', 'WARNING')):
        """
        Read the LAMMPS log file to extract the thermodynamic data.

        :param to_skip tuple: block lines starting with these are to skip.
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
                    if not line.startswith(to_skip):
                        blk.append(line)
                elif line.startswith('Per MPI rank memory allocation'):
                    # Start a new block
                    blk = [fh.readline()]
                # Other information outside the thermo block
                elif line.startswith(self.UNITS):
                    self.unit = line.strip(self.UNITS).strip()
                elif line.startswith(self.TIMESTEP):
                    timestep = float(line.strip(self.TIMESTEP).strip())
                    self.timestep = self.getTimestep(timestep, backend=True)
        if blk:
            # Finishing up the last running thermo block
            data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
            self.thermo = pd.concat((self.thermo, data))

    def setThermo(self):
        """
        Set the thermodynamic data.
        """
        self.thermo = Thermo(self.thermo,
                             timestep=self.timestep,
                             unit=self.unit,
                             options=self.options)


class Thermo(pd.DataFrame):
    """
    Backend thermodynamic data with time in ps, column renaming, start index
    """
    _metadata = ['idx', 'unit', 'timestep']

    def __init__(self, *args, unit=None, timestep=1, options=None, **kwargs):
        """
        :param unit str: the unit of the log file.
        :param timestep float: the timestep.
        :param options `namedtuple`: command line options.
        """
        super().__init__(*args, **kwargs)
        self.unit = unit or Log.REAL
        self.timestep = timestep
        self.options = options
        self.idx = options.last_pct.getSidx(self) if options else 0
        self.setUp()

    def setUp(self,
              key=('Step', 'Temp', 'energy', 'Press', 'Volume'),
              grp=('energy', 'E_pair', 'E_mol', 'TotEng')):
        name = f"{symbols.TIME_LB} ({self.idx})"
        self.index = pd.Index(self.Step * self.timestep, name=name)
        match self.unit:
            case Log.REAL:
                units = ['n', 'K', 'kcal/mol', 'atm', '\u212B^3']
            case Log.METAL:
                units = ['n', 'K', 'eV', 'bar', '\u212B^3']
        units = dict(zip(key, units))
        units.update({x: units[grp[0]] for x in grp[1:]})
        columns = {x: f"{x} ({units[x]})" for x in self.columns}
        self.rename(columns=columns, inplace=True)

    @property
    def range(self):
        """
        return the range of the selected time.

        :return list of floats: the start and end of the selected time range.
        """
        return self.index[self.idx], self.index[-1]
