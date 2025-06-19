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

    def __init__(self, infile, delay=False, unit=lmpin.Base.REAL, **kwargs):
        """
        :param infile str: LAMMPS log file name.

        FIXME: read the unit and timestep from the log file.
        """
        super().__init__(unit=unit, **kwargs)
        self.infile = infile
        self.thermo = None
        self.timestep = None
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up by reading and parsing the log file.
        """
        self.read()
        self.finalize()

    def read(self, to_skip=('SHAKE', 'Bond', 'Angle', 'WARNING')):
        """
        Read the LAMMPS log file to extract the thermodynamic data.

        :param to_skip tuple: block lines starting with these are to skip.
        """
        with open(self.infile, 'r') as fh:
            while line := fh.readline():
                if line.startswith('Loop time of'):
                    # Finishing up previous thermo block
                    self.setThermo()
                elif self:
                    # Inside thermo block: skip lines from fix rigid outputs
                    if not line.startswith(to_skip):
                        self.append(line)
                elif line.startswith('Per MPI rank memory allocation'):
                    # Start a new block
                    self.append(fh.readline())
                # Other information outside the thermo block
                elif line.startswith(self.UNITS):
                    self.unit = line.strip(self.UNITS).strip()
                elif line.startswith(self.TIMESTEP):
                    self.timestep = float(line.strip(self.TIMESTEP).strip())

    def setThermo(self):
        """
        Concatenate and clear the current thermo block.
        """
        data = pd.read_csv(io.StringIO(''.join(self)), sep=r'\s+')
        self.thermo = pd.concat((self.thermo, data))
        self.clear()

    def finalize(self):
        """
        Finalize.
        """
        if self:
            # Finishing up the last running thermo block
            self.setThermo()
        timestep = self.getTimestep(self.timestep, backend=True)
        self.thermo = Thermo(self.thermo,
                             timestep=timestep,
                             unit=self.unit,
                             options=self.options)


class Thermo(pd.DataFrame):
    """
    Backend thermodynamic data with time in ps, column renaming, start index
    """
    _metadata = ['idx', 'unit', 'timestep']

    def __init__(self,
                 *args,
                 unit=Log.REAL,
                 timestep=1,
                 options=None,
                 **kwargs):
        """
        :param unit str: the unit of the log file.
        :param timestep float: the timestep.
        :param options `namedtuple`: command line options.
        """
        super().__init__(*args, **kwargs)
        self.unit = unit
        self.timestep = timestep
        self.options = options
        self.idx = self.options.last_pct.getSidx(self) if self.options else 0
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        self.set_index('Step', inplace=True)
        self.index *= self.timestep
        self.index.name = f"{symbols.TIME_LB} ({self.idx})"
        energy = ('E_pair', 'E_mol', 'TotEng')
        names = ('Step', 'Temp', energy, 'Press', 'Volume')
        match self.unit:
            case Log.REAL:
                units = ['n', 'K', 'kcal/mol', 'atm', '\u212B^3']
            case Log.METAL:
                units = ['n', 'K', 'eV', 'bar', '\u212B^3']
        units = dict(zip(names, units))
        units.update({x: units[energy] for x in energy})
        self.columns = [f"{x} ({units[x]})" for x in self.columns]

    @property
    def range(self):
        """
        return the range of the selected time.

        :return list of floats: the start and end of the selected time range.
        """
        return self.index[self.idx], self.index[-1]
