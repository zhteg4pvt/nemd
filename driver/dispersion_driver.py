# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver calculates dispersion by building crystal, searching symmetries,
displacing atoms, deriving force constants, meshing the K-space, and calculating
vibrational frequencies.
"""
import sys

from nemd import alamode
from nemd import constants
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import pd
from nemd import plotutils
from nemd import symbols


class Plotter:

    THZ = 'THz'
    PNG_EXT = '.png'

    def __init__(self, filename, options=None, unit=THZ):
        """
        :param filename str: the file containing the dispersion data
        :param options 'argparse.Driver':  Parsed command-line options
        :param unit str: the unit of the y data (either THz or cm^-1)
        """
        self.filename = filename
        self.options = options
        self.unit = unit
        self.data = None
        self.ylim = None
        self.xticks = None
        self.xlabels = None
        self.outfile = self.options.JOBNAME + self.PNG_EXT

    def run(self):
        """
        Main method to run.
        """
        self.readData()
        self.setKpoints()
        self.setFigure()

    def readData(self):
        """
        Read the data from the file with unit conversion and range set.
        """
        data = pd.read_csv(self.filename, header=None, skiprows=3, sep=r'\s+')
        self.data = data.set_index(0)
        if self.unit == self.THZ:
            self.data *= constants.CM_INV_THZ
        self.ylim = [
            min([0, self.data.min(axis=None)]),
            self.data.max(axis=None) * 1.05
        ]

    def setKpoints(self):
        """
        Set the point values and labels.
        """
        with open(self.filename, 'r') as fh:
            header = [fh.readline().strip() for _ in range(2)]
        symbols, pnts = [x.strip('#').split() for x in header]
        pnts = [float(x) for x in pnts]
        # Adjacent K points may have the same value
        same_ids = [i for i in range(1, len(pnts)) if pnts[i - 1] == pnts[i]]
        idxs = [x for x in range(len(pnts)) if x not in same_ids]
        self.xticks = [pnts[i] for i in idxs]
        self.xlabels = [symbols[i] for i in idxs]
        for idx in same_ids:
            # Adjacent K points with the same value combine the labels
            self.xlabels[idx - 1] = '|'.join([symbols[idx - 1], symbols[idx]])

    def setFigure(self):
        """
        Plot the frequency vs wave vector with k-point vertical lines.
        """
        with plotutils.pyplot(inav=self.options.INTERAC) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            for col in self.data.columns:
                ax.plot(self.data.index, self.data[col], '-b')
            verticals = self.xticks[1:-1]
            ax.vlines(verticals, *self.ylim, linestyles='--', color='k')
            ax.set_xlim([self.data.index.min(), self.data.index.max()])
            ax.set_ylim(self.ylim)
            ax.set_xticks(self.xticks)
            ax.set_xticklabels(self.xlabels)
            ax.set_xlabel('Wave vector')
            ax.set_ylabel(f'Frequency ({self.unit})')
            fig.tight_layout()
            fig.savefig(self.outfile)


class Dispersion(logutils.Base):
    """
    The main class to calculate the phonon dispersion.
    """

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.Driver': Parsed command-line options
        :param logger 'logging.Logger':  Logger for logging messages.
        """
        super().__init__(logger=logger)
        self.options = options
        self.crystal = None
        self.struct = None
        self.outfile = None

    def run(self):
        """
        Main method to run.
        """
        self.buildCell()
        self.writeData()
        self.write()
        self.plot()

    def buildCell(self):
        """
        Build the supercell based on the unit cell.
        """
        self.crystal = alamode.Crystal.from_database(options=self.options)
        params = map("{:.4}".format, self.crystal.lattice_parameters)
        self.log(
            f"The lattice parameters of the supper cell is {' '.join(params)}")

    def writeData(self):
        """
        Write the LAMMPS data file with the original structure and in script to
        calculate the force.
        """
        mols = [self.crystal.mol]
        self.struct = alamode.Struct.fromMols(mols, options=self.options)
        self.struct.writeData()
        self.log(f"LAMMPS data file written as {self.struct.datafile}")

    def write(self):
        """
        Write the phonon dispersion.
        """
        self.crystal.mode = alamode.SUGGEST
        kwargs = dict(jobname=self.options.JOBNAME)
        suggest = alamode.exe(self.crystal, **kwargs)
        self.log(f"Suggested displacements are written as {suggest[0]}")
        files = [self.struct.datafile] + suggest
        displace = alamode.exe('displace', files=files, **kwargs)
        self.log(f"Data files with displacements: {', '.join(displace)}")
        dats = []
        for dat in displace:
            dats += alamode.exe(self.struct, files=[dat], **kwargs)
        self.log(f"Trajectory files with forces: {' '.join(dats)}")
        files = [self.struct.datafile] + dats
        extract = alamode.exe('extract', files=files, **kwargs)
        self.crystal.mode = alamode.OPTIMIZE
        optimize = alamode.exe(self.crystal, files=extract, **kwargs)
        self.log(f"Force constants are written as {optimize[0]} ")
        self.crystal.mode = alamode.PHONONS
        self.outfile = alamode.exe(self.crystal, files=optimize, **kwargs)[0]
        self.log(f"Phonon band structure is saved as {self.outfile}")
        jobutils.add_outfile(self.outfile, file=True)

    def plot(self):
        """
        Plot the phonon dispersion.
        """
        plotter = Plotter(self.outfile, options=self.options)
        plotter.run()
        self.log(f'Phonon dispersion figure saved as {plotter.outfile}')
        jobutils.add_outfile(plotter.outfile)


class Parser(parserutils.XtalBldr):

    @classmethod
    def add(cls, parser, **kwargs):
        super().add(parser, **kwargs)
        parser.suppress(no_minimize=True, temp=0)
        parser.set_defaults(force_field=[symbols.SW])


def main(argv):
    parser = Parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        dispersion = Dispersion(options, logger=logger)
        dispersion.run()


if __name__ == "__main__":
    main(sys.argv[1:])
