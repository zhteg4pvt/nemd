# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver calculates dispersion by building crystal, searching symmetries,
displacing atoms, deriving force constants, meshing the K-space, and calculating
vibrational frequencies.
"""
import sys

import numpy as np
import pandas as pd

from nemd import alamode
from nemd import constants
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import plotutils
from nemd import symbols


class Plotter:
    """
    Plot phonon dispersion.
    """

    def __init__(self, infile, options=None):
        """
        :param infile str: the file containing the dispersion data
        :param options 'argparse.Driver':  Parsed command-line options
        """
        self.infile = infile
        self.options = options
        self.data = None
        self.unit = None
        self.points = None
        self.outfile = f"{self.options.JOBNAME}.png"

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.plot()

    def read(self):
        """
        Read the data (e.g., wave vector, frequency, unit, and k-points).
        """
        self.data = pd.read_csv(self.infile,
                                header=None,
                                comment='#',
                                sep=r'\s+',
                                index_col=0)
        self.data.index.name = 'Wave vector'
        with open(self.infile, 'r') as fh:
            lines = [fh.readline().strip('#').strip() for _ in range(3)]
        syms, pnts, cols = [x.split() for x in lines]
        self.unit = cols[-1].strip('[]')
        for idx in reversed((pnts == np.roll(pnts, 1)).nonzero()[0]):
            # Adjacent K points of the same value but different labels
            pnts.pop(idx)
            syms[idx - 1] += f"|{syms.pop(idx)}"
        self.points = pd.Series([float(x) for x in pnts], index=syms)

    def plot(self, unit='THz'):
        """
        Plot the frequency vs wave vector with k-point as the vertical lines.

        :param unit str: the frequency unit.
        """
        if self.unit == 'cm^-1':
            self.data *= constants.CM_INV_THZ
        with plotutils.pyplot(inav=self.options.INTERAC) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            for col in self.data.columns:
                ax.plot(self.data.index, self.data[col], '-b')
            ymin = min([0, self.data.min(axis=None)])
            ymax = self.data.max(axis=None) * 1.05
            ax.vlines(self.points.values[1:-1],
                      ymin,
                      ymax,
                      linestyles='--',
                      color='k')
            ax.set_xlim([self.data.index.min(), self.data.index.max()])
            ax.set_ylim([ymin, ymax])
            ax.set_xticks(self.points.values)
            ax.set_xticklabels(self.points.index)
            ax.set_xlabel(self.data.index.name)
            ax.set_ylabel(f'Frequency ({unit})')
            fig.tight_layout()
            fig.savefig(self.outfile)


class Dispersion(logutils.Base):
    """
    The main class to calculate the phonon dispersion.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver': Parsed command-line options
        """
        super().__init__(options=options, **kwargs)
        self.crystal = None
        self.struct = None
        self.outfile = None

    def run(self):
        """
        Main method to run.
        """
        self.build()
        self.write()
        self.plot()

    def build(self):
        """
        Build the crystal and structure.
        """
        self.crystal = alamode.Crystal.fromDatabase(self.options)
        params = [f'{x:.4f}' for x in self.crystal.lattice_parameters[:3]]
        params += [f'{x:.1f}' for x in self.crystal.lattice_parameters[3:]]
        self.log(f"Supercell lattice parameters: {' '.join(params)}")
        self.struct = alamode.Struct.fromMols([self.crystal.mol],
                                              options=self.options)
        self.struct.write()
        self.log(f"Data file written into {self.struct.outfile}")

    def write(self):
        """
        Write the phonon dispersion.
        """
        self.crystal.mode = alamode.SUGGEST
        kwargs = dict(jobname=self.options.JOBNAME)
        suggest = alamode.exe(self.crystal, **kwargs)
        self.log(f"Suggested displacements are written as {suggest[0]}")
        files = [self.struct.outfile] + suggest
        displace = alamode.exe('displace', files=files, **kwargs)
        self.log(f"Data files with displacements: {displace}")
        dats = []
        for dat in displace:
            dats += alamode.exe(self.struct, files=[dat], **kwargs)
        self.log(f"Trajectory files with forces: {dats}")
        files = [self.struct.outfile] + dats
        extract = alamode.exe('extract', files=files, **kwargs)
        self.crystal.mode = alamode.OPTIMIZE
        optimize = alamode.exe(self.crystal, files=extract, **kwargs)
        self.log(f"Force constants are written as {optimize[0]} ")
        self.crystal.mode = alamode.PHONONS
        self.outfile = alamode.exe(self.crystal, files=optimize, **kwargs)[0]
        self.log(f"Phonon band structure is saved as {self.outfile}")
        jobutils.Job.reg(self.outfile, file=True)

    def plot(self):
        """
        Plot the phonon dispersion.
        """
        plotter = Plotter(self.outfile, options=self.options)
        plotter.run()
        self.log(f'Phonon dispersion figure saved as {plotter.outfile}')
        jobutils.Job.reg(plotter.outfile)


class Parser(parserutils.XtalBldr):
    """
    Customized for single point energy.
    """

    @classmethod
    def add(cls, parser, **kwargs):
        """
        See parent.
        """
        super().add(parser, **kwargs)
        parser.suppress(no_minimize=True, temp=0)


def main(argv):
    parser = Parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        dispersion = Dispersion(options, logger=logger)
        dispersion.run()


if __name__ == "__main__":
    main(sys.argv[1:])
