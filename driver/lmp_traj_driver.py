# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This driver analyzes the trajectory from previous molecular dynamics simulations.
"""
import sys

from nemd import analyzer
from nemd import lammpsdata
from nemd import logutils
from nemd import parserutils
from nemd import symbols
from nemd import traj


class Traj(logutils.Base):
    """
    Analyze a dump custom file.
    """
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.Driver': Parsed command-line options
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.options = options
        self.trj = None
        self.rdr = None
        self.gids = None
        self.tasks = [x for x in self.options.task if x in analyzer.ALL_FRM]
        self.start = 0 if self.tasks else None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setAtoms()
        self.setFrames()
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return
        self.rdr = lammpsdata.read(self.options.data_file)

    def setAtoms(self):
        """
        set the atom selection for analysis.
        """
        if not self.rdr:
            return
        if self.options.sel is None:
            self.gids = self.rdr.elements.index.tolist()
            self.log(f"{len(self.gids)} atoms selected.")
            return
        selected = self.rdr.elements.element.isin([self.options.sel])
        self.gids = self.rdr.elements.index[selected].tolist()
        self.log(f"{len(self.gids)} atoms selected.")

    def setFrames(self):
        """
        Read and set trajectory frames.
        """
        self.trj = traj.Traj(self.options.trj,
                             options=self.options,
                             start=self.start)
        if len(self.trj) == 0:
            self.error(f'{self.options.trj} contains no frames.')
        # Report the number of frames, (starting time), and ending time
        self.log(f"{len(self.trj)} trajectory frames found.")
        if self.tasks:
            self.log(
                f"{', '.join(self.tasks)} analyze all frames and save per "
                f"frame results {symbols.ELEMENT_OF} [{self.trj.time[0]:.3f}, "
                f"{self.trj.time[-1]:.3f}] ps")
        lf_tasks = [
            x for x in self.options.task if x in parserutils.LmpTraj.LAST_FRM
        ]
        if lf_tasks:
            label, unit, _ = analyzer.Job.parse(self.trj.time.name)
            self.log(
                f"{', '.join(lf_tasks)} average results from last "
                f"{self.options.last_pct * 100}% frames {symbols.ELEMENT_OF} "
                f"[{self.trj.time.start: .3f}, {self.trj.time[-1]: .3f}] ps")

    def analyze(self):
        """
        Run analyzers.
        """
        for name in self.options.task:
            Analyzer = analyzer.ANLZ[name]
            anl = Analyzer(self.trj,
                           rdr=self.rdr,
                           gids=self.gids,
                           options=self.options,
                           logger=self.logger)
            anl.run()


def main(argv):
    parser = parserutils.LmpTraj(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        cdump = Traj(options, logger=logger)
        cdump.run()


if __name__ == "__main__":
    main(sys.argv[1:])
