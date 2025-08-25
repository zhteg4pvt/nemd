# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This driver analyzes a lammps trajectory.
"""
import sys

from nemd import analyzer
from nemd import lmpfull
from nemd import logutils
from nemd import parserutils
from nemd import symbols
from nemd import traj


class Traj(logutils.Base):
    """
    Analyze a trajectory file.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver': Parsed command-line options.
        """
        super().__init__(options=options, **kwargs)
        self.trj = None
        self.rdr = None
        self.gids = None

    def run(self):
        """
        Main method to run.
        """
        self.setStruct()
        self.setAtoms()
        self.setFrames()
        self.analyze()

    def setStruct(self):
        """
        Read data file.
        """
        if not self.options.data_file:
            return
        self.rdr = lmpfull.Reader.read(self.options.data_file)

    def setAtoms(self, selected=slice(None)):
        """
        Set the atom selection.

        :param selected 'slice': the selected elements.
        """
        if not self.rdr:
            return
        if self.options.sel:
            selected = self.rdr.elements.element.isin([self.options.sel])
        self.gids = self.rdr.elements.index[selected].to_numpy()
        self.log(f"{len(self.gids)} atoms selected.")

    def setFrames(self, task=tuple(x.name for x in analyzer.ALL_FRM)):
        """
        Read and log trajectory frames.
        """
        all_frms = set(self.options.task).intersection(task)
        self.trj = traj.Traj(self.options.trj,
                             options=self.options,
                             start=0 if all_frms else None)
        if len(self.trj) == 0:
            self.error(f'{self.options.trj} contains no frames.')
        self.log(f"{len(self.trj)} trajectory frames found.")
        if all_frms:
            self.log(f"{all_frms} analyze all frames {symbols.ELEMENT_OF} "
                     f"[{self.trj.time[0]:.3f}, {self.trj.time[-1]:.3f}] ps")
        lst = set(self.options.task).intersection(parserutils.LmpTraj.LAST_FRM)
        if not lst:
            return
        self.log(f"{lst} analyze frames of last {self.options.last_pct * 100}%"
                 f" {symbols.ELEMENT_OF} [{self.trj.time.start: .3f}, "
                 f"{self.trj.time[-1]: .3f}] ps")

    def analyze(self):
        """
        Run analyzers.
        """
        for Anlz in analyzer.TRAJ:
            if Anlz.name not in self.options.task:
                continue
            anl = Anlz(self.trj,
                       rdr=self.rdr,
                       gids=self.gids,
                       options=self.options,
                       logger=self.logger)
            anl.run()


if __name__ == "__main__":
    options = parserutils.LmpTraj(descr=__doc__).parse_args(sys.argv[1:])
    with logutils.Script(options, file=True) as logger:
        trj = Traj(options, logger=logger)
        trj.run()
