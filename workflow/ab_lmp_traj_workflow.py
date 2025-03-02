# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow runs amorphous builder, lammps simulation, and trajectory analyser.
"""
import sys

from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task


class Runner(jobcontrol.Runner):

    def setJobs(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        self.add(task.AmorpBldrJob, name='amorp_bldr')
        self.add(task.LammpsJob, name='lammps')
        self.add(task.TrajJob, name='lmp_traj')

    def setAggs(self):
        """
        Aggregate post analysis jobs.
        """
        self.add(task.LmpLogAgg, name='lmp_traj')
        super().setAggs()


class Parser(parserutils.Workflow):

    @classmethod
    def add(cls, parser, **kwargs):
        parserutils.AmorpBldr.add(parser, append=False)
        parserutils.LmpTraj.add(parser)


def main(argv):
    parser = Parser(__file__, descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        runner = Runner(options, argv, logger=logger)
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
