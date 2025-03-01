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

    def setJob(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        amorp_bldr = self.setOpr(task.AmorpBldrJob, name='amorp_bldr')
        lammps_runner = self.setOpr(task.LammpsJob, name='lammps')
        lmp_traj = self.setOpr(task.TrajJob, name='lmp_traj')
        self.setPreAfter(amorp_bldr, lammps_runner)
        self.setPreAfter(lammps_runner, lmp_traj)

    def setAggJobs(self):
        """
        Aggregate post analysis jobs.
        """
        self.setOpr(task.LmpLogAgg, name='lmp_traj')
        super().setAggJobs()


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
