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
        amorp_bldr = self.setOpr(task.AmorpBldr, jobname='amorp_bldr')
        lammps_runner = self.setOpr(task.Lammps, jobname='lammps_runner')
        self.setPreAfter(amorp_bldr, lammps_runner)
        lmp_traj = self.setOpr(task.LmpTraj, jobname='lmp_traj')
        self.setPreAfter(lammps_runner, lmp_traj)

    def setAggJobs(self):
        """
        Aggregate post analysis jobs.
        """
        self.setAgg(task.LmpTraj)
        super().setAggJobs()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.WorkflowParser(__file__, descr=__doc__)
    task.AmorpBldrJob.add_arguments(parser)
    task.TrajJob.add_arguments(parser)
    parser.add_job_arguments()
    parser.add_workflow_arguments()
    return parser


def main(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        runner = Runner(options, argv, logger=logger)
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
