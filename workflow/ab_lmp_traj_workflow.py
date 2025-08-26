# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
Run amorphous builder, lammps simulation, and trajectory analyser.
"""
from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task


class Runner(jobcontrol.Runner):
    """
    Customized for amorphous builder, lammps runner, and trajector analyzer tasks.
    """

    def setJobs(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        self.add(task.AmorpBldr, jobname='amorp_bldr')
        self.add(task.Lammps, jobname='lammps')
        self.add(task.LmpTraj, jobname='lmp_traj')

    def setAggs(self):
        """
        Aggregate post analysis jobs.
        """
        self.add(task.LmpAgg, jobname='lmp_traj_agg')
        super().setAggs()


class Parser(parserutils.Workflow):
    """
    Customized for amorphous builder and trajectory analysis.
    """

    @classmethod
    def add(cls, parser, **kwargs):
        parserutils.AmorpBldr.add(parser, append=False)
        parserutils.LmpTraj.add(parser)


if __name__ == "__main__":
    logutils.Script.run(Runner, Parser(descr=__doc__))