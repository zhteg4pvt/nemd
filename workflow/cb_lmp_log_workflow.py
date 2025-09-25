# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Runs crystal builder, lammps simulation, and log analyser.
"""
import functools

import numpy as np

from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task


class Runner(jobcontrol.Runner):
    """
    Customized for crystal builder, lammps runner, and log analyzer tasks.
    """

    def setJobs(self):
        """
        See parent.
        """
        self.add(task.XtalBldr, jobname='crystal_builder')
        self.add(task.Lammps, jobname='lammps_runner')
        self.add(task.LmpLog)

    @functools.cached_property
    def state(self):
        """
        See parent.
        """
        return {
            parserutils.XtalBldr.FLAG_SCALED_FACTOR:
            np.linspace(*self.options.scale_range)
        }

    def setAggs(self):
        """
        Aggregate the log analysis jobs.
        """
        self.add(task.LmpAgg, jobname='lmp_log_agg')
        super().setAggs()


class LinspaceAction(parserutils.Action):
    """
    Action for np.linspace.
    """

    def doTyping(self, *args):
        """
        Check and return the start, stop, and num.

        :return tuple: start, stop, and num.
        """
        if len(args) == 2:
            args += ((args[1] - args[0]) / 0.01 + 1, )
        if len(args) != 3:
            self.error("Please define start, stop, and num.")
        return args[0], args[1], round(args[2])


class Parser(parserutils.Workflow):
    """
    Customized for scaled factors.
    """
    WFLAGS = parserutils.Workflow.WFLAGS[1:]

    @classmethod
    def add(cls, parser, **kwargs):
        """
        See parent.
        """
        parser.add_argument(
            '-scale_range',
            metavar='START STOP NUM',
            default=(0.95, 1.05, 11),
            type=parserutils.Float.typePositive,
            action=LinspaceAction,
            nargs='+',
            help='The scale range on the crystal lattice parameters.')
        parserutils.XtalBldr.add(parser, append=False)
        parserutils.LmpLog.add(parser)


if __name__ == "__main__":
    logutils.Script.run(Runner, Parser(descr=__doc__))
