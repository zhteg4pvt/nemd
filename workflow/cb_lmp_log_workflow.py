# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Runs crystal builder, lammps simulation, and log analyser.
"""
import sys

from nemd import jobcontrol
from nemd import logutils
from nemd import np
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

    def setState(self):
        """
        Set the scale factor flag which compress or expand the supercell.
        """
        super().setState()
        self.state[parserutils.XtalBldr.FLAG_SCALED_FACTOR] = np.arange(
            *self.options.scale_range)

    def setAggs(self):
        """
        Aggregate the log analysis jobs.
        """
        self.add(task.LmpAgg, jobname='lmp_log_agg')
        super().setAggs()


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
            default=(0.95, 1.05, 0.01),
            nargs=3,
            metavar='FLOAT',
            type=parserutils.type_positive_float,
            help='The range of scale factors on the crystal lattice parameters.'
        )
        parserutils.XtalBldr.add(parser, append=False)
        parserutils.LmpLog.add(parser)


def main(argv):
    parser = Parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        runner = Runner(options, argv, logger=logger)
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
