# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow runs crystal builder, lammps simulation, and log analyser.
"""
import sys

from nemd import jobcontrol
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import task

FLAG_SCALED_RANGE = '-scaled_range'


class Runner(jobcontrol.Runner):

    def setJobs(self):
        """
        Set crystal builder, lammps runner, and log analyzer tasks.
        """
        self.add(task.XtalBldr, jobname='crystal_builder')
        self.add(task.Lammps, jobname='lammps_runner')
        self.add(task.LmpLog)

    def setState(self):
        """
        Set the state keys and values.
        """
        super().setState()
        scaled_range = list(map(str, np.arange(*self.options.scaled_range)))
        self.state[parserutils.XtalBldr.FLAG_SCALED_FACTOR] = scaled_range

    def setAggs(self):
        """
        Set aggregators over all parameter sets.
        """
        self.add(task.LmpAgg, jobname='lmp_log_agg')
        super().setAggs()


class Parser(parserutils.Workflow):

    WFLAGS = parserutils.Workflow.WFLAGS[1:]

    @classmethod
    def add(cls, parser, **kwargs):
        parser.add_argument(
            FLAG_SCALED_RANGE,
            default=(0.95, 1.05, 0.01),  # yapf: disable
            nargs='+',
            metavar=FLAG_SCALED_RANGE[1:].upper(),
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
