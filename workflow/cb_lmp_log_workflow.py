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

    def setJob(self):
        """
        Set crystal builder, lammps runner, and log analyzer tasks.
        """
        crystal_builder = self.setOpr(task.XtalBldr, name='crystal_builder')
        lammps_runner = self.setOpr(task.Lammps, name='lammps_runner')
        self.setPreAfter(crystal_builder, lammps_runner)
        lmp_log = self.setOpr(task.LmpLog)
        self.setPreAfter(lammps_runner, lmp_log)

    def setState(self):
        """
        Set the state keys and values.
        """
        super().setState()
        scaled_range = list(map(str, np.arange(*self.options.scaled_range)))
        self.state[parserutils.FLAG_SCALED_FACTOR] = scaled_range

    def setAggJobs(self):
        """
        Aggregate post analysis jobs.
        """
        self.setAgg(task.LmpLog)
        super().setAggJobs()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.DriverParser': argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.WorkflowParser(__file__, descr=__doc__)
    parser.add_argument(
        FLAG_SCALED_RANGE,
        default=(0.95, 1.05, 0.01),  # yapf: disable
        nargs='+',
        metavar=FLAG_SCALED_RANGE[1:].upper(),
        type=parserutils.type_positive_float,
        help='The range of scale factors on the crystal lattice parameters.')
    task.XtalBldrJob.add_arguments(parser)
    task.LogJob.add_arguments(parser)
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
