# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow runs molecule builder, lammps simulation, and log analyzer.
"""
import argparse
import sys

from nemd import jobcontrol
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import task

FLAG_SUBSTRUCT = parserutils.Bldr.FLAG_SUBSTRUCT


class LmpAgg(task.LmpAgg):
    """
    See the parent class for details.
    """

    AnalyzerAgg = task.AnalyzerAgg


class Runner(jobcontrol.Runner):
    """
    Set up the workflow with parameterized regular jobs and aggregator jobs.
    """

    def setJobs(self):
        """
        Set molecule builder, lammps runner, and log analyzer tasks.
        """
        self.add(task.MolBldr)
        self.add(task.Lammps)
        self.add(task.LmpLog)

    def setState(self):
        """
        Set the substruct flag which measures or sets certain geometry.
        """
        super().setState()
        if self.options.struct_rg is None:
            return
        if self.options.struct_rg[1] is None:
            self.state[FLAG_SUBSTRUCT] = self.options.struct_rg[:1]
            return
        range_values = map(str, np.arange(*self.options.struct_rg[1:]))
        structs = [f"{self.options.struct_rg[0]} {x}" for x in range_values]
        self.state[FLAG_SUBSTRUCT] = structs

    def setAggs(self):
        """
        Aggregate the log analysis jobs.
        """
        self.add(LmpAgg, jobname='lmp_log_agg')
        super().setAggs()


class StructRgAction(parserutils.StructAction):
    """
    Action for argparse that allows a mandatory smile str followed by optional
    START END, and STEP values of type float, float, and int, respectively.
    """

    def doTyping(self, smiles, start=None, end=None, step=None):
        """
        Check the validity of the smiles string and the range.

        :param smiles str: the smiles str to select a substructure.
        :param start str: the start to define a range.
        :param end str: the end to define a range.
        :param step str: the step to define a range.
        :return str, float, float, float: the smiles str, start, end, and step.
        """
        _, start = super().doTyping(smiles, start)
        if start is None:
            return [smiles, None]
        if end is None or step is None:
            raise argparse.ArgumentTypeError(
                "start, end, and step partially provided.")
        return [
            smiles, start,
            parserutils.type_float(end),
            parserutils.type_float(step)
        ]


class Parser(parserutils.Workflow):

    WFLAGS = parserutils.Workflow.WFLAGS[1:]

    @classmethod
    def add(cls, parser, **kwargs):
        parser.add_argument('-struct_rg',
                            metavar='SMILES START END STEP',
                            nargs='+',
                            action=StructRgAction,
                            help='The range of the degree to scan in degrees.')
        parserutils.MolBldr.add(parser, append=False)
        parserutils.LmpLog.add(parser)
        parser.suppress([
            parserutils.LmpLog.FLAG_LAST_PCT, parserutils.LmpLog.FLAG_SLICE,
            FLAG_SUBSTRUCT
        ])
        return parser


def main(argv):
    parser = Parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        runner = Runner(options, argv, logger=logger)
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
