# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow runs molecule builder, lammps simulation, and log analyzer.
"""
import argparse
import sys

from nemd import analyzer
from nemd import jobcontrol
from nemd import jobutils
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import pd
from nemd import rdkitutils
from nemd import symbols
from nemd import task

FLAG_SUBSTRUCT = parserutils.Bldr.FLAG_SUBSTRUCT


class Reader(logutils.Reader):
    """
    A LAMMPS log reader customized for substructure.
    """

    def getSubstruct(self, smiles):
        """
        Get the value of a substructure from the log file.

        :param smiles str: the substructure smiles
        :return str: the value of the substructure
        """
        for line in self.lines:
            if not line.startswith(smiles):
                continue
            # e.g. 'CCCC dihedral angle: 73.50 deg'
            return line.split(symbols.COLON)[-1].split()[0]


class AnalyzerAgg(analyzer.Agg):
    """
    An analyzer Agg customized for substructure.
    """

    def set(self):
        """
        Modify the result substructure column so that the name includes the
        structure smiles and geometry type.
        """
        super().set()
        if len(self.groups) == 1 and self.groups[0][0].empty:
            return
        substruct = self.data.index.str.split(expand=True)
        has_value = self.data.index[0] != substruct[0]
        smiles = substruct[0][0] if has_value else substruct[0]
        # Set the name of the substructure column (e.g. CC Bond (Angstrom))
        match rdkitutils.MolFromSmiles(smiles).GetNumAtoms():
            case 2:
                name = f"{smiles} Bond (Angstrom)"
            case 3:
                name = f"{smiles} Angle (Degree)"
            case 4:
                name = f"{smiles} Dihedral Angle (Degree)"
        if has_value:
            # result.substruct contains the values  (e.g. CC: 2)
            self.data.index = pd.Index([x[1] for x in substruct], name=name)
            return
        # result.substruct contains the smiles (e.g. CCCC)
        # Read the reported value from the log (e.g. dihedral angle: 73.50 deg)
        for job in jobutils.Job.search(self.groups[0][1][0].fn('')):
            reader = Reader(job.logfile)
            if reader.options.NAME != task.MolBldr.name:
                continue
            values = reader.getSubstruct(smiles)
            self.data.index = pd.Index([values], name=name)
            return
        raise ValueError("Cannot extract the smiles from the log file.")


class LmpAgg(task.LmpAgg):
    """
    See the parent class for details.
    """

    AnalyzerAgg = AnalyzerAgg


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
