# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow runs molecule builder, lammps simulation, and log analyzer.
"""
import sys

from nemd import analyzer
from nemd import jobcontrol
from nemd import jobutils
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import rdkitutils
from nemd import symbols
from nemd import task


class LogReader(logutils.Reader):
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

    def setResult(self):
        """
        Modify the result substructure column so that the name includes the
        structure smiles and geometry type.
        """
        super().setResult()
        if len(self.groups) == 1 and self.groups[0][0].empty:
            return
        smiles = self.extractSmiles()
        # Set the name of the substructure column (e.g. CC Bond (Angstrom))
        match rdkitutils.MolFromSmiles(smiles).GetNumAtoms():
            case 2:
                name = f"{smiles} Bond (Angstrom)"
            case 3:
                name = f"{smiles} Angle (Degree)"
            case 4:
                name = f"{smiles} Dihedral Angle (Degree)"
        self.result.rename(columns={self.result.substruct.name: name},
                           inplace=True)

    def extractSmiles(self):
        """
        Extract the smiles with the substructure set with values only.

        :return str: the smiles str of the substructure
        :raises ValueError: if the smiles cannot be extracted from the log file
        """
        substruct = self.result.substruct.str.split(expand=True)
        smiles = substruct.iloc[0, 0]
        if substruct.shape[1] > 1:
            # result.substruct contains the values  (e.g. CC: 2)
            self.result.substruct = substruct.iloc[:, 1]
            return smiles
        # result.substruct contains the smiles (e.g. CCCC)
        # Read the reported value from the log (e.g. dihedral angle: 73.50 deg)
        job = self.groups[0][1][0]
        for logfile in job.doc[jobutils.LOGFILE].values():
            reader = LogReader(job.fn(logfile))
            if reader.options.NAME != task.MolBldrJob.name:
                continue
            self.result.substruct = reader.getSubstruct(smiles)
            return smiles
        raise ValueError("Cannot extract the smiles from the log file.")


class LogAgg(task.LogAgg):
    """
    See the parent class for details.
    """

    AnalyzerAgg = AnalyzerAgg


class LmpLog(task.LmpLog):
    """
    See the parent class for details.
    """

    AggClass = LogAgg


class Runner(jobcontrol.Runner):
    """
    Set up the workflow with parameterized regular jobs and aggregator jobs.
    """

    def setJob(self):
        """
        Set molecule builder, lammps runner, and log analyzer tasks.
        """
        mol_bldr = self.setOpr(task.MolBldr)
        lmp_runner = self.setOpr(task.Lammps)
        self.setPreAfter(mol_bldr, lmp_runner)
        lmp_log = self.setOpr(LmpLog)
        self.setPreAfter(lmp_runner, lmp_log)

    def setState(self):
        """
        Set the substruct flag which measures or sets certain geometry.
        """
        super().setState()
        if self.options.struct_rg is None:
            return
        if self.options.struct_rg[1] is None:
            self.state[parserutils.FLAG_SUBSTRUCT] = self.options.struct_rg[:1]
            return
        range_values = map(str, np.arange(*self.options.struct_rg[1:]))
        structs = [f"{self.options.struct_rg[0]} {x}" for x in range_values]
        self.state[parserutils.FLAG_SUBSTRUCT] = structs

    def setAggJobs(self):
        """
        Aggregate the log analysis jobs.
        """
        self.setAgg(LmpLog)
        super().setAggJobs()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser': argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.WorkflowParser(__file__, descr=__doc__)
    parser.add_argument('-struct_rg',
                        metavar='SMILES START END STEP',
                        nargs='+',
                        action=parserutils.StructRg,
                        help='The range of the degree to scan in degrees.')
    task.MolBldrJob.add_arguments(parser)
    task.LogJob.add_arguments(parser)
    parser.suppress([task.LogJob.FLAG_LAST_PCT, task.LogJob.FLAG_SLICE])
    parser.add_job_arguments()
    parser.add_workflow_arguments()
    parser.suppress([parserutils.FLAG_STATE_NUM, parserutils.FLAG_SUBSTRUCT])
    return parser


def main(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        runner = Runner(options, argv, logger=logger)
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
