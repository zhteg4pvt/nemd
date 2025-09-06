# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Runs molecule builder, lammps simulation, and log analyzer.
"""
import functools

import numpy as np
import pandas as pd

from nemd import analyzer
from nemd import jobcontrol
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import structure
from nemd import symbols
from nemd import task

FLAG_SUBSTRUCT = parserutils.Bldr.FLAG_SUBSTRUCT


class Reader(logutils.Reader):
    """
    A builder log reader customized for substructure.
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


class Merge(analyzer.Merge):
    """
    Customized for substructures.
    """

    def merge(self):
        """
        Modify the result substructure column so that the name includes the
        structure smiles and geometry type.
        """
        # FIXME: a RuntimeError error without the local import.
        #  RuntimeError: Unable to parallelize execution due to a pickling
        #  error: cannot pickle 'module' object.
        #  Reproduce: nemd_run mb_lmp_log_workflow.py CC -struct_rg CC 0 10 5 -CPU 2
        from nemd import structure
        super().merge()
        substruct = self.data.index.str.split(expand=True)
        has_value = self.data.index[0] != substruct[0]
        smiles = substruct[0][0] if has_value else substruct[0]
        # Set the name of the substructure column (e.g. CC Bond (Angstrom))
        name = f"{smiles} {structure.Mol.MolFromSmiles(smiles).name}"
        if has_value:
            # result.substruct contains the values  (e.g. CC: 2)
            self.data.index = pd.Index([x[1] for x in substruct], name=name)
            logutils.Reader.sort(self.data)
            return
        # result.substruct contains the smiles (e.g. CCCC)
        # Read the reported value from the log (e.g. dihedral angle: 73.50 deg)
        for job in jobutils.Job.search(self.groups[0].jobs[0].fn('')):
            reader = Reader(job.logfile)
            if reader.options.NAME != task.MolBldr.name:
                continue
            values = reader.getSubstruct(smiles)
            self.data.index = pd.Index([values], name=name)


class LmpAgg(task.LmpAgg):
    """
    Customized for substructures.
    """
    Merge = Merge


class Runner(jobcontrol.Runner):
    """
    Customized for molecule builder, lammps runner, and log analyzer tasks.
    """

    def setJobs(self):
        """
        See parent.
        """
        self.add(task.MolBldr)
        self.add(task.Lammps)
        self.add(task.LmpLog)

    @functools.cached_property
    def state(self):
        """
        See parent.
        """
        if not self.options.struct_rg:
            return {}
        smiles, vals = self.options.struct_rg[0], self.options.struct_rg[1:]
        return {
            FLAG_SUBSTRUCT:
            [f"{smiles} {x}" for x in np.arange(*vals)] if vals else [smiles]
        }

    def setAggs(self):
        """
        Aggregate the log analysis jobs.
        """
        self.add(LmpAgg, jobname='lmp_log_agg')
        super().setAggs()


class Action(parserutils.StructAction):
    """
    Customized for a smile string (followed by START END, and STEP).
    """

    def doTyping(self, smiles, *args):
        """
        Check the validity of the smiles string and the range.

        :param smiles str: the smiles str to select a substructure.
        :param args tuple: the start, end, and step to define a range.
        :return tuple: the smiles str, (start, end, and step).
        """
        typed = super().doTyping(smiles, *args[:1])
        if not args:
            return typed
        if len(args) != 3:
            self.error('expected 4 arguments')
        return (*typed, *[parserutils.type_float(x) for x in args[1:]])


class Parser(parserutils.Workflow):
    """
    Customized for substructures.
    """
    WFLAGS = parserutils.Workflow.WFLAGS[1:]

    @classmethod
    def add(cls, parser, **kwargs):
        """
        See parent.
        """
        parser.add_argument('-struct_rg',
                            metavar='SMILES START END STEP',
                            nargs='+',
                            action=Action,
                            help='The range of the degree to scan in degrees.')
        parserutils.MolBldr.add(parser, append=False)
        parserutils.LmpLog.add(parser)
        parser.suppress([
            parserutils.LmpLog.FLAG_LAST_PCT, parserutils.LmpLog.FLAG_SLICE,
            FLAG_SUBSTRUCT
        ])
        return parser


if __name__ == "__main__":
    logutils.Script.run(Runner, Parser(descr=__doc__))
