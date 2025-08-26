# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Runs molecule builder, lammps simulation, and log analyzer.
"""
import functools

import numpy as np

from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task

FLAG_SUBSTRUCT = parserutils.Bldr.FLAG_SUBSTRUCT


class LmpAgg(task.LmpAgg):
    """
    Customized for substructures.
    """

    AnalyzerAgg = task.AnalyzerAgg


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
