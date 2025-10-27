# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver runs lammps executable using the in script.
"""
import functools
import os
import pathlib
import re

from nemd import jobutils
from nemd import lmpin
from nemd import logutils
from nemd import parserutils
from nemd import process


class Lammps(logutils.Base, process.Lmp):
    """
    Main class to run the lammps executable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        process.Lmp.__init__(self,
                             infile=os.path.basename(self.options.inscript),
                             jobname=f"{self.options.JOBNAME}_lmp")
        jobutils.Job.reg(self.logfile, file=True)
        self.env = {'OMP_NUM_THREADS': str(self.options.CPU[0]), **os.environ}

    def setUp(self, data=re.compile(rf"{lmpin.Script.READ_DATA}\s+(\S+)")):
        """
        See parent.

        :param `re.Pattern` data: the regular expression to search read_data.
        """
        if self.parent.samefile(os.curdir):
            return
        self.setPair()
        self.addPath(data)
        self.write()

    @functools.cached_property
    def parent(self):
        """
        Return the in script parent (dirname).

        :return 'pathlib.PosixPath': input script path.
        """
        parent = pathlib.Path(self.options.inscript).parent
        return parent.relative_to(os.curdir) if parent.is_relative_to(os.curdir) \
            else parent

    def setPair(
        self,
        style=re.compile(rf"{lmpin.Script.PAIR_STYLE}\s+(\w+)"),
        coeff=re.compile(rf"{lmpin.Script.PAIR_COEFF}\s+\S+\s+\S+\s+(\S+)")):
        """
        Add path to the filename in pair coefficients command.

        :param `re.Pattern` style: the regular expression to search pair_style.
        :param `re.Pattern` coeff: the regular expression to search pair_coeff.
        """
        match = style.search(self.cont)
        if match and match.group(1) in ['sw']:
            # e.g. 'pair_style sw\n'
            # e.g. 'pair_coeff * * Si.sw Si\n'
            self.addPath(coeff)

    @functools.cached_property
    def cont(self):
        """
        Read the cont of the input script.

        return str: the input script cont.
        """
        with open(self.options.inscript, 'r') as fh:
            return fh.read()

    def addPath(self, rex):
        """
        Add path to the filename in the contents.

        :param `re.Pattern` rex: the regular expression to search filename.
        """
        match = rex.search(self.cont)
        if not match or os.path.isfile(match.group(1)):
            return
        pathname = self.parent / match.group(1)
        sid, eid = match.span(1)
        self.cont = self.cont[:sid] + str(pathname) + self.cont[eid:]

    def write(self):
        """
        Write the contents.
        """
        with open(self.infile, 'w') as fh:
            fh.write(self.cont)

    def run(self):
        """
        See parent.
        """
        self.log('Running lammps simulations...')
        super().run()
        if self.proc.returncode:
            self.error(self.err)

    @functools.cached_property
    def args(self):
        """
        See parent.
        """
        proc = process.Process(self.RUN + ['-h', '|', 'grep', 'GPU'],
                               jobname=f'{self.options.JOBNAME}_gpu')
        proc.run()
        args = [] if proc.proc.returncode else [
            '-sf', 'gpu', '-pk', 'gpu', '1'
        ]
        return super().args + args


if __name__ == "__main__":
    logutils.Script.run(Lammps, parserutils.Lammps(descr=__doc__))
