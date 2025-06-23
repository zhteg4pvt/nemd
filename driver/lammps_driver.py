# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver runs lammps executable with the given input file.
"""
import functools
import os
import re
import sys

from nemd import jobutils
from nemd import lmpin
from nemd import logutils
from nemd import parserutils
from nemd import process


class Lammps(logutils.Base, process.Lmp):
    """
    Main class to run the lammps executable.
    """
    KEY_RE = r"\b{key}\s+(\S*)\s+(\S*)\s+(\S*)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        process.Lmp.__init__(self,
                             infile=os.path.basename(self.options.inscript),
                             jobname=f"{self.options.JOBNAME}_lmp")
        self.contents = None
        self.path = os.path.dirname(self.options.inscript)
        jobutils.Job.reg(self.logfile, file=True)
        if self.options.CPU is None:
            return
        self.env = {'OMP_NUM_THREADS': str(self.options.CPU[0]), **os.environ}

    def setUp(self):
        """
        Run lammps executable with the given input file and output file.
        """
        if not self.path:
            return
        # The input script is not in the current directory
        self.read()
        self.setPair()
        self.addPath()
        self.writeIn()

    def read(self):
        """
        Read the contents of the input script.
        """
        with open(self.options.inscript, 'r') as fh:
            self.contents = fh.read()

    def setPair(self,
                style=re.compile(KEY_RE.format(key=lmpin.Script.PAIR_STYLE)),
                coeff=re.compile(KEY_RE.format(key=lmpin.Script.PAIR_COEFF))):
        """
        Set the pair coefficients with the input script path.
        """
        match = style.search(self.contents)
        if match and match.group(1) in ['sw']:
            # e.g. 'pair_style sw\n'
            # e.g. 'pair_coeff * * Si.sw Si\n'
            self.addPath(coeff, grp_id=3)

    def addPath(self,
                rex=re.compile(KEY_RE.format(key=lmpin.Script.READ_DATA)),
                grp_id=1):
        """
        Add path to the filename in the contents.

        :param rex str: the regular expression to search for the filename
        :param grp_id int: the group id of the filename in the match
        """
        match = rex.search(self.contents)
        if not match or os.path.isfile(match.group(grp_id)):
            return
        # File not found in the current directory
        pathname = os.path.join(self.path, match.group(grp_id))
        if not os.path.isfile(pathname):
            return
        # Missing file sits with the in script
        sidx, eidx = match.span(grp_id)
        self.contents = self.contents[:sidx] + pathname + self.contents[eidx:]

    def writeIn(self):
        """
        Write the contents of the input script to a file.
        """
        with open(os.path.basename(self.options.inscript), 'w') as fh:
            fh.writelines(self.contents)

    def run(self, rex=re.compile(f'ERROR: (.*)')):
        """
        See parent.
        """
        self.log('Running lammps simulations...')
        if not super().run().returncode:
            return
        with open(self.logfile, 'r') as fh:
            self.error('\n'.join(x.group(1) for x in rex.finditer(fh.read())))

    @functools.cached_property
    def args(self):
        """
        See parent.
        """
        args = super().args
        proc = process.Process(self.RUN + ['-h', '|', 'grep', 'GPU'],
                               jobname=f'{self.options.JOBNAME}_gpu')
        if not proc.run().returncode:
            args += ['-sf', 'gpu', '-pk', 'gpu', '1']
        return args


def main(argv):
    parser = parserutils.Lammps(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        lmp = Lammps(options=options, logger=logger)
        lmp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
