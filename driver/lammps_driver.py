# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver runs lammps executable with the given input file.
"""
import os
import re
import subprocess
import sys

from nemd import jobutils
from nemd import lammpsfix
from nemd import lammpsin
from nemd import logutils
from nemd import symbols
from nemd import task


class Lammps(logutils.Base):
    """
    Main class to run the lammps executable.
    """
    FLAG_IN = '-in'
    KEYWORD_RE = r"\b{key}\s+(\S*)\s+(\S*)\s+(\S*)"
    READ_DATA_RE = KEYWORD_RE.format(key=lammpsfix.READ_DATA)
    PAIR_STYLE_RE = KEYWORD_RE.format(key=lammpsin.In.PAIR_STYLE)
    PAIR_COEFF_RE = KEYWORD_RE.format(key=lammpsin.In.PAIR_COEFF)

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param logger 'logging.Logger':  Logger for logging messages.
        """
        super().__init__(logger=logger)
        self.options = options
        self.args = None
        self.contents = None
        self.path = os.path.dirname(self.options.inscript)
        self.outfile = self.options.log
        self.env = None
        if self.outfile is None:
            self.outfile = f"{self.options.jobname}_{symbols.LMP_LOG}"
        jobutils.add_outfile(self.outfile, file=True)

    def run(self):
        """
        Run lammps executable with the given input file and output file.
        """
        if self.path:
            # The input script is not in the current directory
            self.read()
            self.setPair()
            self.addPath()
            self.writeIn()
        self.setArgs()
        self.setGpu()
        self.setCpu()
        self.execute()

    def read(self):
        """
        Read the contents of the input script.
        """
        with open(self.options.inscript, 'r') as fh:
            self.contents = fh.read()

    def setPair(self):
        """
        Set the pair coefficients with the input script path.
        """
        match = re.search(self.PAIR_STYLE_RE, self.contents)
        if not match or match.group(1) not in ['sw']:
            return
        # e.g. 'pair_style sw\n'
        # e.g. 'pair_coeff * * Si.sw Si\n'
        self.addPath(self.PAIR_COEFF_RE, grp_id=3)

    def addPath(self, rex=READ_DATA_RE, grp_id=1):
        """
        Add path to the filename in the contents.

        :param rex str: the regular expression to search for the filename
        :param grp_id int: the group id of the filename in the match
        """
        match = re.search(rex, self.contents)
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

    def setArgs(self):
        """
        Set the arguments for the lammps executable.
        """
        self.args = [jobutils.FLAG_IN, os.path.basename(self.options.inscript)]
        self.args += [task.LammpsJob.FLAG_LOG, self.outfile]
        self.args += [jobutils.FLAG_SCREEN, self.options.screen]

    def setGpu(self):
        """
        Set the GPU arguments.
        """
        lmp = subprocess.run(
            f'{jobutils.RUN_MODULE} {symbols.LMP} -h | grep GPU',
            capture_output=True,
            shell=True)
        if not lmp.stdout:
            return
        self.args += ['-sf', 'gpu', '-pk', 'gpu', '1']

    def setCpu(self):
        """
        Set the omp environment variables.
        """
        if self.options.cpu is None:
            return
        self.env = {'OMP_NUM_THREADS': str(self.options.cpu[0]), **os.environ}

    def execute(self):
        """
        Run lammps executable with the given input file and output file.
        """
        self.log('Running lammps simulations...')
        process = subprocess.Popen([jobutils.RUN_MODULE, symbols.LMP] +
                                   self.args,
                                   env=self.env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate()
        self.log(stdout)
        self.log(stderr)

        # lmp_serial -in amorp_bldr.in
        # env OMP_NUM_THREADS=4 lmp_serial -sf omp -in amorp_bldr.in
        # mpirun -np 4 lmp_mpi -in amorp_bldr.in

        # https://docs.lammps.org/jump.html
        # The SELF option is not guaranteed to work when the current input
        # script is being read through stdin (standard input), e.g.
        # with logutils.redirect(logger=logger):
        #     # "[xxx.local:xxx] shmem: mmap: an error occurred while determining
        #     # whether or not xxx could be created." while lammps.lammps()
        #     lmp = lammps.lammps(cmdargs=self.args)
        # with open(self.options.inscript, 'r') as fh:
        #     cmds = fh.readlines()
        #     rdata = f'{lammpsin.In.READ_DATA} {self.options.data_file}'
        #     cmds = [rdata if self.READ_DATA.match(x) else x for x in cmds]
        #     breakpoint()
        #     lmp.commands_list(cmds)
        # lmp.close()


def main(argv):
    parser = task.LammpsJob.get_parser()
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        lmp = Lammps(options, logger=logger)
        lmp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
