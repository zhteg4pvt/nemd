# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class processes the executables.
"""
import glob
import os
import re
import subprocess

from nemd import builtinsutils
from nemd import envutils
from nemd import jobutils
from nemd import osutils
from nemd import symbols


class Base(builtinsutils.Object):
    """
    Run subprocess.
    """
    RUN = None
    SEP = ' '
    STD_EXT = '.std'

    def __init__(self, dirname=None, jobname=None, env=None):
        """
        :param dirname str: the subdirectory to run.
        :param jobname str: the jobname.
        :param env dict: the environmental variables.
        """
        self.dirname = dirname or os.curdir
        self.jobname = jobname or envutils.get_jobname() or self.name
        self.env = env
        self.stdout = f"{self.jobname}{self.STD_EXT}"
        self.proc = None

    def run(self):
        """
        Make & change directory, set up, build command, and run command.
        """
        with osutils.chdir(self.dirname), open(self.stdout, 'w') as stdout:
            self.proc = subprocess.run(self.getCmd(),
                                       stdout=stdout,
                                       stderr=subprocess.PIPE,
                                       env=self.env,
                                       text=True,
                                       shell=True)

    def getCmd(self, write_cmd=True):
        """
        Get the command to run the program.

        :param write_cmd bool: whether to write the command to a file
        :return str: the command
        """
        args = self.RUN + self.args if self.RUN else self.args
        cmd = self.SEP.join(args)
        if write_cmd:
            with open(f'{self.jobname}_cmd', 'w') as fh:
                fh.write(cmd)
        return cmd

    @property
    def args(self):
        """
        The args to build the command from.

        :return list: the arguments to build the command
        """
        return ['echo', 'hi']

    @property
    def msg(self):
        """
        The stdout message.

        :return str: the stdout message.
        """
        if os.path.exists(self.stdout):
            with open(self.stdout) as fh:
                return fh.read()
        return ''

    @property
    def err(self):
        """
        Get the error message.

        :return str: the error message.
        """
        if self.proc is None:
            return
        return self.proc.stderr or \
            (self.proc.returncode and 'non-zero return code') or ''


class Process(Base):
    """
    Run subprocess.
    """

    def __init__(self, tokens, *args, **kwargs):
        """
        :param tokens list: the arguments to build the cmd from
        """
        super().__init__(*args, **kwargs)
        self.tokens = tokens

    @property
    def args(self):
        """
        The args to build the command from.

        :return list: the arguments to build the command
        """
        return self.tokens


class Check(Process):
    """
    Subprocess to run check cmd.
    """
    SEP = ' && '


class Submodule(Base):
    """
    Customized with setup and outfiles.
    """
    RUN = [jobutils.NEMD_MODULE]

    def __init__(self, *args, files=None, **kwargs):
        """
        :param files list: input files
        """
        super().__init__(*args, **kwargs)
        self._files = files
        self.start = os.getcwd()

    def getCmd(self, *args, **kwargs):
        """
        See parent.
        """
        self.setUp()
        return super().getCmd(*args, **kwargs)

    def setUp(self):
        """
        Set up the input files.
        """
        pass

    @property
    def outfiles(self):
        """
        Search for output files from the extension, directory, and jobname.

        :return list: the outfiles found.
        :raise FileNotFoundError: no outfiles found.
        """
        patt = os.path.join(self.start, self.dirname,
                            f"{self.jobname}{self.ext}")
        outfiles = glob.glob(os.path.relpath(patt, start=os.curdir))
        if not outfiles:
            raise FileNotFoundError(f"{patt} not found.\n{self.err}")
        return outfiles

    @property
    def ext(self):
        """
        Get the output file extension.

        :return str: the file extension.
        """
        return self.STD_EXT

    @property
    def files(self):
        """
        The input files to run the program with paths modified with the name.

        :return list: input files.
        """
        if not self._files:
            return []
        relpath = os.path.relpath(self.start, start=os.curdir)
        return [
            x if os.path.isabs(x) else os.path.join(relpath, x)
            for x in self._files
        ]


class Lmp(Submodule):
    """
    Class to run lammps simulations.
    """
    RUN = [jobutils.NEMD_MODULE, symbols.LMP]

    def __init__(self, *args, infile=None, **kwargs):
        """
        :param infile str: the in script.
        """
        super().__init__(*args, **kwargs)
        self.infile = infile
        self.logfile = f"{self.jobname}{symbols.LOG_EXT}"

    @property
    def args(self):
        """
        See parent.
        """
        return [jobutils.FLAG_IN, self.infile, jobutils.FLAG_LOG, self.logfile]

    @property
    def err(self, rex=re.compile(f'ERROR: (.*)')):
        """
        The error message.

        :param `re.Pattern` rex: the regular expression to search error.
        :return str: the error message.
        """
        # FIXME: the message by error->one (src/input.cpp:666) not in either stdout or stderr
        err = '\n'.join(x.group(1) for x in rex.finditer(self.msg))
        return err or super().err


class Submodules(Submodule):
    """
    Customized with mode.
    """
    EXTS = {}

    def __init__(self, mode, *args, **kwargs):
        """
        param mode str: the mode.
        """
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.dirname = self.mode

    @property
    def ext(self):
        """
        See parent.
        """
        return self.EXTS.get(self.mode, super().ext)


class Tools(Submodules):
    """
    Class to run the scripts in the 'tools' directory.
    """
    RUN = [jobutils.NEMD_RUN]
    DISPLACE = 'displace'
    EXTRACT = 'extract'
    EXTS = {DISPLACE: "*.lammps"}

    @property
    def args(self):
        """
        See parent.
        """
        scr = envutils.get_data('tools', f'{self.mode}.py', module='alamode')
        match self.mode:
            case self.EXTRACT:
                return [scr, '--LAMMPS'] + self.files
            case self.DISPLACE:
                return [
                    scr, '--prefix', self.jobname, '--mag', '0.01', '--LAMMPS',
                    self.files[0], '-pf', self.files[1]
                ]


class Alamode(Submodules):
    """
    Class to run one alamode binary.
    """
    SUGGEST = symbols.SUGGEST
    OPTIMIZE = symbols.OPTIMIZE
    PHONONS = symbols.PHONONS
    EXTS = {
        SUGGEST: ".pattern_*",
        OPTIMIZE: symbols.XML_EXT,
        PHONONS: ".bands"
    }
    EXES = {SUGGEST: "alm", OPTIMIZE: "alm", PHONONS: "anphon"}

    def __init__(self, crystal, *args, **kwargs):
        """
        :param crystal Crystal: the crystal to get in script from.
        """
        super().__init__(crystal.mode, *args, **kwargs)
        self.crystal = crystal

    def setUp(self):
        """
        See parent.
        """
        self.crystal.write()
        if self.mode == self.SUGGEST:
            return
        filename = os.path.basename(self.files[0])
        if self.mode == self.OPTIMIZE:
            filename = f"{self.jobname}{symbols.DFSET_EXT}"
        osutils.symlink(self.files[0], filename)

    @property
    def args(self):
        """
        See parent.
        """
        return [self.EXES[self.mode], self.crystal.outfile]
