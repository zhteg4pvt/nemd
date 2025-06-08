# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class processes the executables.
"""
import glob
import os
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
    PRE_RUN = None
    SEP = ' '

    def __init__(self, dirname=None, jobname=None):
        """
        :param dirname str: the subdirectory to run.
        :param files list: input files.
        """
        self.dirname = dirname or os.curdir
        self.jobname = jobname or envutils.get_jobname() or self.name
        self.logfile = f'{self.jobname}{symbols.LOG_EXT}'

    def run(self):
        """
        Make & change directory, set up, build command, and run command.

        :return `subprocess.CompletedProcess`: a CompletedProcess instance.
        """
        with osutils.chdir(self.dirname), open(self.logfile, 'w') as fh:
            return subprocess.run(self.getCmd(),
                                  stdout=fh,
                                  stderr=fh,
                                  shell=True)

    def getCmd(self, write_cmd=True):
        """
        Get the command to run the program.

        :param write_cmd bool: whether to write the command to a file
        :return str: the command
        """
        args = [self.PRE_RUN] + self.args if self.PRE_RUN else self.args
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

    SEP = symbols.SEMICOLON


class Submodule(Base):
    """
    Customized with setup and outfiles.
    """
    PRE_RUN = jobutils.NEMD_MODULE

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
        :raise FileNotFoundError: no outfiles found
        """
        pattern = os.path.join(self.start, self.dirname,
                               f"{self.jobname}{self.ext}")
        outfiles = glob.glob(os.path.relpath(pattern, start=os.curdir))
        if not outfiles:
            raise FileNotFoundError(f"{pattern} not found.")
        return outfiles

    @property
    def ext(self):
        """
        Get the output file extension.

        :return str: the file extension.
        """
        return symbols.LOG_EXT

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

    def __init__(self, struct, **kwargs):
        """
        :param struct Struct: the structure to get in script and data file from.
        """
        super().__init__(**kwargs)
        self.struct = struct
        if not self._files:
            return
        basename = os.path.splitext(os.path.basename(self._files[0]))[0]
        self.dirname = f"lammps{basename.removeprefix(self.jobname)}"

    def setUp(self):
        """
        See parent.
        """
        self.struct.writeIn()
        if self.files:
            osutils.symlink(self.files[0], self.struct.datafile)
        else:
            self.struct.writeData()

    @property
    def args(self):
        """
        See parent.
        """
        return [
            symbols.LMP, jobutils.FLAG_IN, self.struct.inscript,
            jobutils.FLAG_SCREEN, symbols.LMP_LOG, jobutils.FLAG_LOG,
            symbols.LMP_LOG
        ]


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
    PRE_RUN = jobutils.NEMD_RUN
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
        return [self.EXES[self.mode], self.crystal.inscript]
