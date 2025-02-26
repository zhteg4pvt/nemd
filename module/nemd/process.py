# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class processes the executables.
"""
import functools
import glob
import os
import subprocess

from nemd import envutils
from nemd import jobutils
from nemd import lammpsfix
from nemd import osutils
from nemd import symbols


class Base:
    """
    Base class to build command, execute subprocess, and search for output files
    """
    NAME = None
    PRE_RUN = None
    SEP = ' '
    EXT = symbols.LOG_EXT
    EXTS = {}

    def __init__(self, tokens=None, name=None, jobname=None, files=None):
        """
        :param tokens list: the arguments to build the cmd from
        :param name str: the subdirectory name
        :param jobname str: name output files based on this
        :param files list: input files
        """
        self.tokens = tokens
        self.name = name
        self.jobname = jobname
        self._files = files
        if self.name is None:
            self.name = os.curdir
        self.logfile = f'{self.jobname}{symbols.LOG_EXT}' if self.jobname else 'log'

    def run(self):
        """
        Make & change directory, set up, build command, and run command.

        :return `subprocess.CompletedProcess`: a CompletedProcess instance.
        """
        with osutils.chdir(self.name), open(self.logfile, 'w') as fh:
            self.setUp()
            cmd = self.getCmd()
            return subprocess.run(cmd, stdout=fh, stderr=fh, shell=True)

    def setUp(self):
        """
        Set up the input files.
        """
        pass

    def getCmd(self, write_cmd=True):
        """
        Get the command to run the program.

        :param write_cmd bool: whether to write the command to a file
        :return str: the command
        """
        pre = [x for x in [self.PRE_RUN] if x]
        cmd = self.SEP.join(map(str, pre + self.getArgs()))
        if write_cmd:
            with open(f'{self.NAME}_cmd' if self.NAME else 'cmd', 'w') as fh:
                fh.write(cmd)
        return cmd

    def getArgs(self):
        """
        The args to build the command from.

        :return list: the arguments to build the command
        """
        return self.tokens

    @property
    @functools.cache
    def outfiles(self):
        """
        Search for output files from the extension, directory, and jobname.

        :return list: the outfiles found.
        :raise FileNotFoundError: no outfiles found
        """
        ext = self.EXTS.get(self.name, self.EXT)
        pattern = f"{self.jobname}{ext}"
        relpath = os.path.join(self.name, pattern)
        outfiles = glob.glob(relpath)
        if not outfiles:
            raise FileNotFoundError(f"No output file found with {relpath}")
        return outfiles

    @property
    @functools.cache
    def files(self):
        """
        The input files to run the program with paths modified with the dirname.

        :return list: input files.
        """
        if not self._files:
            return []
        relpath = os.path.relpath(os.curdir, start=self.name)
        return [
            x if os.path.isabs(x) else os.path.join(relpath, x)
            for x in self._files
        ]


class Submodule(Base):

    PRE_RUN = jobutils.NEMD_MODULE


class Lmp(Submodule):
    """
    Class to run lammps simulations.
    """

    EXT = lammpsfix.CUSTOM_EXT

    def __init__(self, struct, *args, **kwargs):
        """
        :param struct Struct: the structure to get in script and data file from.
        """
        super().__init__(*args, **kwargs)
        self.struct = struct
        name = os.path.splitext(os.path.basename(self.files[0]))[0]
        self.name = f"lammps{name.removeprefix(self.jobname)}"
        Lmp.files.fget.cache_clear()

    def getArgs(self):
        """
        See parent class for docs.
        """
        return [
            symbols.LMP, jobutils.FLAG_IN, self.struct.inscript,
            jobutils.FLAG_SCREEN, symbols.LMP_LOG, jobutils.FLAG_LOG,
            symbols.LMP_LOG
        ]

    def setUp(self):
        """
        See parent class for docs.
        """
        self.struct.writeIn()
        osutils.symlink(self.files[0], self.struct.datafile)


class Alamode(Submodule):
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
        super().__init__(*args, **kwargs)
        self.crystal = crystal
        self.name = self.crystal.mode

    def getArgs(self):
        """
        See parent class for docs.
        """
        return [self.EXES[self.crystal.mode], self.crystal.inscript]

    def setUp(self):
        """
        See parent class for docs.
        """
        self.crystal.write()
        if self.crystal.mode == self.SUGGEST:
            return
        filename = os.path.basename(self.files[0])
        if self.crystal.mode == self.OPTIMIZE:
            filename = f"{self.jobname}{symbols.DFSET_EXT}"
        osutils.symlink(self.files[0], filename)


class Tools(Submodule):
    """
    Class to run the scripts in the 'tools' directory.
    """
    PRE_RUN = jobutils.NEMD_RUN
    DISPLACE = 'displace'
    EXTRACT = 'extract'
    EXTS = {DISPLACE: "*.lammps"}

    def __init__(self, name, *args, **kwargs):
        """
        :param name str: the subdirectory name
        """
        super().__init__(*args, name=name, **kwargs)

    def getArgs(self):
        """
        See parent class for docs.
        """
        scr = envutils.get_data('tools', f'{self.name}.py', module='alamode')
        if self.name == self.EXTRACT:
            return [scr, '--LAMMPS'] + self.files
        return [
            scr, '--prefix', self.jobname, '--mag', 0.01, '--LAMMPS',
            self.files[0], '-pf', self.files[1]
        ]
