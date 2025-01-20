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
    PRE_RUN = None
    EXT = symbols.LOG
    EXTS = {}

    def __init__(self, name=os.curdir, options=None, files=None):
        """
        :param name str: the subdirectory name to run the command
        :param options 'argparse.Namespace': the command line options
        :param files list: input files
        """
        self.name = name
        self.options = options
        self._files = files
        self.logfile = f'{self.options.jobname}{symbols.LOG}'

    def run(self):
        """
        Make & change directory, set up, build command, and run command.
        """
        with osutils.chdir(self.name), open(self.logfile, 'w') as fh:
            self.setUp()
            cmd = self.getCmd()
            subprocess.run(cmd, stdout=fh, stderr=fh, shell=True)

    def setUp(self):
        """
        Set up the input files and symbolic links.
        """
        pass

    def getCmd(self, write_cmd=True):
        """
        Get the command to run the program.

        :param write_cmd bool: whether to write the command to a file
        :return str: the command
        """
        pre = [x for x in [self.PRE_RUN] if x]
        cmd = ' '.join(map(str, pre + self.args))
        if write_cmd:
            with open('cmd', 'w') as fh:
                fh.write(cmd)
        return cmd

    @property
    def args(self):
        """
        The args to build the command from.

        :return list: the arguments to build the command
        """
        return ["echo"]

    @property
    @functools.cache
    def outfiles(self):
        """
        Search for output files from the extension, directory, and jobname.

        :return list: the outfiles found.
        """
        ext = self.EXTS.get(self.name, self.EXT)
        pattern = f"{self.options.jobname}{ext}"
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


class Lmp(Base):
    """
    Class to run lammps simulations.
    """

    EXT = lammpsfix.CUSTOM_EXT

    def __init__(self, struct, *arg, **kwargs):
        """
        :param struct Struct: the structure to get in script and data file from.
        """
        super().__init__(*arg, **kwargs)
        self.struct = struct
        name = os.path.splitext(os.path.basename(self.files[0]))[0]
        self.name = f"lammps{name.removeprefix(self.options.jobname)}"
        Lmp.files.fget.cache_clear()

    @property
    def args(self):
        """
        See parent class for docs.
        """
        return [
            symbols.LMP_SERIAL, jobutils.FLAG_IN, self.struct.inscript,
            jobutils.FLAG_SCREEN, symbols.NONE, jobutils.FLAG_LOG,
            symbols.LMP_LOG
        ]

    def setUp(self):
        """
        See parent class for docs.
        """
        self.struct.writeIn()
        osutils.symlink(self.files[0], self.struct.datafile)


class Alamode(Base):
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

    def __init__(self, crystal, *arg, **kwargs):
        """
        :param crystal Crystal: the crystal to get in script from.
        """
        super().__init__(*arg, **kwargs)
        self.crystal = crystal
        self.name = self.crystal.mode

    @property
    def args(self):
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
            filename = f"{self.options.jobname}{symbols.DFSET_EXT}"
        osutils.symlink(self.files[0], filename)


class Tools(Base):
    """
    Class to run the scripts in the 'tools' directory.
    """

    PRE_RUN = jobutils.RUN_NEMD
    DISPLACE = 'displace'
    EXTRACT = 'extract'
    EXTS = {DISPLACE: "*.lammps"}

    @property
    def args(self):
        """
        See parent class for docs.
        """
        scr = envutils.get_data('tools', f'{self.name}.py', module='alamode')
        if self.name == self.EXTRACT:
            return [scr, '--LAMMPS'] + self.files
        return [
            scr, '--prefix', self.options.jobname, '--mag', 0.01, '--LAMMPS',
            self.files[0], '-pf', self.files[1]
        ]
