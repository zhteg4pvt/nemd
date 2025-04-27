# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module shares job flags, handles command list, defines jobname, and
registers output.
"""
import glob
import json
import os
import re

from nemd import builtinsutils
from nemd import envutils
from nemd import symbols

NEMD_RUN = 'nemd_run'
NEMD_MODULE = 'nemd_module'

FLAG_NAME = '-NAME'
FLAG_JOBNAME = '-JOBNAME'
FLAG_PYTHON = '-PYTHON'
FLAG_INTERAC = '-INTERAC'
FLAG_DEBUG = '-DEBUG'
FLAG_CPU = '-CPU'
FLAG_SEED = '-seed'
FLAG_DIRNAME = '-dirname'
FLAG_TASK = '-task'
FLAG_LOG = '-log'
FLAG_IN = '-in'
FLAG_SCREEN = '-screen'
PARALLEL = 'parallel'
JOB = 'job'
WORKSPACE = 'workspace'


def get_arg(args, flag, default=None, first=True):
    """
    Get the value after the flag in command arg list.

    :param args list: the arg list
    :param flag str: set the value after this flag
    :param default str: the default if the flag doesn't exist or not followed by
        value(s)
    :param first bool: only return the first value after the flag
    :return str or list: the value(s) after the flag
    """
    try:
        idx = args.index(flag)
    except ValueError:
        # Flag not found
        return default

    val = args[idx + 1]
    if val.startswith('-'):
        # Flag followed by another flag
        return

    if first:
        return val

    selected = []
    for delta, arg in enumerate(args[idx + 1:]):
        if arg.startswith('-'):
            break
        selected.append(arg)
    return selected


def pop_arg(args, flag, val=None):
    """
    Pop the value after the flag in command arg list.

    :param args list: the arg list
    :param flag str: set the value after this flag
    :param val str: the default if no flag found or no value(s) followed
    :return list: the values after the flag
    """
    values = get_arg(args, flag, first=False)
    if values is None:
        try:
            args.remove(flag)
        except ValueError:
            pass
        return val

    flag_idx = args.index(flag)
    deta = len(values) if isinstance(values, list) else 1
    for idx in reversed(range(flag_idx, flag_idx + deta + 1)):
        args.pop(idx)
    return values


def set_arg(args, flag, val):
    """
    Set the value after the flag in command arg list.

    :param args list: the arg list
    :param flag str: set the value after this flag
    :param val str: the new value
    :return list: the modified arg list
    """
    try:
        idx = args.index(flag)
    except ValueError:
        args.extend([flag, str(val)])
        return args
    args[idx + 1] = val
    return args


class Job(builtinsutils.Dict, builtinsutils.Object):
    """
    File-centered job applications.
    """

    JOB_DOC = f'.{{jobname}}_document{symbols.JSON_EXT}'

    def __init__(self, jobname=None, dirname=None):
        """
        :param jobname str: the jobname
        :param dirname str: the job dirname
        """
        super().__init__(_logfile=None, _outfile=None, _outfiles=[])
        self.setattr('jobname', jobname or envutils.get_jobname() or self.name)
        self.setattr('dirname', dirname or os.getcwd())
        self.setattr('_file', self.JOB_DOC.format(jobname=self.jobname))
        try:
            with open(self.file) as fh:
                self.update(json.load(fh))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass

    @property
    def file(self):
        """
        Get the pathname of the json file.

        :return: the job json file
        """
        return self.getFile(self._file)

    def getFile(self, filename):
        """
        Get the pathname of a filename.

        :param filename str: the filename
        :return: the pathname
        """
        return os.path.join(self.dirname, filename) if filename else filename

    @property
    def outfile(self):
        """
        Get the pathname of the outfile.

        :return: the outfile
        """
        return self.getFile(self._outfile)

    @property
    def logfile(self):
        """
        Get the pathname of the logfile.

        :return: the logfile
        """
        return self.getFile(self._logfile)

    def write(self):
        """
        Write the data to the json.
        """
        with open(self.file, 'w') as fh:
            json.dump(self, fh)

    @classmethod
    def search(cls, dirname=None, patt=JOB_DOC.format(jobname='*')):
        """
        Search the directory and return the found jobs.

        :param dirname 'str': the job driname
        :param patt str: the pattern to search job json files
        :return 'Job' list: the jobs in the directory
        """
        patt = os.path.join(dirname if dirname else os.getcwd(), patt)
        return [Job.fromFile(x) for x in glob.glob(patt)]

    @staticmethod
    def fromFile(pathname, rex=re.compile(JOB_DOC.format(jobname=r'(\w*)'))):
        """
        Get the job based on the job json file.

        :param pathname 'str': the pathname of a job json file
        :param rex 're.Pattern': the regular expression to identify jobname
        :return 'Job' list: the json jobs
        """
        driname, basename = os.path.split(pathname)
        return Job(rex.match(basename).group(1), dirname=driname)

    @classmethod
    def reg(cls, outfile, jobname=None, file=False, log=False):
        """
        Register the outfile under the job control.

        :param outfile str: the single output file
        :param jobname str: the jobname to determine the json file
        :param file bool: set this file as the single output file
        :param log bool: set this file as the log file
        """
        if outfile is None:
            return
        job = cls(jobname)
        job._outfiles.append(outfile)
        if file:
            job._outfile = outfile
        if log:
            job._logfile = outfile
        job.write()
