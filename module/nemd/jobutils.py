# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module shares job flags, handles command list, defines jobname, and
registers output.
"""
import functools
import glob
import json
import os
import re

from nemd import envutils
from nemd import objectutils
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
FLAG_DIR = '-dir'
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


class Job(objectutils.Object):
    """
    Json file centered job applications.
    """

    LOGFILE = 'logfile'
    OUTFILE = 'outfile'
    OUTFILES = 'outfiles'
    JOB_DOCUMENT = f'.{{jobname}}_document{symbols.JSON_EXT}'

    def __init__(self, jobname=None, dirname=None):
        """
        :param jobname str: the jobname
        :param dirname str: the job dirname
        """
        self.jobname = jobname or envutils.get_jobname() or self.name
        self.dirname = dirname or os.curdir

    @property
    @functools.cache
    def data(self):
        """
        Return the job data.

        :return dict: the data in the job json.
        """
        try:
            with open(self.file) as fh:
                return json.load(fh)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            return {}

    @property
    @functools.cache
    def file(self):
        """
        Get the json file pathname.

        :return str: the pathname of the json file
        """
        filename = self.JOB_DOCUMENT.format(jobname=self.jobname)
        return os.path.join(self.dirname, filename)

    def append(self, value, key=OUTFILES):
        """
        Add the file to the outfiles.

        :param value `str`: the value to add
        :param key `key`: the key
        :param file str: the file to be added.
        """
        self.data.setdefault(key, [])
        if value not in self.data[key]:
            self.data[key].append(value)

    def set(self, value, key=OUTFILE):
        """
        Set the key / value.

        :param value str: the value to be set.
        :param key str: the key
        """
        self.data[key] = value

    def write(self):
        """
        Write the data to the json.
        """
        with open(self.file, 'w') as fh:
            json.dump(self.data, fh)

    def getFile(self, key=OUTFILE):
        """
        Get the file.

        :param key str: the file type.
        :return str: the obtained & existed file.
        """
        filename = self.data.get(key)
        return os.path.join(self.dirname, filename) if filename else None

    @property
    @functools.cache
    def logfile(self):
        """
        Return the log file.

        :return str: the log file
        """
        return self.getFile(key=self.LOGFILE)

    def getJobs(self, dirname=None, patt=JOB_DOCUMENT.format(jobname='*')):
        """
        Get the all the jobs in a directory.

        :param dirname 'str': the job driname
        :param patt str: the pattern to search job json files
        :return 'Job' list: the jobs in the directory
        """
        if dirname is None:
            dirname = self.dirname
        files = glob.glob(os.path.join(dirname, patt))
        return [Job.fromFile(x) for x in files]

    @staticmethod
    def fromFile(namepath,
                 rex=re.compile(JOB_DOCUMENT.format(jobname=r'(\w*)'))):
        """
        Get the job based on the job json file.

        :param rex 're.Pattern': the regular expression to identify jobname
        :return 'Job' list: the json jobs
        """
        driname, basename = os.path.split(namepath)
        return Job(rex.match(basename).group(1), dirname=driname)

    def clean(self):
        """
        Clean the json file.
        """
        try:
            os.remove(self.file)
        except FileNotFoundError:
            pass

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
        job.append(outfile)
        if file:
            job.set(outfile)
        if log:
            job.set(outfile, key=cls.LOGFILE)
        job.write()
