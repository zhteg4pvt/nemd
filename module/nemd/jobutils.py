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
import types

from nemd import envutils

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
FLAG_SCREEN = '-screen'
FlAG_NAME = '-name'
FLAG_TASK = '-task'
FLAG_SLOW = '-slow'
FLAG_LOG = '-log'
FLAG_IN = '-in'
PROGRESS = 'progress'
JOB = 'job'

LOGFILE = 'logfile'
OUTFILE = 'outfile'
OUTFILES = 'outfiles'

SIGNAC = 'signac'
JSON_EXT = '.json'
FN_DOCUMENT = f'{SIGNAC}_job_document{JSON_EXT}'  # signac.job.Job.FN_DOCUMENT
FN_STATE_POINT = f'{SIGNAC}_statepoint{JSON_EXT}'
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


def get_name(file):
    """
    Get the jobname from the filename.

    :param file str: the filename of a driver or workflow.
    :return str: the jobname extracted from the filename.
    """
    basename = os.path.basename(file)
    match = re.match(r'(\w+)_(driver|workflow).py', basename)
    return match.group(1) if match else os.path.splitext(basename)[0]


def add_outfile(outfile, jobname=None, file=False, log=False):
    """
    Register the outfile under the job control.

    :param outfile str: the single output file
    :param jobname str: the jobname to determine the json file
    :param file bool: set this file as the single output file
    :param log bool: set this file as the log file
    """
    if outfile is None:
        return
    job = Job(name=jobname)
    job.add(outfile)
    if file:
        job.set(outfile)
    if log:
        job.set(outfile, ftype=LOGFILE)
    job.write()


class Job:
    """
    Json file centered job applications.
    """

    OUT = OUTFILE
    JOB_DOCUMENT = f'.{{jobname}}_document{JSON_EXT}'

    def __init__(self, name=None, job=None):
        """
        :param jobname str: the jobname
        :param job 'signac.job.Job': the signac job instance for json path
        """
        self.name = name
        self.job = job.project if self.agg else job
        self.jobname = self.name if self.name else self.default
        self.file = self.JOB_DOCUMENT.format(jobname=self.jobname)
        if self.job:
            self.file = self.job.fn(self.file)

    @classmethod
    @property
    def default(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        return envutils.get_jobname()

    @classmethod
    @property
    def agg(cls):
        """
        Whether this is an aggregator class.

        :return bool: True when this is an aggregator.
        """
        return cls.__name__.endswith('Agg')

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

    def add(self, file, ftype=OUTFILES):
        """
        Add the file to the outfiles.

        :param file str: the file to be added.
        """
        self.data.setdefault(ftype, [])
        if file not in self.data[ftype]:
            self.data[ftype].append(file)

    def set(self, file, ftype=OUT):
        """
        set the file.

        :param file str: the file to be set.
        :param ftype str: the type of the file
        :return:
        """
        self.data[ftype] = file

    def write(self):
        """
        Write the data to the json.
        """
        with open(self.file, 'w') as fh:
            json.dump(self.data, fh)

    def getFile(self, ftype=OUTFILE):
        """
        Get the file.

        :param ftype str: the file type.
        :return str: the obtained & existed file.
        """
        file = self.data.get(ftype)
        if not file:
            return
        if self.job:
            file = self.job.fn(file)
        if os.path.isfile(file):
            return file

    @property
    @functools.cache
    def logfile(self):
        """
        Return the log file.

        :return str: the namepath of the log file
        """
        return self.getFile(ftype=LOGFILE)

    def getJobs(self,
                job=None,
                patt=JOB_DOCUMENT.format(jobname='*'),
                doc_re=re.compile(JOB_DOCUMENT.format(jobname='(\w*)'))):
        """
        Get the all the jobs in a directory.

        :param job 'signac.job.Job': the signac job defines the directory
        :param patt str: the pattern to search job json files
        :param doc_re 're.Pattern': the regular expression to get jobname
        :return 'Job' list: the json jobs
        """
        if job is None:
            job = self.job
        if job:
            patt = job.fn(patt)
        files = glob.glob(patt)
        names = [doc_re.match(os.path.basename(x)).group(1) for x in files]
        return [Job(name=x, job=job) for x in names]

    def clean(self):
        """
        Clean the json file.

        Note: The jobnames of cmd tasks are determined by the cmd content.
        """
        try:
            os.remove(self.file)
        except FileNotFoundError:
            pass


class Mimic:
    """
    A class to mimic a signac.job.Job for testing purpose.
    """

    def __init__(self, dir=os.curdir):
        """
        Initialize a Job object.

        :param dir str: the directory of the job
        """
        self.dir = dir
        self.statepoint = self.load(FN_STATE_POINT)
        self.document = self.load(FN_DOCUMENT)
        self.project = types.SimpleNamespace(doc={},
                                             workspace=WORKSPACE,
                                             path=os.curdir)

    def load(self, basename):
        """
        Load the json file.

        :param basename str: the file name to be loaded.
        :return dict: the loaded json dictionary.
        """
        pathname = os.path.join(self.dir, basename)
        if not os.path.isfile(pathname):
            return {}
        with open(pathname, 'r') as fh:
            return json.load(fh)

    def fn(self, filename):
        """
        Return the full path of the file in the job directory.

        :param filename str: the file name
        :return str: the full path of the file
        """
        return os.path.join(self.dir, filename) if self.dir else filename
