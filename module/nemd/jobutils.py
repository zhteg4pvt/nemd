# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module shares job flags, handles command list, defines jobname, and
registers output.
"""
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
FN_DOCUMENT = f'{SIGNAC}_job_document{JSON_EXT}'
FN_STATE_POINT = f'{SIGNAC}_statepoint{JSON_EXT}'


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


def get_name(file, name='Name'):
    """
    Get the jobname from the filename.

    :param file str: the filename of a driver or workflow.
    :param name str: class name
    :return str: the jobname extracted from the filename.
    """
    if file is None:
        words = re.findall('[A-Z][^A-Z]*', name)
        return '_'.join([x.lower() for x in words])
    basename = os.path.basename(file)
    match = re.match(r'(\w+)_(driver|workflow).py', basename)
    return match.group(1) if match else os.path.splitext(basename)[0]


def add_outfile(outfile,
                jobname=None,
                document=FN_DOCUMENT,
                file=False,
                log=False):
    """
    Register the outfile to the job control.

    :param outfile str: the outfile to be registered
    :param jobname str: register the file under this jobname
    :param document str: the job control information is saved into this file
    :param file bool: set this file as the single output file
    :param log bool: set this file as the log file
    """
    if outfile is None:
        return
    if jobname is None:
        jobname = envutils.get_jobname()
    try:
        with open(document) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    data.setdefault(OUTFILES, {}).setdefault(jobname, [])
    if outfile not in data[OUTFILES][jobname]:
        data[OUTFILES][jobname].append(outfile)
    if file:
        data.setdefault(OUTFILE, {})
        data[OUTFILE].setdefault(jobname, outfile)
    if log:
        data.setdefault(LOGFILE, {})
        data[LOGFILE].setdefault(jobname, outfile)
    with open(document, 'w') as fh:
        json.dump(data, fh)


class Job:
    """
    A class to mimic a signac.job.Job for testing purpose.
    """

    def __init__(self, job_dir=os.curdir):
        """
        Initialize a Job object.

        :param job_dir str: the directory of the job
        """
        self.dir = job_dir
        self.statepoint = self.load(FN_STATE_POINT)
        self.doc = self.load(FN_DOCUMENT)
        self.document = self.doc
        self.project = types.SimpleNamespace(doc={},
                                             workspace='workspace',
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
