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

from nemd import envutils
from nemd import symbols

NEMD_RUN = 'nemd_run'
NEMD_MODULE = 'nemd_module'

OUTFILE = 'outfile'
LOGFILE = 'logfile'
OUTFILES = 'outfiles'
FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_NAME = '-NAME'
FLAG_DEBUG = '-DEBUG'
FLAG_PYTHON = '-PYTHON'
FLAG_CPU = '-cpu'
FLAG_SEED = '-seed'
FLAG_DIR = '-dir'
FLAG_SCREEN = '-screen'
FLAG_SLOW = '-slow'
FLAG_LOG = '-log'
FLAG_IN = '-in'
PROGRESS = 'progress'
TQDM = 'tqdm'
JOB = 'job'

TASK = 'task'
AGGREGATOR = 'aggregator'


def get_arg(args, flag, default=None, first=True):
    """
    Get the value after the flag in command arg list.

    :param args list: the arg list
    :param flag str: set the value after this flag
    :param default str: the default if the flag doesn't exist or not followed by value(s)
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
    :return str or list: the value(s) after the flag
    """
    arg = get_arg(args, flag)
    if arg is None:
        try:
            args.remove(flag)
        except ValueError:
            pass
        return val

    flag_idx = args.index(flag)
    deta = len(arg) if isinstance(arg, list) else 1
    for idx in reversed(range(flag_idx, flag_idx + deta + 1)):
        args.pop(idx)
    return arg


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
    if file is None:
        return
    basename = os.path.basename(file)
    match = re.match('(\w+)_(driver|workflow).py', basename)
    return match.group(1) if match else os.path.splitext(basename)


def add_outfile(outfile,
                jobname=None,
                document=symbols.FN_DOCUMENT,
                file=False,
                log=False):
    """
    Register the outfile to the job control.

    :param outfile str: the outfile to be registered
    :param jobname str: register the file under this jobname
    :param document str: the job control information is saved into this file
    :param file bool: set this file as the single output file
    :param file bool: set this file as the log file
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
