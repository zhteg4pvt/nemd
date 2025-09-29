# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles system-wide environmental variables.
"""
import collections
import functools
import importlib
import os

NEMD = 'nemd'
NEMD_SRC = 'NEMD_SRC'
DEBUG = 'DEBUG'
PYTHON = 'PYTHON'
JOBNAME = 'JOBNAME'
INTERAC = 'INTERAC'
MEM_INTVL = 'MEM_INTVL'


class Base(collections.UserDict):

    def __init__(self, environ=None):
        """
        :param environ dict: the environmental dict.
        """
        super().__init__(environ if os.environ is None else os.environ)


class Mode(int):
    """
    Python mode.
    """
    # -1: without compilation; 0: native python; 1: compiled python
    CACHE = 2  # cached compilation

    def __new__(cls, mode=CACHE):
        return super().__new__(cls, os.environ.get(PYTHON, mode))

    @functools.cached_property
    def orig(self):
        """
        Whether the python mode is the original flavor.

        :return str: whether the python mode is the original flavor.
        """
        return self == 0

    @functools.cached_property
    def kwargs(self):
        """
        Get the jit decorator kwargs.

        :return dict: jit decorator kwargs.
        """
        return {'nopython': self.no, 'cache': self == self.CACHE}

    @functools.cached_property
    def no(self, modes=(1, CACHE)):
        """
        Whether the nopython mode is on.

        :param modes tuple: nopython modes.
        :return str: whether the nopython mode is on.
        """
        return self in modes


def is_interac():
    """
    Whether the interactive modo is on.

    :return bool: whether the debug modo is on.
    """
    return bool(os.environ.get(INTERAC))


def get_jobname():
    """
    The jobname of the current execution.

    :return str: the jobname.
    """
    return os.environ.get(JOBNAME)


def get_mem_intvl():
    """
    Return the memory profiling interval if the execution is in the memory
    profiling mode.

    :return float: The memory recording interval if the memory profiling
        environment flag is on.
    """
    try:
        intvl = float(os.environ.get(MEM_INTVL))
    except (TypeError, ValueError):
        # MEM_INTVL is not set or set to a non-numeric value
        pass
    else:
        # Only return the valid time interval
        return intvl if intvl > 0 else None


def get_src(*arg):
    """
    Get the source code dir.

    :return str: the source code dir
    """
    nemd_src = os.environ.get(NEMD_SRC)
    return os.path.join(nemd_src, *arg) if nemd_src else None


def test_data(*args):
    """
    Get the pathname of the test data.

    :param args str list: the directory and file name of the test file.
    :return str: the pathname
    """
    return get_src('test', 'data', *args)


def get_module_dir(name=NEMD):
    """
    Get the module path.

    :param name str: the module name.
    :return str: the module path
    """
    pathname = get_src('module', name)
    if pathname:
        return pathname
    return os.path.dirname(importlib.util.find_spec(name).origin)


def get_data(*args, module=NEMD, base=None):
    """
    Get the data path of in a module.

    :param args tuple: the str to be joined as the further path from the base.
    :param module str: the module name.
    :param base str: the base dir name with respect to the module path.
    :return str: the data path.
    """
    if base is None:
        base = 'data' if module == NEMD else module
    pathname = [get_module_dir(name=module), base] + list(args)
    return os.path.join(*[x for x in pathname if x])
