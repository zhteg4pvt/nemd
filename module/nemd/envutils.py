# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles system-wide environmental variables.
"""
import importlib
import os

NEMD = 'nemd'
NEMD_SRC = 'NEMD_SRC'
DEBUG = 'DEBUG'
PYTHON = 'PYTHON'
JOBNAME = 'JOBNAME'
INTERACTIVE = 'INTERACTIVE'
MEM_INTVL = 'MEM_INTVL'
PYTHON_MODE = '-1'  # without compilation
ORIGINAL_MODE = '0'  # native python
NOPYTHON_MODE = '1'  # compiled python
CACHE_MODE = '2'  # cached compilation
NOPYTHON = 'nopython'
PYTHON_MODES = [PYTHON_MODE, ORIGINAL_MODE, NOPYTHON_MODE, CACHE_MODE]
NOPYTHON_MODES = [NOPYTHON_MODE, CACHE_MODE]


def is_interactive():
    """
    Whether interactive mode is on.

    :return bool: If interactive mode is on.
    """
    return os.environ.get(INTERACTIVE)


def get_python_mode():
    """
    Get the mode of python compilation.

    :return str: The mode of python compilation.
    """
    return os.environ.get(PYTHON, CACHE_MODE)


def is_original():
    """
    Whether the python mode is the original flavor.

    :return str: whether the python mode is the original flavor.
    """
    return get_python_mode() == ORIGINAL_MODE


def is_nopython():
    """
    Whether the nopython mode is on.

    :return str: whether the nopython mode is on.
    """
    return get_python_mode() in NOPYTHON_MODES


def get_jit_kwargs(**kwargs):
    """
    Get the jit decorator kwargs.

    :return dict: jit decorator kwargs.
    """
    mode = get_python_mode()
    return {
        NOPYTHON: mode in NOPYTHON_MODES,
        'cache': mode == CACHE_MODE,
        **kwargs
    }


def set_jobname_default(name):
    """
    Set the default jobname of the current execution.

    :param name str: The default jobname.
    """
    os.environ.setdefault(JOBNAME, name)


def get_jobname(name=None):
    """
    The jobname of the current execution.

    :return str: the jobname.
    """
    return os.environ.get(JOBNAME, name)


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


def get_nemd_src(*arg):
    """
    Get the source code dir.

    :return str: the source code dir
    """
    nemd_src = os.environ.get(NEMD_SRC)
    return os.path.join(nemd_src, *arg) if nemd_src else None


def get_module_dir(name=NEMD):
    """
    Get the module path.

    :param name str: the module name.
    :return str: the module path
    """
    pathname = get_nemd_src('module', name)
    if pathname:
        return pathname
    return os.path.dirname(importlib.util.find_spec(name).origin)


def test_data(*args):
    """
    Get the pathname of the test data.

    :param args str list: the directory and file name of the test file.
    :return str: the pathname
    :raise FileNotFoundError: if the source directory cannot be found.
    """
    pathname = get_nemd_src('test', 'data', *args)
    if pathname is None:
        raise FileNotFoundError("Error: test directory cannot be found.")
    return pathname


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
