# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles system-wide environmental variables.
"""
import collections
import importlib
import os


class Env(collections.UserDict):
    """
    Class for environmental variables.
    """
    INTVL = 'INTVL'

    def __init__(self, env=None):
        """
        :param env dict: the environmental dict.
        """
        super().__init__(os.environ if env is None else env)

    @property
    def interac(self):
        """
        Whether the interactive modo is on.

        :return bool: whether the interactive modo is on.
        """
        return bool(self.get('INTERAC'))

    @property
    def jobname(self):
        """
        The jobname of the current execution.

        :return str: the jobname.
        """
        return self.get('JOBNAME')

    @property
    def mode(self):
        """
        The jobname of the current execution.

        :return str: the jobname.
        """
        return int(self.get('PYTHON', Mode.CACHE))

    @property
    def intvl(self):
        """
        Return the memory profiling interval if the execution is in the memory
        profiling mode.

        :return float: The memory recording interval if the memory profiling
            environment flag is on.
        """
        try:
            intvl = float(self.get(self.INTVL))
        except (TypeError, ValueError):
            # INTVL is not set or set to a non-numeric value
            pass
        else:
            # Only return the valid time interval
            return intvl if intvl > 0 else None

    @property
    def src(self):
        """
        Get the source code dir.

        :return str: the source code dir
        """
        return self.get('NEMD_SRC', '')


class Mode(int):
    """
    Python mode.
    """
    # -1: without compilation; 0: native python; 1: compiled python
    CACHE = 2  # cached compilation

    def __new__(cls, mode=CACHE):
        return super().__new__(cls, Env().mode)

    @property
    def orig(self):
        """
        Whether the python mode is the original flavor.

        :return str: whether the python mode is the original flavor.
        """
        return self == 0

    @property
    def kwargs(self):
        """
        Get the jit decorator kwargs.

        :return dict: jit decorator kwargs.
        """
        return {'nopython': self.no, 'cache': self == self.CACHE}

    @property
    def no(self, modes=(1, CACHE)):
        """
        Whether the nopython mode is on.

        :param modes tuple: nopython modes.
        :return str: whether the nopython mode is on.
        """
        return self in modes


class Src(str):
    """
    Source code dir.
    """
    NEMD = 'nemd'

    def __new__(cls, src=None):
        """
        :param src str: the source code root dir.
        """
        return super().__new__(cls, src or Env().src)

    def test(self, *args, base='data'):
        """
        Get the pathname of the test data.

        :param base str: the test base dir name with respect to test root.
        :return str: the pathname
        """
        return os.path.join(self, 'test', base, *args)

    def get(self, *args, module=NEMD, base=None):
        """
        Get the data path of in a module.

        :param module str: the module name.
        :param base str: the base dir name with respect to the module path.
        :return str: the data path.
        """
        if base is None:
            base = 'data' if module == self.NEMD else module
        paths = [self, 'module', module
                 ] if self else importlib.import_module(module).__path__
        return os.path.join(*paths, base, *args)
