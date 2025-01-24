# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module reads, parsers, and analyzes trajectories.

Unzip a GZ trajectory file: gzip -dv dump.custom.gz
"""
import base64
import contextlib
import gzip
import io
import itertools
import os
import subprocess

import numpy as np
import pandas as pd

from nemd import analyzer
from nemd import constants
from nemd import frame
from nemd import symbols


class Time(pd.Index):
    """
    Class to represent the time index of a trajectory.
    """

    def __new__(cls, *args, options=None):
        """
        :param options `namedtuple`: command line options
        :return `pd.Index`: the time index
        """
        obj = super().__new__(cls, *args)
        obj.sidx = 0 if options is None else options.last_pct.getSidx(obj)
        obj.name = f"{symbols.TIME_LB} ({obj.sidx})"
        obj.start = obj[obj.sidx]
        return obj


class Traj(list):
    """
    A class to read, parse, and analyze a trajectory file.
    """
    STEP_MK = 'ITEM: TIMESTEP'
    STEP_CMD = f"zgrep -A1 '{STEP_MK}' {{file}} | sed '/{STEP_MK}/d;/^--$/d'"

    def __init__(self, file=None, contents=None, options=None):
        super().__init__()
        self.file = file
        self.contents = contents
        self.options = options
        self.time = None
        self.slice = getattr(self.options, 'slice') or [None]
        self.tasks = [x for x in self.options.task if x in analyzer.ALL_FRM]
        if self.contents:
            _, contents = contents.split(',')
            self.contents = base64.b64decode(contents).decode("utf-8")

    def load(self):
        """
        Read, parse, and set the trajectory frames.
        """
        self.extend(list(itertools.islice(self.frame, *self.slice)))
        # FIXME: obtain the timestep from log file instead of options
        fac = getattr(self.options, 'timestep', 1) / constants.PICO_TO_FEMTO
        self.time = Time([x.step * fac for x in self], options=self.options)

    @property
    def frame(self, start=0):
        """
        Open and read the trajectory frames.

        :param start int: frames with step number < this value are fully read.
        :return iterator of 'Frame': trajectory frames
        """
        if not self.tasks:
            # No all-frame tasks found, fully read the last frames.
            proc = subprocess.Popen(self.STEP_CMD.format(file=self.file),
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            stdout, stderr = proc.communicate()
            steps = np.loadtxt(io.StringIO(stdout), dtype=int)
            # From the 2nd to the last frame in case of a broken last one.
            start = steps[self.options.last_pct.getSidx(steps, buffer=1)]
        with self.open(file=self.file, contents=self.contents) as fh:
            while True:
                try:
                    yield frame.Frame.read(fh, start=start)
                except EOFError:
                    return

    @staticmethod
    @contextlib.contextmanager
    def open(file=None, contents=None):
        """
        Open trajectory file.

        :param file: the file with path
        :type file: str
        :param contents: the trajectory contents
        :type contents: str
        :return: the file handle
        :rtype: '_io.TextIOWrapper'
        """
        if all([file is None, contents is None]):
            raise ValueError(f'Please specify either file or contents.')
        if file:
            if os.path.isfile(file):
                func = gzip.open if file.endswith('.gz') else open
                fh = func(file, 'rt')
            else:
                raise FileNotFoundError(f'{file} not found')
        else:
            fh = io.StringIO(contents)
        try:
            yield fh
        finally:
            fh.close()

    @property
    def sel(self):
        """
        Return the selected frames.

        :return list of 'Frame': selected frames.
        """
        return self[self.time.sidx:]
