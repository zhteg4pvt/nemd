# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module reads, parsers, and analyzes trajectories.

Unzip a GZ trajectory file: gzip -dv dump.custom.gz
"""
import gzip
import io
import itertools
import subprocess

import mdtraj
import numpy as np
import pandas as pd

from nemd import constants
from nemd import frame
from nemd import pbc
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
        obj.start = obj[obj.sidx] if obj.size else None
        obj.name = f"{symbols.TIME_LB} ({obj.sidx})"
        return obj


class Traj(list):
    """
    A class to read, parse, and analyze a trajectory file.
    """
    STEP_MK = 'ITEM: TIMESTEP'
    STEP_CMD = f"zgrep -A1 '{STEP_MK}' {{file}} | sed '/{STEP_MK}/d;/^--$/d'"

    def __init__(self, file=None, options=None, start=None, delay=False):
        """
        :param file str: the trajectory file
        :param options 'argparse.Namespace': command line options
        :param start int: frames with step number < this value are fully read.
        """
        super().__init__()
        self.file = file
        self.options = options
        self.start = start
        self.time = None
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        self.setStart()
        sliced = getattr(self.options, 'slice', [None])
        self.extend(list(itertools.islice(self.frame, *sliced)))
        # FIXME: obtain the timestep from log file instead of options
        fac = getattr(self.options, 'timestep', 1) * constants.FEMTO_TO_PICO
        self.time = Time([x.step * fac for x in self], options=self.options)

    def setStart(self):
        """
        Set the start time for full coordinates.
        """
        if self.start is not None:
            return
        if self.file.endswith(symbols.XTC_EXT) or self.options is None or \
                self.options.slice != [None]:
            self.start = 0
            return
        # No all-frame tasks found, only last frames are fully read.
        proc = subprocess.Popen(self.STEP_CMD.format(file=self.file),
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        stdout, stderr = proc.communicate()
        if not stdout:
            return
        steps = np.loadtxt(io.StringIO(stdout), dtype=int)
        if not steps.shape:
            steps = steps.reshape(1)
        # From the 2nd to the last frame in case of a broken last one.
        self.start = steps[self.options.last_pct.getSidx(steps, buffer=1)]

    @property
    def frame(self):
        """
        Open and read the trajectory frames.

        https://docs.lammps.org/Howto_triclinic.html
        https://manual.gromacs.org/current/reference-manual/algorithms/periodic-boundary-conditions.html
        https://mdtraj.org/1.9.4/api/generated/mdtraj.formats.XTCTrajectoryFile.html

        :return generator of 'Frame': trajectory frames
        """
        if self.file.endswith('.xtc'):
            with mdtraj.formats.XTCTrajectoryFile(self.file) as fh:
                for xyz, _, step, box in zip(*fh.read()):
                    # The conventional units in the XTC file are nanometers and picoseconds.
                    xyz *= 10
                    # FIXME: triclinic support
                    box = pbc.Box.fromParams(*np.diag(box * 10))
                    yield frame.Frame(xyz, box=box, step=step.item())
            return
        func = gzip.open if self.file.endswith('.gz') else open
        with func(self.file, 'rt') as fh:
            while True:
                try:
                    yield frame.Frame.read(fh, start=self.start)
                except EOFError:
                    return

    @property
    def sel(self):
        """
        Return the selected frames.

        :return list of 'Frame': selected frames.
        """
        return self[self.time.sidx:]
