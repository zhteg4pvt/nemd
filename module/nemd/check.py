# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides check related classes.
"""
import filecmp
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from nemd import envutils
from nemd import jobutils
from nemd import lmpfull
from nemd import logutils
from nemd import plotutils


class Exist(logutils.Base):
    """
    The class to perform file existence check.
    """
    KEYS = set()

    def __init__(self, *args, logger=None, **kwargs):
        """
        :param args str: the target filenames
        """
        super().__init__(logger=logger)
        self.args = args
        self.kwargs = kwargs
        for key in self.KEYS.intersection(self.kwargs.keys()):
            self.kwargs[key] = eval(self.kwargs[key])

    def run(self):
        """
        The main method to check the existence of files.
        """
        for target in self.args:
            if os.path.isfile(target):
                continue
            self.error(f"{target} not found.")

    def error(self, msg):
        """
        Print this message and exit the program.

        :param msg str: the msg to be printed
        """
        self.log(msg)
        sys.exit(1)


class Glob(Exist):
    """
    The class to perform file glob check.
    """

    def __init__(self, *args, num=None, **kwargs):
        """
        :param num str: the number of files to be found.
        """
        super().__init__(*args, **kwargs)
        self.num = int(num) if num else None

    def run(self):
        """
        The main method to check the existence of files.
        """
        num = sum([len(glob.glob(x)) for x in self.args])
        if num == self.num or (num and self.num is None):
            return
        self.error(f"{num} files found. ({self.num})")


class Has(Exist):
    """
    The class to check the containing file strings.
    """

    def __init__(self, *args):
        super().__init__(args[0])
        self.contents = args[1:]

    def run(self):
        """
        The main method to check the containing file strings.
        """
        super().run()
        with open(self.args[0]) as fh:
            contents = fh.read()
        for content in self.contents:
            if content in contents:
                continue
            self.error(f"{content} not found in {self.args[0]}.")


class Cmp(Exist):
    """
    The class to perform file comparison.
    """
    KEYS = {'atol', 'rtol', 'equal_nan'}

    def __init__(self, *args, selected=None, **kwargs):
        """
        :param selected str: the selected csv column.
        """
        super().__init__(*args, **kwargs)
        self.selected = selected

    def run(self):
        """
        The main method to compare files.
        """
        super().run()
        self.file()
        self.csv()
        self.data()

    def file(self):
        """
        Compare the file content for exact match.
        """
        if self.kwargs:
            return
        for target in self.args[1:]:
            if filecmp.cmp(self.args[0], target):
                continue
            self.errorDiff(target)

    def errorDiff(self, target):
        """
        Error message on file difference.

        :param target: the target file that is different from the original.
        """
        self.error(f"{self.args[0]} is different from {target}.")

    def csv(self):
        """
        Compare csv files using np.allclose.
        """
        if not self.kwargs or not all(x.endswith('.csv') for x in self.args):
            return
        obj, non = self.readCsv(self.args[0])
        shapes = [*obj.shape, *non.shape]
        for target in self.args[1:]:
            t_obj, t_non = self.readCsv(target)
            if shapes == [*t_obj.shape, *t_non.shape] and all(obj == t_obj) \
                    and np.allclose(non, t_non, **self.kwargs):
                continue
            self.errorDiff(target)

    def readCsv(self, filename, object='object'):
        """
        Read csv and split by object type.

        :param filename str: the csv file to read.
        :param object str: the object data type.
        return tuple: include object, exclude object.
        """
        data = pd.read_csv(filename)
        if self.selected:
            data = data[[self.selected]]
        return data.select_dtypes(include=object), data.select_dtypes(
            exclude=object)

    def data(self):
        """
        Compare the lammps data files.
        """
        if not self.kwargs or not all(x.endswith('.data') for x in self.args):
            return
        origin = lmpfull.Reader.read(self.args[0])
        for target in self.args[1:]:
            if origin.allClose(lmpfull.Reader.read(target), **self.kwargs):
                continue
            self.errorDiff(target)


class Collect(Exist):
    """
    The class to collect the log files and plot the requested data.
    """
    DROPNA = 'dropna'
    KEYS = {DROPNA}
    TASK_TIME = 'task_time'
    TIME_MIN = 'Task Time (min)'
    MEMORY_MB = 'Memory (MB)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.figs = []
        self.outfile = f"{self.name}.csv"
        if not self.args:
            self.args = [TASK_TIME]
        self.kwargs.setdefault(self.DROPNA, True)

    def run(self):
        """
        Main method to run.
        """
        self.set()
        self.plot()

    def set(self, finished='finished', func=lambda x: x.total_seconds() / 60):
        """
        Set the time and memory data from the log files.

        :param func 'func': map the task time from seconds to minutes.
        """
        self.data = logutils.Reader.collect(*self.args, **self.kwargs)
        if not self.kwargs[self.DROPNA] and (self.data.empty
                                             or self.data.isna().any().any()):
            self.error(self.data[self.data.isna().any(axis=1)].to_markdown())
        self.data.rename(columns={'memory': 'Memory (MB)'}, inplace=True)
        if self.TASK_TIME in self.data.columns:
            self.data[self.TASK_TIME] = self.data[self.TASK_TIME].map(func)
            columns = {self.TASK_TIME: self.TIME_MIN}
            self.data.rename(columns=columns, inplace=True)
        if finished in self.data.columns:
            first = self.data[finished].min()
            self.data[finished] -= first
            self.data[finished] = self.data[finished].map(func)
            columns = {finished: f"{finished.capitalize()} on {first} (min)"}
            self.data.rename(columns=columns, inplace=True)
        self.data.to_csv(self.outfile)
        jobutils.Job.reg(self.outfile)

    def plot(self):
        """
        Plot the data. Time and memory are plotted together if possible.
        """
        if self.data.empty:
            return
        columns, twinx = self.data.columns, None
        if len(columns) == 2:
            columns, twinx = columns[:1], columns[1]
        for col in columns:
            with plotutils.pyplot(inav=envutils.is_interac()) as plt:
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(1, 1, 1)
                color = 'g' if twinx else 'k'
                ax1.plot(self.data.index, self.data[col], f'{color}-.*')
                ax1.set_xlabel(self.data.index.name)
                ax1.set_ylabel(col, color=color)
                if twinx:
                    ax1.tick_params(axis='y', colors='g')
                    ax2 = ax1.twinx()
                    ax2.spines['left'].set_color('g')
                    ax2.plot(self.data.index,
                             self.data[twinx],
                             'b--o',
                             markerfacecolor='none',
                             alpha=0.9)
                    ax2.set_ylabel(twinx, color='b')
                    ax2.tick_params(axis='y', colors='b')
                    ax2.spines['right'].set_color('b')
                fig.tight_layout()
                outfile = self.name
                if not twinx:
                    outfile += f'_{col.split()[0].lower()}'
                outfile += '.png'
                fig.savefig(outfile)
                jobutils.Job.reg(outfile)
                self.figs.append(fig)


if __name__ == "__main__":
    """
    Run module as a script.
    """
    Classes = [Exist, Glob, Has, Cmp, Collect]
    try:
        Class = next(x for x in Classes if x.name == sys.argv[1])
    except StopIteration:
        sys.exit(f'{sys.argv[1]} found. ({[x.name for x in Classes]})')
    kwargs_re = re.compile(r'(\w*)=(.*)')
    matches = [kwargs_re.match(x) for x in sys.argv[2:]]
    args = [x.strip() for x, y in zip(sys.argv[2:], matches) if y is None]
    kwargs = [[y.strip() for y in x.groups()] for x in matches if x]
    Class(*args, **dict(kwargs)).run()
