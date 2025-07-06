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

from nemd import envutils
from nemd import jobutils
from nemd import lmpfull
from nemd import logutils
from nemd import np
from nemd import pd
from nemd import plotutils


class Exist(logutils.Base):
    """
    The class to perform file existence check.
    """

    def __init__(self, *args, logger=None):
        """
        :param args str: the target filenames
        """
        super().__init__(logger=logger)
        self.args = args

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

    def __init__(self, *args, **kwargs):
        self.keys = self.KEYS.intersection(kwargs.keys())
        self.kwargs = {x: eval(kwargs.pop(x)) for x in self.keys}
        super().__init__(*args, **kwargs)

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
        if self.keys:
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
        Compare csv files via np.allclose.
        """
        if not self.keys or not all(x.endswith('.csv') for x in self.args):
            return
        origin = pd.read_csv(self.args[0])
        object = origin.select_dtypes(include='object')
        nonobj = origin.select_dtypes(exclude='object')
        for target in self.args[1:]:
            data = pd.read_csv(target)
            tgt_obj = data.select_dtypes(include='object')
            if object.shape != tgt_obj.shape:
                self.errorDiff(target)
            if not all(object == tgt_obj):
                self.errorDiff(target)
            tgt_nonobj = data.select_dtypes(exclude='object')
            if nonobj.shape != tgt_nonobj.shape:
                self.errorDiff(target)
            if not np.allclose(nonobj, tgt_nonobj, **self.kwargs):
                self.errorDiff(target)

    def data(self):
        """
        Compare the lammps data files.
        """
        if not self.keys or not all(x.endswith('.data') for x in self.args):
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
    TIME_MIN = 'Task Time (min)'
    MEMORY_MB = 'Memory (MB)'
    COLUMNS = {'task_time': TIME_MIN, 'memory': 'Memory (MB)'}
    TWINX = {TIME_MIN: MEMORY_MB}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.fig = None
        self.outfile = f"{self.name}.csv"

    def run(self):
        """
        Main method to run.
        """
        self.set()
        self.plot()

    def set(self, func=lambda x: x.total_seconds() / 60):
        """
        Set the time and memory data from the log files.

        :param func 'func': map the task time from seconds to minutes.
        """
        self.data = logutils.Reader.collect(*self.args)
        self.data.rename(columns=self.COLUMNS, inplace=True)
        if self.TIME_MIN in self.data.columns:
            self.data[self.TIME_MIN] = self.data[self.TIME_MIN].map(func)
        if self.data.empty:
            return
        self.data.to_csv(self.outfile)
        jobutils.Job.reg(self.outfile)

    def plot(self):
        """
        Plot the data. Time and memory are plotted together if possible.
        """
        if self.data.empty:
            return
        twinx_lbs = [self.TWINX.get(x) for x in self.args]
        for col in self.data.columns.difference(twinx_lbs):
            twinx_lb = self.TWINX.get(col)
            if twinx_lb and twinx_lb not in self.data.columns:
                twinx_lb = None
            with plotutils.pyplot(inav=envutils.is_interac()) as plt:
                self.fig = plt.figure(figsize=(10, 6))
                ax1 = self.fig.add_subplot(1, 1, 1)
                color = 'g' if twinx_lb else 'k'
                ax1.plot(self.data.index, self.data[col], f'{color}-.*')
                ax1.set_xlabel(self.data.index.name)
                ax1.set_ylabel(col, color=color)
                if twinx_lb:
                    ax1.tick_params(axis='y', colors='g')
                    ax2 = ax1.twinx()
                    ax2.spines['left'].set_color('g')
                    ax2.plot(self.data.index,
                             self.data[twinx_lb],
                             'b--o',
                             markerfacecolor='none',
                             alpha=0.9)
                    ax2.set_ylabel(twinx_lb, color='b')
                    ax2.tick_params(axis='y', colors='b')
                    ax2.spines['right'].set_color('b')
                self.fig.tight_layout()
                outfile = f"{self.name}.png"
                self.fig.savefig(outfile)
                jobutils.Job.reg(outfile)


if __name__ == "__main__":
    """
    Run library module as a script.
    """
    Classes = [Exist, Glob, Has, Cmp, Collect]
    try:
        Class = next(x for x in Classes if x.name == sys.argv[1])
    except StopIteration:
        sys.exit(f'{sys.argv[1]} found. ({[x.name for x in Classes]})')
    kwargs_re = re.compile(r'(.*)=(.*)')
    matches = [kwargs_re.match(x) for x in sys.argv[2:]]
    args = [x.strip() for x, y in zip(sys.argv[2:], matches) if y is None]
    kwargs = [[y.strip() for y in x.groups()] for x in matches if x]
    Class(*args, **dict(kwargs)).run()
