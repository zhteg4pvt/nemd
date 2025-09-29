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

from nemd import constants
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
            if filecmp.cmp(self.args[0], target, shallow=False):
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


class CollBase(Exist):
    """
    The collection base class.
    """
    MEMORY_GB = 'Memory (GB)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.outfile = f"{self.name}.csv"
        self.figs = []

    def run(self):
        """
        Main method to run.
        """
        self.set()
        self.plot()

    def set(self):
        """
        Set the data from the collected csv files.
        """
        files = [os.path.join(x, self.file) for x in self.args]
        datas = [
            pd.read_csv(x, index_col=0) for x in files if os.path.exists(x)
        ]
        if not datas or all([x.empty for x in datas]):
            self.error(f"{files} not found or empty.")
        self.data = pd.concat(datas, axis=1, keys=self.args)
        self.data.columns = self.data.columns.swaplevel(0, 1)
        self.data.drop(columns=[(self.MEMORY_GB, 'mac')], inplace=True)
        self.data.to_csv(self.outfile)
        jobutils.Job.reg(self.outfile)

    def plot(self):
        """
        Plot the data.
        """
        twinx = self.cols[1] if len(self.cols) == 2 else None
        for col in self.cols[:1] if twinx else self.cols:
            with plotutils.pyplot(inav=envutils.is_interac()) as plt:
                fig = plt.figure(figsize=(6, 4.5))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_xlabel(self.data.index.name)
                color = 'g' if twinx else 'k'
                ax1.set_ylabel(col, color=color)
                self.axPlot(ax1, col, color=color)
                if twinx:
                    ax1.tick_params(axis='y', colors='g')
                    ax2 = ax1.twinx()
                    ax2.spines['left'].set_color('g')
                    self.axPlot(ax2,
                                twinx,
                                color='b',
                                marker='o',
                                markerfacecolor='none',
                                alpha=0.9)
                    ax2.set_ylabel(twinx, color='b')
                    ax2.tick_params(axis='y', colors='b')
                    ax2.spines['right'].set_color('b')
                fig.tight_layout()
                self.save(fig, col=not twinx and col)

    @property
    def cols(self):
        """
        Get the task columns.

        :return list: task columns.
        """
        return self.data.columns

    def axPlot(self,
               ax,
               col,
               color='k',
               marker='*',
               linestyle='--',
               markerfacecolor=None,
               **kwargs):
        """
        Plot on ax.

        :param ax matplotlib.axes.Axes: the ax to plot on.
        :param col str: the column name of y-axis data.
        :param color str: the color of the scatter, line, and marker.
        :param marker str: scatter marker.
        :param linestyle str: the linestyle.
        :param markerfacecolor str: the markerfacecolor.
        """
        ax.plot(self.data.index,
                self.data[col],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=markerfacecolor,
                **kwargs)

    def save(self, fig, col=None):
        """
        Save the figure.

        :param fig matplotlib.figure.Figure: the figure to save.
        :param col str: the column name of y-axis data.
        """
        outfile = f"{self.name}{f'_{col.split()[0].lower()}' if col else ''}.svg"
        fig.savefig(outfile)
        jobutils.Job.reg(outfile)
        self.figs.append(fig)


class Collect(CollBase):
    """
    The class to collect the log files and plot the requested data.
    """
    DROPNA = 'dropna'
    KEYS = {DROPNA}
    TASK_TIME = 'task_time'
    TIME_MIN = 'Task Time (min)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.args:
            self.args = [self.TASK_TIME]
        self.kwargs.setdefault(self.DROPNA, True)

    def set(self, finished='finished', func=lambda x: x.total_seconds() / 60):
        """
        Set the data from the log files.

        :param func 'func': map the task time from seconds to minutes.
        """
        self.data = logutils.Reader.collect(*self.args, **self.kwargs)
        if self.data.empty:
            self.error(f"Empty data collected ({' '.join(self.args)}).")
        if not self.kwargs[self.DROPNA] and self.data.isna().any().any():
            self.error(self.data[self.data.isna().any(axis=1)].to_markdown())
        self.data.rename(columns={'memory': self.MEMORY_GB}, inplace=True)
        if self.MEMORY_GB in self.data.columns:
            self.data[self.MEMORY_GB] *= constants.MB_TO_GB
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


class Merge(CollBase):
    """
    The class to merge collected files.
    """

    def __init__(self, *args, file=f"{Collect.name}.csv", **kwargs):
        super().__init__(*args, **kwargs)
        self.file = file
        if not self.args:
            self.args = ['mac', 'ubuntu']

    @property
    def cols(self):
        """
        See parent.
        """
        columns = [x[0] for x in self.data.columns]
        return sorted(set(columns), key=lambda x: columns.index(x))

    def axPlot(self, ax1, col, markers="*osv^<>12348pP*hH+xXDd|_", **kwargs):
        """
        See parent.

        :param excluded tuple: the column to exclude.
        """
        for column, marker in zip(self.data.columns, markers):
            if column[0] != col:
                continue
            kwargs = {**kwargs, **dict(marker=marker, label=column[1])}
            super().axPlot(ax1, column, **kwargs)

    def save(self, fig, **kwargs):
        """
        See parent.
        """
        fig.legend(loc='lower right', bbox_to_anchor=(0.85, 0.15))
        super().save(fig, **kwargs)


class Main(logutils.Base):
    """
    Main class to run the check module as a script.
    """
    CLASSES = (Exist, Glob, Has, Cmp, Collect, Merge)

    def __init__(self, args, **kwargs):
        """
        :param args list: command line arguments without the script name.
        """
        super().__init__(**kwargs)
        self.args = args
        self.Class = (x for x in self.CLASSES if x.name == self.args[0])
        self.Class = next(self.Class, None)

    def run(self, rex=re.compile(r'(\w*)=(.*)')):
        """
        Run the check.

        :param rex `re.compile`: regular expression to match keyword arguments.
        """
        if not self.Class:
            self.error(f"{self.args[0]} found. Please select from "
                       f"{', '.join([x.name for x in self.CLASSES])}.\n")
        matches = [rex.match(x) for x in self.args[1:]]
        args = [x.strip() for x, y in zip(self.args[1:], matches) if y is None]
        kwargs = dict([[y.strip() for y in x.groups()] for x in matches if x])
        self.Class(*args, **kwargs).run()


if __name__ == "__main__":
    Main(sys.argv[1:]).run()
