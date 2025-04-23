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
from nemd import lammpsdata
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
            self.error(target)

    def error(self, target):
        """
        See parent.
        """
        super().error(f"{self.args[0]} is different from {target}.")

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
                self.error(target)
            if not all(object == tgt_obj):
                self.error(target)
            tgt_nonobj = data.select_dtypes(exclude='object')
            if nonobj.shape != tgt_nonobj.shape:
                self.error(target)
            if not np.allclose(nonobj, tgt_nonobj, **self.kwargs):
                self.error(target)

    def data(self):
        """
        Compare the lammps data files.
        """
        if not self.keys or not all(x.endswith('.data') for x in self.args):
            return
        origin = lammpsdata.read(self.args[0])
        for target in self.args[1:]:
            if origin.allClose(lammpsdata.read(target), **self.kwargs):
                continue
            self.error(target)


class CollectLog(Exist):
    """
    The class to collect the log files and plot the requested data.
    """

    TIME = 'time'
    MEMORY = 'memory'
    CSV_EXT = '.csv'
    PNG_EXT = '.png'
    TIME_LB = f'{TIME.capitalize()} (min)'
    MEMORY_LB = f'{MEMORY.capitalize()} (MB)'
    LABELS = {TIME: TIME_LB, MEMORY: MEMORY_LB}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.outfile = f"{self.name}{self.PNG_EXT}"
        self.logs = {
            x.jobname: x.logfile
            for x in jobutils.Job().getJobs() if x.logfile
        }

    def run(self):
        """
        Main method to run.
        """
        self.set()
        self.plotData()

    def set(self):
        """
        Set the time and memory data from the log files.
        """
        files = {x: y for x, y in self.logs.items() if os.path.exists(y)}
        rdrs = [logutils.Reader(x) for x in files.values()]
        data = {}
        if self.TIME in self.args:
            data[self.TIME_LB] = [x.task_time for x in rdrs]
        if self.MEMORY in self.args:
            data[self.MEMORY_LB] = [x.memory for x in rdrs]
        name = rdrs[0].options.NAME
        params = [x.removeprefix(name)[1:] for x in files.keys()]
        index = pd.Index(params, name=name.replace('_', ' '))
        self.data = pd.DataFrame(data, index=index)
        self.data.set_index(self.data.index.astype(float), inplace=True)
        func = lambda x: x.total_seconds() / 60. if x is not None else None
        self.data[self.TIME_LB] = self.data[self.TIME_LB].map(func)
        out_csv = f"{self.name}{self.CSV_EXT}"
        self.data.to_csv(out_csv)
        jobutils.add_outfile(out_csv)

    def plotData(self):
        """
        Plot the data. Time and memory are plotted together if possible.
        """
        for key in self.args:
            if key == self.MEMORY and self.TIME in self.args:
                # memory and time are plotted together when key == self.TIME
                continue
            label = self.LABELS[key]
            with plotutils.pyplot(inav=envutils.is_interac()) as plt:
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(1, 1, 1)
                data = self.data.get(label)
                twinx = key == self.TIME and self.MEMORY in self.args and not self.data[
                    self.MEMORY_LB].isnull().all()
                color = 'g' if twinx else 'k'
                ax1.plot(self.data.index, data, f'{color}-.*')
                ax1.set_xlabel(self.data.index.name)
                ax1.set_ylabel(key, color=color)
                if twinx:
                    ax1.tick_params(axis='y', colors='g')
                    ax2 = ax1.twinx()
                    ax2.spines['left'].set_color('g')
                    ax2.plot(self.data.index,
                             self.data[self.MEMORY_LB],
                             'b--o',
                             markerfacecolor='none',
                             alpha=0.9)
                    ax2.set_ylabel(self.MEMORY_LB, color='b')
                    ax2.tick_params(axis='y', colors='b')
                    ax2.spines['right'].set_color('b')
                fig.tight_layout()
                fig.savefig(self.outfile)
                jobutils.add_outfile(self.outfile)


if __name__ == "__main__":
    """
    Run library module as a script.
    """
    Classes = [Exist, Glob, Has, Cmp, CollectLog]
    try:
        Class = next(x for x in Classes if x.name == sys.argv[1])
    except StopIteration:
        sys.exit(f'{sys.argv[1]} found. ({[x.name for x in Classes]})')
    kwargs_re = re.compile(r'(.*)=(.*)')
    matches = [kwargs_re.match(x) for x in sys.argv[2:]]
    args = [x.strip() for x, y in zip(sys.argv[2:], matches) if y is None]
    kwargs = [[y.strip() for y in x.groups()] for x in matches if x]
    Class(*args, **dict(kwargs)).run()
