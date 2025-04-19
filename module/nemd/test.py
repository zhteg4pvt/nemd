# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides test related classes to parse files for command, parameter,
checking, and labels.
"""
import datetime
import filecmp
import functools
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from nemd import envutils
from nemd import jobutils
from nemd import lammpsdata
from nemd import logutils
from nemd import objectutils
from nemd import plotutils
from nemd import process
from nemd import symbols
from nemd import timeutils


class Cmd(objectutils.Object):
    """
    The class to parse a cmd test file.
    """

    POUND = symbols.POUND

    def __init__(self, dir=None, options=None):
        """
        :param dir str: the path containing the file
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        """
        self.dir = dir
        self.options = options
        self.pathname = os.path.join(self.dir, self.name)
        self.jobname = os.path.basename(self.dir)

    @property
    @functools.cache
    def raw(self):
        """
        The raw string by reading the file.
        """
        if not os.path.isfile(self.pathname):
            return
        with open(self.pathname) as fh:
            return [x.strip() for x in fh.readlines() if x.strip()]

    @property
    @functools.cache
    def args(self):
        """
        Return arguments out of the raw.

        :return list of str: each str is a command
        """
        if self.raw is None:
            return
        return [x for x in self.raw if not x.startswith(self.POUND)]

    @property
    @functools.cache
    def header(self):
        """
        Return the header out of the raw.

        :return str: the header
        """
        header = f"# {os.path.basename(self.dir)}"
        cmts = symbols.SPACE.join(self.cmts)
        return f"{header}: {cmts}" if cmts else header

    @property
    def cmts(self):
        """
        Return the comment out of the raw.

        :return generator: the comments
        """
        if self.raw is None:
            return
        for line in self.raw:
            if not line.startswith(self.POUND):
                return
            yield line.strip(self.POUND).strip()


class Param(Cmd):
    """
    The class to parse the parameter file.
    """

    PARAM = f'$param'

    def __init__(self, *args, cmd=None, **kwargs):
        """
        :param cmd `Cmd`: the Cmd instance parsing the cmd file.
        """
        super().__init__(*args, **kwargs)
        self.cmd = cmd

    @property
    @functools.cache
    def label(self):
        """
        Get the xlabel from the header of the parameter file.

        :return str: The xlabel.
        """
        label = next(self.cmts, None)
        if not label and self.cmd:
            match = re.search(f'-(\w*) \{self.PARAM}', self.cmd.args[0])
            name = match.group(1).replace('_', ' ') if match else self.name
            label = ' '.join([x.capitalize() for x in name.split()])
            with open(self.pathname, 'w') as fh:
                fh.write('\n'.join([f"# {label}"] + self.args))
        return label

    @property
    @functools.cache
    def args(self):
        """
        The arguments from the file filtered by the slow.

        :return list: the arguments filtered by slow.
        """
        args = super().args
        if args is None:
            return
        if self.options.slow is None:
            return args
        params = Tag(dir=self.dir).slowParams(slow=self.options.slow)
        return [x for x in args if x in params]

    def getCmds(self,
                jobname_re=re.compile(f'{jobutils.FLAG_JOBNAME} +\w+'),
                script_re=re.compile('.* +(.*)_(driver|workflow).py( +.*|$)')):
        """
        Get the parameterized commands.

        :param jobname_re `re.compile`: the jobname regular express
        :param script_re `re.compile`:the script regular express
        :return list: each value is one command.
        """
        if not self.args or self.PARAM not in self.cmd.args[0]:
            return self.cmd.args
        name = self.label or script_re.match(self.cmd.args[0]).group(1)
        name = name.lower().replace(' ', '_')
        cmd = jobname_re.sub('', self.cmd.args[0])
        cmd = f"{cmd} {jobutils.FLAG_NAME} {name}"
        return [
            f'{cmd.replace(self.PARAM, x)} {jobutils.FLAG_JOBNAME} {name}_{x}'
            for x in self.args
        ]


class Exist(objectutils.Object):
    """
    The class to perform file existence check.
    """

    def __init__(self, *args):
        """
        :param args str: the target filenames
        """
        self.args = args
        self.setUp()

    def run(self):
        """
        The main method to check the existence of files.
        """
        for target in self.args:
            if os.path.isfile(target):
                continue
            self.error(f"{target} not found")

    def setUp(self):
        """
        Get the cmd tokens.
        """
        self.args = [x.strip() for x in self.args]

    def error(self, msg):
        """
        Print he message and exit with 1.

        :param msg str: the message to print
        """
        print(msg)
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

        :self.error: the number of found files is not the expected
        """
        for target in self.args:
            files = glob.glob(target)
            if self.num is None and len(files) > 0:
                continue
            if self.num and len(files) == self.num:
                continue
            self.error(f"{files} found for {target} (num={self.num})")


class In(Exist):
    """
    The class to check the containing file strings.
    """

    def __init__(self, *args):
        super().__init__(args[-1])
        self.strs = args[:-1]

    def run(self):
        """
        The main method to check the containing file strings.
        """
        super().run()
        with open(self.args[-1]) as fh:
            file_str = fh.read()
        for content in self.strs:
            if content in file_str:
                continue
            self.error(f"{content} not found in {self.args[-1]}")


class Cmp(Exist):
    """
    The class to perform file comparison.
    """

    def __init__(self, *args, atol=None, rtol=None, equal_nan=None, **kwargs):
        """
        :param atol str: the absolute tolerance parameter for numpy.isclose
        :param rtol str: the relative tolerance parameter for numpy.isclose
        :param equal_nan bool: whether to compare NaNs as equal.
        """
        super().__init__(*args, **kwargs)
        self.atol = atol
        self.rtol = rtol
        self.equal_nan = equal_nan
        self.dir = dir

    def run(self):
        """
        The main method to compare files.
        """
        super().run()
        self.cmpFile()
        self.cmpCsv()
        self.cmpData()

    def cmpFile(self):
        """
        Compare the file content.
        """
        if not all(x is None for x in [self.atol, self.rtol, self.equal_nan]):
            return
        # Exact Match
        if not filecmp.cmp(*self.args):
            self.raiseError(', '.join(self.args[1:]))

    def cmpCsv(self):
        """
        Compare csv files via np.allclose.
        """
        if not all(x.endswith('.csv') for x in self.args):
            return
        origin = pd.read_csv(self.args[0])
        object = origin.select_dtypes(include='object')
        nonobj = origin.select_dtypes(exclude='object')
        for target in self.args[1:]:
            data = pd.read_csv(target)
            tgt_obj = data.select_dtypes(include='object')
            if object.shape != tgt_obj.shape:
                self.raiseError(target)
            if not all(object == tgt_obj):
                self.raiseError(target)
            tgt_nonobj = data.select_dtypes(exclude='object')
            if nonobj.shape != tgt_nonobj.shape:
                self.raiseError(target)
            if not np.allclose(nonobj, tgt_nonobj, **self.kwargs):
                self.raiseError(target)

    @property
    @functools.cache
    def kwargs(self, atol=1e-08, rtol=1e-05, equal_nan=True):
        """
        Set the parameters for the comparison of csv files (e.g., tolerance).

        :param atol str: the absolute tolerance parameter for numpy.isclose
        :param rtol str: the relative tolerance parameter for numpy.isclose
        :param equal_nan bool: whether to compare NaNs as equal.
        :return dict: the parameters for the comparison of csv files.
        """
        equal = equal_nan if self.equal_nan is None else eval(self.equal_nan)
        atol = atol if self.atol is None else float(self.atol)
        rtol = rtol if self.rtol is None else float(self.rtol)
        return dict(atol=atol, rtol=rtol, equal_nan=equal)

    def cmpData(self):
        """
        Compare the lammps data files.
        """
        if not all(x.endswith('.data') for x in self.args):
            return
        origin = lammpsdata.read(self.args[0])
        for target in self.args[1:]:
            data = lammpsdata.read(target)
            if not origin.allClose(data, **self.kwargs):
                self.raiseError(target)

    def raiseError(self, target):
        """
        Raise the error with proper message.

        :param target str: the target filename(s)
        :raises CheckError: raise the error message as the files are different.
        """
        self.error(f"{self.args[0]} and {target} are different.")


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
        index = pd.Index(params, name=name)
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


class CheckError(Exception):
    """
    Raised when the results are unexpected.
    """
    pass


class Check(Cmd):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    NEMD_CHECK = 'nemd_check'
    CMP = 'cmp'

    def run(self, rex=re.compile(rf'{NEMD_CHECK} +{CMP} +(\w+)')):
        """
        Check the results by execute all operators. Raise errors if any failed.

        :raise CheckError: if check failed.
        """
        print(f"# {self.jobname}: {'; '.join(self.args)}")
        rpl = rf'{self.NEMD_CHECK} {self.CMP} {self.dir}{os.path.sep}\1'
        proc = Process([rex.sub(rpl, x) for x in self.args],
                       jobname=self.jobname)
        completed = proc.run()
        if not completed.returncode:
            return
        with open(proc.logfile) as fh:
            raise CheckError(fh.read())


class Process(process.Base):
    """
    Sub process to run check cmd.
    """

    NAME = Check.name
    SEP = ';\n'


class Tag(Cmd):
    """
    The class parses and interprets the tag file. The class also creates a new
    tag file (or update existing one).
    """

    SLOW = 'slow'
    LABEL = 'label'

    def __init__(self, *args, delay=False, **kwargs):
        """
        :param delay 'bool': delay the setup if True.
        """
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.operators = []
        self.logs = []
        if self.delay:
            return
        self.setOperators()

    def setOperators(self, rex=re.compile('(?:;|&&|\|\|)?(\w+)\\((.*?)\\)')):
        """
        Parse the one line command to get the operators.
        """
        if self.args is None:
            return
        for match in rex.finditer(' '.join(self.args)):
            name, value = [x.strip("'\"") for x in match.groups()]
            values = [x.strip(" '\"") for x in value.split(symbols.COMMA)]
            self.operators.append([name] + [x for x in values if x])

    def run(self):
        """
        Main method to run.
        """
        self.setLogs()
        self.setSlow()
        self.setLabel()
        self.write()

    def setLogs(self):
        """
        Set the log readers.
        """
        for job in jobutils.Job(dir=self.dir).getJobs():
            if not job.logfile:
                continue
            self.logs.append(logutils.Reader(job.logfile))

    def setSlow(self):
        """
        Set the slow tag with the total job time from the driver log files.
        """
        times = [x.task_time for x in self.logs]
        if not self.param:
            total = sum(times, datetime.timedelta())
            job_time = timeutils.delta2str(total)
            self.set(self.SLOW, job_time)
            return
        roots = [os.path.splitext(x.namepath)[0] for x in self.logs]
        params = [x.split('_')[-1] for x in roots]
        param_time = []
        for parm in self.param:
            try:
                task_time = next(y for x, y in zip(params, times) if x == parm)
            except StopIteration:
                continue
            else:
                param_time.append([parm, task_time])
        tags = [f"{x}, {timeutils.delta2str(y)}" for x, y in param_time]
        self.setMult(self.SLOW, *tags)

    @property
    @functools.cache
    def param(self):
        """
        Return the parameters from the parameter file.

        :return list: the parameters from the parameter file.
        """
        return Param(dir=self.dir, options=self.options).args

    def set(self, key, *value):
        """
        Set the value (and the key) of one operator.

        :param key str: the key to be set
        :param value tuple of str: the value(s) to be set
        """
        try:
            idx = next(i for i, x in enumerate(self.operators) if x[0] == key)
        except StopIteration:
            self.operators.append([key, *value])
        else:
            self.operators[idx] = [key, *value]

    def setMult(self, key, *values):
        """
        Set the values (and the key) of multiple operators.

        :param key str: the key to be set
        :param values tuple of list: each list contains the value(s) of one
            operator.
        """
        self.operators = [x for x in self.operators if x[0] != key]
        for value in values:
            self.operators.append([self.SLOW, value])

    def setLabel(self):
        """
        Set the label of the job.
        """
        labels = self.get(self.LABEL, [])
        labels += [x.options.NAME for x in self.logs]
        if not labels:
            return
        self.set(self.LABEL, *set(labels))

    def get(self, key, default=None):
        """
        Get the value of a specific key.

        :param key str: the key to be searched
        :param default str: the default value if the key is not found
        :return tuple of str: the value(s)
        """
        for name, *value in self.operators:
            if name == key:
                return value
        return tuple() if default is None else default

    def write(self):
        """
        Write the tag file.
        """
        print(f"# {self.jobname}: {symbols.COMMA.join(self.args)}")
        with open(self.pathname, 'w') as fh:
            for key, *value in self.operators:
                values = symbols.COMMA.join(value)
                fh.write(f"{key}({values})\n")

    def selected(self):
        """
        Select the operators by the options.

        :return bool: Whether the test is selected.
        """
        return all([not self.slow(), self.labeled()])

    def slow(self):
        """
        Whether the test is slow.

        :return bool: Whether the test is slow.
        """
        if self.options.slow is None or not self.get(self.SLOW):
            return False
        params = self.slowParams()
        return not params

    def slowParams(self, slow=None):
        """
        Get the slow parameters filtered by slow.

        :param slow float: above this value is slow
        :return list of str: parameters filtered by slow
        """
        if slow is None:
            slow = self.options.slow
        # [['slow', '00:00:01']]
        # [['slow', '1', '00:00:01'], ['slow', '9', '00:00:04']]
        params = [x[1:] for x in self.operators if x[0] == self.SLOW]
        time = [timeutils.str2delta(x[-1]).total_seconds() for x in params]
        return [x[0] for x, y in zip(params, time) if y <= slow]

    def labeled(self):
        """
        Whether the test is labeled with the specified labels.

        :return bool: Whether the test is labeled.
        """
        if self.options.label is None:
            return True
        for tagged_label in self.get(self.LABEL, []):
            for label in self.options.label:
                if tagged_label.startswith(label):
                    return True
        return False


TASKS = [Cmd, Check, Tag]

if __name__ == "__main__":
    """
    Run library module as a script.
    """
    Classes = [Exist, Glob, In, Cmp, CollectLog]
    try:
        Class = next(x for x in Classes if x.name == sys.argv[1])
    except StopIteration:
        sys.exit(f'Please choose from {[x.name for x in Classes]}')
    kwargs_re = re.compile(r'(.*)=(.*)')
    matches = [kwargs_re.match(x) for x in sys.argv[2:]]
    args = [x for x, y in zip(sys.argv[2:], matches) if y is None]
    kwargs = [[y.strip() for y in x.groups()] for x in matches if x]
    Class(*args, **dict(kwargs)).run()
