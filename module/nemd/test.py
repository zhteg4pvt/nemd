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
import json
import os
import re
import types

import numpy as np
import pandas as pd

from nemd import envutils
from nemd import jobutils
from nemd import lammpsdata
from nemd import logutils
from nemd import plotutils
from nemd import symbols
from nemd import timeutils

CMD = 'cmd'
CHECK = 'check'
TAG = 'tag'


class Cmd:
    """
    The class to parse a cmd test file.
    """

    NAME = 'cmd'
    POUND = symbols.POUND
    NAME_BRACKET_RE = '(?:(?:^(?: +)?)|(?: +))(.*?)\\((.*?)\\)'
    AND_NAME_RE = re.compile('^and +(.*)')

    def __init__(self, dir=None, job=None):
        """
        :param dir str: the path containing the file
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.dir = dir
        self.job = job
        if self.dir is None:
            self.dir = self.job.statepoint[jobutils.FLAG_DIR]
        self.pathname = os.path.join(self.dir, self.NAME)

    @property
    @functools.cache
    def raw_args(self):
        """
        Set arguments by reading the file.
        """
        if not os.path.isfile(self.pathname):
            return
        with open(self.pathname) as fh:
            return [x.strip() for x in fh.readlines() if x.strip()]

    @property
    @functools.cache
    def args(self):
        """
        Return arguments out of the raw_args.

        :return list of str: each str is a command
        """
        if self.raw_args is None:
            return
        return [x for x in self.raw_args if not x.startswith(self.POUND)]

    @property
    @functools.cache
    def comment(self):
        """
        Return the comment out of the raw_args.

        :return str: the comment
        """
        if self.raw_args is None:
            return
        comments = []
        for line in self.raw_args:
            if not line.startswith(self.POUND):
                break
            comments.append(line.strip(self.POUND).strip())
        return symbols.SPACE.join(comments)


class Param(Cmd):
    """
    The class to parse the parameter file.
    """

    NAME = 'param'
    PARAM = f'${NAME}'

    def __init__(self, *args, cmd=None, **kwargs):
        """
        :param cmd `Cmd`: the Cmd instance parsing the cmd file.
        """
        super().__init__(*args, **kwargs)
        self.cmd = cmd

    def setXlabel(self):
        """
        Set the xlabel of the parameter file.
        """
        if self.raw_args is None or self.xlabel:
            return
        Param.xlabel.fget.cache_clear()  # clear previous loaded label
        match = re.search(f'-(\w*) \{self.PARAM}', self.cmd.args[0])
        name = match.group(1).replace('_', ' ') if match else self.NAME
        header = f"# {' '.join([x.capitalize() for x in name.split()])}"
        with open(self.pathname, 'w') as fh:
            fh.write('\n'.join([header] + self.args))
        self.raw_args.insert(0, header)

    @property
    @functools.cache
    def xlabel(self):
        """
        Get the xlabel from the header of the parameter file.

        :return str: The xlabel.
        """
        header = self.raw_args[0]
        if header.startswith(symbols.POUND):
            return header.strip(symbols.POUND).strip()


class Base:

    def __init__(self, *args, job=None):
        """
        :param args str: the target filenames
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        self.args = args
        self.job = job


class Exist(Base):
    """
    The class to perform file existence check.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.job is None:
            return
        self.targets = [self.job.fn(x) for x in self.args]

    def run(self):
        """
        The main method to check the existence of files.
        """
        for target in self.targets:
            if os.path.isfile(target):
                continue
            raise FileNotFoundError(f"{self.job.fn(target)} not found")


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
        for target in self.targets:
            files = glob.glob(target)
            if self.num is None and len(files) > 0:
                continue
            if self.num and len(files) == self.num:
                continue
            raise ValueError(f"{files} found for {target} (num={self.num})")


class NotExist(Exist):
    """
    The class to perform file non-existence check.
    """

    def run(self):
        """
        The main method to check the existence of a file.
        """
        for target in self.targets:
            if not os.path.isfile(target):
                continue
            raise FileNotFoundError(f"{self.job.fn(target)} found")


class In(Exist):
    """
    The class to check the containing file strings.
    """

    def __init__(self, *args, job=None):
        super().__init__(args[-1], job=job)
        self.strs = args[:-1]

    def run(self):
        """
        The main method to check the containing file strings.
        """
        super().run()
        with open(self.targets[0]) as fh:
            file_str = fh.read()
        for content in self.strs:
            if content in file_str:
                continue
            raise ValueError(f"{content} not found in {self.targets[0]}")


class Cmp(Exist):
    """
    The class to perform file comparison.
    """

    def __init__(self,
                 original,
                 target,
                 atol=None,
                 rtol=None,
                 equal_nan=None,
                 job=None):
        """
        :param original str: the original filename
        :param target str: the target filename
        :param atol str: the absolute tolerance parameter for numpy.isclose
        :param rtol str: the relative tolerance parameter for numpy.isclose
        :param equal_nan bool: whether to compare NaNs as equal.
        :param job 'signac.contrib.job.Job': the signac job instance
        """
        super().__init__(target, job=job)
        self.original = original
        self.atol = atol
        self.rtol = rtol
        self.equal_nan = equal_nan
        if self.job is None:
            return
        pathname = os.path.join(self.job.statepoint[jobutils.FLAG_DIR],
                                self.original)
        self.targets.insert(0, pathname)

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
        if not filecmp.cmp(*self.targets):
            self.raiseError(', '.join(self.targets[1:]))

    def cmpCsv(self):
        """
        Compare csv files via np.allclose.
        """
        if not all(x.endswith('.csv') for x in self.targets):
            return
        origin = pd.read_csv(self.targets[0])
        object = origin.select_dtypes(include='object')
        nonobj = origin.select_dtypes(exclude='object')
        for target in self.targets[1:]:
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
        if not all(x.endswith('.data') for x in self.targets):
            return
        origin = lammpsdata.read(self.targets[0])
        for target in self.targets[1:]:
            data = lammpsdata.read(target)
            if not origin.allClose(data, **self.kwargs):
                self.raiseError(target)

    def raiseError(self, target):
        """
        Raise the error with proper message.

        :param target str: the target filename(s)
        :raises ValueError: raise the error message as the files are different.
        """
        raise ValueError(f"{self.targets[0]} and {target} are different.")


class CollectLog(Base):
    """
    The class to collect the log files and plot the requested data.
    """
    TIME = 'time'
    MEMORY = 'memory'
    CSV_EXT = '.csv'
    PNG_EXT = '.png'
    TIME_LB = f'{TIME.capitalize()} (min)'
    MEMORY_LB = f'{MEMORY.capitalize()} ({logutils.MEMORY_UNIT})'
    LABELS = {TIME: TIME_LB, MEMORY: MEMORY_LB}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = os.path.basename(self.job.statepoint[jobutils.FLAG_DIR])
        self.outfile = f"{self.name}{self.CSV_EXT}"
        if self.args is None:
            self.args = [self.TIME]

    def run(self):
        """
        Main method to run.
        """
        self.setData()
        self.plotData()

    def setData(self):
        """
        Set the time and memory data from the log files.
        """
        files = self.job.doc[jobutils.LOGFILE]
        files = {x: self.job.fn(y) for x, y in files.items()}
        files = {x: y for x, y in files.items() if os.path.exists(y)}
        rdrs = [logutils.LogReader(x) for x in files.values()]
        names = [x.options.NAME for x in rdrs]
        params = [x.removeprefix(y) for x, y in zip(files.keys(), names)]
        index = pd.Index(params, name=Param(job=self.job).xlabel)
        data = {}
        if self.TIME in self.args:
            data[self.TIME_LB] = [x.task_time for x in rdrs]
        if self.MEMORY in self.args:
            data[self.MEMORY_LB] = [x.memory for x in rdrs]
        self.data = pd.DataFrame(data, index=index)
        self.data.set_index(self.data.index.astype(float), inplace=True)
        func = lambda x: x.total_seconds() / 60. if x is not None else None
        self.data[self.TIME_LB] = self.data[self.TIME_LB].map(func)
        self.data.to_csv(self.outfile)
        jobutils.add_outfile(self.outfile, file=True)

    def plotData(self):
        """
        Plot the data. Time and memory are plotted together if possible.
        """
        for key in self.args:
            if key == self.MEMORY and self.TIME in self.args:
                # memory and time are plotted together when key == self.TIME
                continue
            label = self.LABELS[key]
            with plotutils.get_pyplot(inav=envutils.is_interactive()) as plt:
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(1, 1, 1)
                data = self.data.get(label)
                twinx = key == self.TIME and self.MEMORY in self.args
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
                    key = f"{self.TIME}_{self.MEMORY}"
                fig.tight_layout()
                outfile = f"{self.name}_{key}{self.PNG_EXT}"
                fig.savefig(outfile)
                jobutils.add_outfile(outfile)


class Opr(Cmd):
    """
    The class sets the operators in addition to the parsing a file.
    """

    NAME = 'opr'

    def __init__(self, *args, delay=False, **kwargs):
        """
        :param delay 'bool': read, parse, and set the operators if False
        """
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.operators = []
        if self.delay:
            return
        self.setOperators()

    def setOperators(self):
        """
        Parse the one line command to get the operators.
        """
        if self.args is None:
            return
        for match in re.finditer(self.NAME_BRACKET_RE, ' '.join(self.args)):
            name, value = [x.strip("'\"") for x in match.groups()]
            match = self.AND_NAME_RE.match(name)
            if match:
                name = match.groups()[0]
            values = [x.strip(" '\"") for x in value.split(symbols.COMMA)]
            self.operators.append([name] + [x for x in values if x])


class Check(Opr):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    NAME = 'check'
    CHECK = {
        'cmp': Cmp,
        'glob': Glob,
        'exist': Exist,
        'not_exist': NotExist,
        'in': In,
        'collect_log': CollectLog
    }
    ERRORS = (FileNotFoundError, KeyError, ValueError)

    def check(self):
        """
        Check the results by execute all operators. Raise errors if any failed.
        """
        name = os.path.basename(os.path.dirname(self.pathname))
        ops = [symbols.SPACE.join(x) for x in self.operators]
        print(f"# {name}: Checking {symbols.COMMA.join(ops)}")
        for operator in self.operators:
            CheckClass = self.getClass(operator)
            args, kwargs = self.getArg(operator)
            runner = CheckClass(*args, **kwargs, job=self.job)
            runner.run()

    def getClass(self, operator):
        """
        Get the command class.

        :param operator list of str: the operator to be executed.
            For example, [name, arg1, arg2, arg3=val, arg4=val2, ...]
        :return: the class to perform check operation.
        :rtype: 'type'
        """
        name = operator[0]
        try:
            return self.CHECK[name]
        except KeyError:
            raise KeyError(
                f'{name} is one unknown command. Please select from '
                f'{self.CHECK.keys()}')

    def getArg(self, operator):
        """
        Get the args and kwargs from the operator.

        :param operator list of str: the operator to be executed.
            For example, [name, arg1, arg2, arg3=val, arg4=val2, ...]
        :return: the args and kwargs to perform check operation.
        :rtype: list, dict
        """
        args, kwargs = [], []
        for val in operator[1:]:
            match = re.match("(.*)=(.*)", val)
            if match:
                kwargs.append(tuple(x.strip() for x in match.groups()))
            else:
                args.append(val)
        return args, dict(kwargs)


class Tag(Opr):
    """
    The class parses and interprets the tag file. The class also creates a new
    tag file (or update existing one).
    """

    NAME = 'tag'
    SLOW = 'slow'
    LABEL = 'label'

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, **kwargs)
        self.options = options
        self.logs = []

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
        logfiles = self.job.doc.get(jobutils.LOGFILE)
        if logfiles is None:
            return
        for logfile in logfiles.values():
            self.logs.append(logutils.LogReader(self.job.fn(logfile)))

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
        roots = [os.path.splitext(x.filepath)[0] for x in self.logs]
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
        return Param(job=self.job).args

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
        ops = [f"{x[0]}({symbols.COMMA.join(x[1:])})" for x in self.operators]
        name = os.path.basename(os.path.dirname(self.pathname))
        print(f"# {name}: Tagging {symbols.COMMA.join(ops)}")
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
        if self.options.slow is None:
            return False
        values = self.get(self.SLOW, ['00:00'])
        if len(values) != 1:
            return
        delta = timeutils.str2delta(values[0])
        return delta.total_seconds() > self.options.slow

    def slowParam(self, threshold):
        if threshold is None:
            return []
        threshold = float(threshold)
        params = [x[1:] for x in self.operators if x[0] == self.SLOW]
        return [
            x for x, y in params
            if timeutils.str2delta(y).total_seconds() > threshold
        ]

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


class Job:
    """
    A class to mimic a signac.job.Job for testing purpose.
    """

    def __init__(self, job_dir=os.curdir):
        """
        Initialize a Job object.

        :param job_dir str: the directory of the job
        """
        self.dir = job_dir
        self.statepoint = self.load(symbols.FN_STATE_POINT)
        self.doc = self.load(symbols.FN_DOCUMENT)
        self.document = self.doc
        self.project = types.SimpleNamespace(doc={},
                                             workspace='workspace',
                                             path=os.curdir)

    def load(self, basename):
        """
        Load the json file.

        :param basename str: the file name to be loaded.
        :return dict: the loaded json dictionary.
        """
        pathname = os.path.join(self.dir, basename)
        if not os.path.isfile(pathname):
            return {}
        with open(pathname, 'r') as fh:
            return json.load(fh)

    def fn(self, filename):
        """
        Return the full path of the file in the job directory.

        :param filename str: the file name
        :return str: the full path of the file
        """
        return os.path.join(self.dir, filename) if self.dir else filename
