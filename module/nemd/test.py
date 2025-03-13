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
import shlex
import sys

import numpy as np
import pandas as pd

from nemd import envutils
from nemd import jobutils
from nemd import lammpsdata
from nemd import logutils
from nemd import plotutils
from nemd import process
from nemd import symbols
from nemd import timeutils

CMD = 'cmd'
CHECK = 'check'
TAG = 'tag'

FILE_RE = re.compile('.* +(.*)_(driver|workflow).py( +.*)?$')
NAME_BRKT_RE = re.compile('(?:;|&&|\|\|)?(\w+)\\((.*?)\\)')


class Cmd:
    """
    The class to parse a cmd test file.
    """

    NAME = 'cmd'
    POUND = symbols.POUND

    def __init__(self, dir=None, job=None, options=None):
        """
        :param dir str: the path containing the file
        :param job 'signac.contrib.job.Job': the signac job instance
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        """
        self.dir = dir
        self.job = job
        self.options = options
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
    JOBNAME_RE = re.compile(f'{jobutils.FLAG_JOBNAME} +(\w+)')

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
        header = self.raw_args[0]
        if header.startswith(symbols.POUND):
            return header.strip(symbols.POUND).strip()
        if not self.cmd:
            return
        match = re.search(f'-(\w*) \{self.PARAM}', self.cmd.args[0])
        name = match.group(1).replace('_', ' ') if match else self.NAME
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
        params = Tag(job=self.job).slowParams(slow=self.options.slow)
        return [x for x in args if x in params]

    def getCmds(self):
        """
        Get the parameterized commands.

        :return list: each value is one command.
        """
        if not self.cmd:
            return
        cmd = self.cmd.args[0]
        if self.PARAM not in cmd or not self.args:
            return self.cmd.args
        match = self.JOBNAME_RE.search(cmd)
        if match:
            name = match.group(1)
            cmd = cmd.replace(cmd[slice(*match.span())], '')
        else:
            name = FILE_RE.match(cmd).group(1)
        cmd += f' {jobutils.FLAG_NAME} {name}'
        cmds = [cmd.replace(self.PARAM, x) for x in self.args]
        names = [x.replace(' ', '_') for x in self.args]
        flag_names = [f'{jobutils.FLAG_JOBNAME} {name}_{x}' for x in names]
        return [f'{x} {y}' for x, y in zip(cmds, flag_names)]


class Exist:
    """
    The class to perform file existence check.
    """

    def __init__(self, *args):
        """
        :param args str: the target filenames
        """
        self.args = args

    def run(self):
        """
        The main method to check the existence of files.

        :self.error: if file doesn't exist
        """
        for target in self.args:
            if os.path.isfile(target):
                continue
            self.error(f"{target} not found")

    @classmethod
    def getTokens(cls, values, **kwargs):
        """
        Get the cmd tokens.

        :param values str: generate token from the input values.
        :return list: token list
        """
        return [x.strip('\'\" ') for x in values.split(',')]

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

    @classmethod
    def getTokens(cls, values, job=None):
        """
        Get the cmd tokens.

        :param values str: generate token from the input values.
        :param job `signac.job.Job`: the check job.
        :return list: token list
        """
        params = super().getTokens(values)
        pathname = os.path.join(job.statepoint[jobutils.FLAG_DIR], params[0])
        return [pathname] + params[1:]

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
            self.raiseError(', '.join(self.args))

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
        super().__init__(*args)
        self.kwargs = kwargs
        self.data = None
        self.name = os.path.basename(self.args[-1])

    @classmethod
    def getTokens(cls, values, job=None):
        """
        Get the cmd tokens.

        :param values str: generate token from the input values.
        :param job `signac.job.Job`: the check job.
        :return list: token list
        """
        tokens = [x.strip('\'\" ') for x in values.split(',')]
        if not tokens:
            tokens = [cls.TIME]
        jobs = jobutils.Job(job).getJobs()
        files = [f"{x.jobname}={x.logfile}" for x in jobs]
        return tokens + [job.statepoint[jobutils.FLAG_DIR]] + files

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
        files = {x: y for x, y in self.kwargs.items() if os.path.exists(y)}
        rdrs = [logutils.Reader(x) for x in files.values()]
        names = [x.options.NAME for x in rdrs]
        params = [x.removeprefix(y)[1:] for x, y in zip(files.keys(), names)]
        index = pd.Index(params, name=Param(dir=self.args[-1]).label)
        data = {}
        if self.TIME in self.args:
            data[self.TIME_LB] = [x.task_time for x in rdrs]
        if self.MEMORY in self.args:
            data[self.MEMORY_LB] = [x.memory for x in rdrs]
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
        for key in self.args[:-1]:
            if key == self.MEMORY and self.TIME in self.args:
                # memory and time are plotted together when key == self.TIME
                continue
            label = self.LABELS[key]
            with plotutils.get_pyplot(inav=envutils.is_interac()) as plt:
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
                    key = f"{self.TIME}_{self.MEMORY}"
                fig.tight_layout()
                out_png = f"{self.name}_{key}{self.PNG_EXT}"
                fig.savefig(out_png)
                jobutils.add_outfile(out_png)


class CheckError(Exception):
    """
    Raised when the results are unexpected.
    """
    pass


class Check(Cmd):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    NAME = 'check'
    Class = {
        'exist': Exist,
        'glob': Glob,
        'in': In,
        'cmp': Cmp,
        'collect_log': CollectLog
    }

    def run(self):
        """
        Check the results by execute all operators. Raise errors if any failed.

        :raise CheckError: if check failed.
        """
        jobname = os.path.basename(self.dir)
        print(f"{jobname}: {'; '.join(self.args)}")
        proc = Process(tokens=list(self.tokens), jobname=jobname)
        completed = proc.run()
        if not completed.returncode:
            return
        with open(proc.logfile) as fh:
            raise CheckError(fh.read())

    @property
    def tokens(self):
        """
        The args to build the command from.

        :return list: each value is a one-line command
        """
        for line in self.args:
            token = line
            for match in NAME_BRKT_RE.finditer(line):
                name, values = match.groups()
                params = Check.Class[name].getTokens(values, job=self.job)
                shell_cmd = shlex.join(['nemd_check', name] + params)
                python_cmd = line[match.span(1)[0]:match.span()[-1]]
                token = token.replace(python_cmd, shell_cmd)
            yield token


class Process(process.Base):
    """
    Sub process to run check cmd.
    """

    NAME = Check.NAME
    SEP = ';\n'


class Tag(Cmd):
    """
    The class parses and interprets the tag file. The class also creates a new
    tag file (or update existing one).
    """

    NAME = 'tag'
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

    def setOperators(self):
        """
        Parse the one line command to get the operators.
        """
        if self.args is None:
            return
        for match in NAME_BRKT_RE.finditer(' '.join(self.args)):
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
        for job in jobutils.Job(job=self.job).getJobs():
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
        return Param(job=self.job, options=self.options).args

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


if __name__ == "__main__":
    args, kwargs = [], {}
    for val in sys.argv[2:]:
        match = re.match("(.*)=(.*)", val)
        if match:
            kwargs[match.group(1).strip()] = match.group(2).strip()
        else:
            args.append(val)
    Check.Class[sys.argv[1]](*args, **kwargs).run()
