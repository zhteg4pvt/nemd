# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides test related classes to parse files for command, parameter,
checking, and labels.
"""
import functools
import os
import re

from nemd import jobutils
from nemd import logutils
from nemd import objectutils
from nemd import process
from nemd import symbols
from nemd import timeutils
from nemd import check


class Cmd(objectutils.Object):
    """
    The class to parse a cmd test file.
    """

    POUND = symbols.POUND

    def __init__(self, dir, options=None):
        """
        :param dir str: the path containing the file
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.dir = dir
        self.options = options
        self.jobname = os.path.basename(self.dir)
        self.infile = os.path.join(self.dir, self.name)
        self.raw = []
        self.args = []
        if not os.path.isfile(self.infile):
            return
        with open(self.infile) as fh:
            self.raw = [x.strip() for x in fh.readlines() if x.strip()]
        self.args = [x for x in self.raw if not x.startswith(self.POUND)]

    @property
    def cmts(self):
        """
        Return the comment out of the raw.

        :return generator: the comments
        """
        for line in self.raw:
            if not line.startswith(self.POUND):
                return
            yield line.strip(self.POUND).strip()

    @property
    def prefix(self):
        """
        Return the prefix out of the raw.

        :return str: the prefix
        """
        cmt = symbols.SPACE.join(self.cmts)
        msg = f"{self.jobname}: {cmt}" if cmt else self.jobname
        return f'echo "{msg}"'


class Param(Cmd):
    """
    The class to parse the parameter file.
    """

    PARAM = f'$param'

    def __init__(self, cmd, **kwargs):
        """
        :param cmd `Cmd`: the Cmd instance parsing the cmd file.
        """
        super().__init__(cmd.dir, **kwargs)
        self.cmd = cmd
        self.args = Tag(self.dir, options=self.options).fast(self.args)

    @property
    def cmds(self,
             jobname_re=re.compile(rf'{jobutils.FLAG_JOBNAME} +\w+'),
             script_re=re.compile(r'.* +(.*)_(driver|workflow).py( +.*|$)')):
        """
        Get the parameterized commands.

        :param jobname_re `re.compile`: the jobname regular express
        :param script_re `re.compile`:the script regular express
        :return list: each value is one command.
        """
        if not all([self.args, self.PARAM in self.cmd.args[0]]):
            return self.cmd.args
        name = self.label or script_re.match(self.cmd.args[0]).group(1)
        cmd = f"{jobname_re.sub('', self.cmd.args[0])} {jobutils.FLAG_NAME} {name}"
        cmds = [
            f'{cmd.replace(self.PARAM, x)} {jobutils.FLAG_JOBNAME} {name}_{x}'
            for x in self.args
        ]
        return cmds

    @property
    @functools.cache
    def label(self, rex=re.compile(rf'-(\w*) \{PARAM}')):
        """
        Get the xlabel from the header of the parameter file.

        :param rex `re.compile`: the regular express to search param label
        :return str: The xlabel.
        """
        label = next(self.cmts, '')
        if not label and self.cmd:
            match = rex.search(self.cmd.args[0])
            label = match.group(1) if match else self.name
            with open(self.infile, 'w') as fh:
                fh.write('\n'.join([f"{self.POUND} {label}"] + self.args))
        return label.lower().replace(' ', '_')


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
    CMP = check.Cmp.name

    def run(self, rex=re.compile(rf'{NEMD_CHECK} +{CMP} +(\w+)')):
        """
        Check the results by execute all operators. Raise errors if any failed.

        :raise CheckError: if check failed.
        """
        print(f"{self.jobname}: {'; '.join(self.args)}")
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

    def setOperators(self, rex=re.compile(r'(?:;|&&|\|\|)?(\w+)\\((.*?)\\)')):
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
        roots = [os.path.splitext(x.namepath)[0] for x in self.logs]
        params = [x.split('_')[-1] for x in roots]
        tags = [
            f"{x}, {timeutils.delta2str(y)}" for x, y in zip(params, times)
        ]
        self.setMult(self.SLOW, *tags)

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
        print(f"{self.jobname}: {symbols.COMMA.join(self.args)}")
        with open(self.infile, 'w') as fh:
            for key, *value in self.operators:
                values = symbols.COMMA.join(value)
                fh.write(f"{key}({values})\n")

    def selected(self):
        """
        Select the operators by the options.

        :return bool: Whether the test is selected.
        """
        return self.fast() and self.labeled

    def fast(self, args=None):
        """
        Get the parameters considered as fast.

        :param args list: the parameters to filter
        :return bool or list of str: parameters filtered by slow
        """
        if self.options.slow is None:
            return True if args is None else args
        # [['slow', 'traj', '00:00:01'], ['slow', '9', '00:00:04']]
        params = [x[1:] for x in self.operators if x[0] == self.SLOW]
        time = [timeutils.str2delta(x[-1]).total_seconds() for x in params]
        fast = {x[0] for x, y in zip(params, time) if y <= self.options.slow}
        return True if args is None else [x for x in args if x in fast]

    @property
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
