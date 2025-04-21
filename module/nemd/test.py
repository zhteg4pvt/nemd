# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides test related classes to parse files for command, parameter,
checking, and labels.
"""
import functools
import os
import re

from nemd import check
from nemd import jobutils
from nemd import logutils
from nemd import objectutils
from nemd import process
from nemd import symbols
from nemd import timeutils


class Base(objectutils.Object):
    """
    The base class to parse a test file.
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

    def getHeader(self, args=None):
        """
        Return the header.

        :param args list: the arguments to generate the message.
        :return str: the header
        """
        msg = symbols.SEMICOLON.join(args or self.args)
        return f"{self.jobname}: {msg}" if msg else self.jobname

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
    @functools.cache
    def args(self):
        """
        Return the argument.

        :return list: the argument in the files. (non-comment lines)
        """
        return [x for x in self.raw if not x.startswith(self.POUND)]

    @property
    @functools.cache
    def raw(self):
        """
        Return the raw lines.

        :return list: the raw lines.
        """
        if not os.path.isfile(self.infile):
            return []
        with open(self.infile) as fh:
            return [x.strip() for x in fh.readlines() if x.strip()]


class Cmd(Base):
    """
    The class to parse a cmd test file.
    """

    @property
    def prefix(self):
        """
        Return the prefix.

        :return str: the prefix
        """
        return f'echo "{self.getHeader([symbols.SPACE.join(self.cmts)])}"'


class Param(Base):
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
    def args(self):
        """
        See the parent.
        """
        return Tag(self.dir, options=self.options).fast(super().args)

    @property
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


class Check(Base):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    NEMD_CHECK = 'nemd_check'
    CMP = check.Cmp.name
    SEP = symbols.SEMICOLON
    RPL = rf'{NEMD_CHECK} {CMP} {{dir}}{os.path.sep}\1'

    def run(self, rex=re.compile(rf'{NEMD_CHECK} +{CMP} +(\w+)')):
        """
        Check the results by execute all operators.

        :return str: error message.
        """
        print(self.getHeader())
        rpl = self.RPL.format(dir=self.dir)
        args = [rex.sub(rpl, x) for x in self.args]
        proc = Process(args, jobname=self.jobname)
        completed = proc.run()
        if not completed.returncode:
            return
        with open(proc.logfile) as fh:
            return fh.read()


class Process(process.Base):
    """
    Sub process to run check cmd.
    """

    NAME = Check.name
    SEP = ';\n'


class Tag(Base):
    """
    The class parses and interprets the tag file. The class also creates a new
    tag file (or update existing one).
    """

    SLOW = 'slow'
    LABEL = 'label'

    def __init__(self, *args, **kwargs):
        """
        :param delay 'bool': delay the setup if True.
        """
        super().__init__(*args, **kwargs)
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
        parms = [x.split('_')[-1] for x in roots]
        tags = [[x, timeutils.delta2str(y)] for x, y in zip(parms, times)]
        self.setMult(self.SLOW, *tags)

    def setMult(self, key, *values):
        """
        Set the values (and the key) of multiple operators.

        :param key str: the key to be set
        :param values tuple of list: each list contains the value(s) of one
            operator.
        """
        matches = [i for i, x in enumerate(self.args) if x[0] == key]
        for idx in reversed(matches):
            self.args.pop(idx)
        for value in values:
            self.args.append([self.SLOW, *value])

    @property
    @functools.cache
    def args(self):
        """
        See the parent.
        """
        return [x.split() for x in super().args]

    def setLabel(self):
        """
        Set the label of the job.
        """
        labels = self.get(self.LABEL, [])
        labels += [x.options.NAME for x in self.logs]
        if not labels:
            return
        self.set(self.LABEL, *set(labels))

    def set(self, key, *value):
        """
        Set the value (and the key) of one operator.

        :param key str: the key to be set
        :param value tuple of str: the value(s) to be set
        """
        try:
            idx = next(i for i, x in enumerate(self.args) if x[0] == key)
        except StopIteration:
            self.args.append([key, *value])
        else:
            self.args[idx] = [key, *value]

    def get(self, key, default=None):
        """
        Get the value of a specific key.

        :param key str: the key to be searched
        :param default str: the default value if the key is not found
        :return tuple of str: the value(s)
        """
        for name, *value in self.args:
            if name == key:
                return value
        return tuple() if default is None else default

    def write(self):
        """
        Write the tag file.
        """
        print(self.getHeader([symbols.SPACE.join(x) for x in self.args]))
        with open(self.infile, 'w') as fh:
            for values in self.args:
                fh.write(f"{symbols.SPACE.join(values)}\n")

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
        params = [x[1:] for x in self.args if x[0] == self.SLOW]
        time = [timeutils.str2delta(x[-1]).total_seconds() for x in params]
        fast = {x[0] for x, y in zip(params, time) if y <= self.options.slow}
        return bool(fast) if args is None else [x for x in args if x in fast]

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
