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
from nemd import np
from nemd import objectutils
from nemd import pd
from nemd import process
from nemd import symbols
from nemd import timeutils


class Base(logutils.Base, objectutils.Object):
    """
    The base class to parse a test file.
    """

    POUND = symbols.POUND

    def __init__(self, dir, options=None, **kwargs):
        """
        :param dir str: the path containing the file
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        super().__init__(**kwargs)
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
        self.cmd = next(iter(cmd.args), None)
        if self.options is None:
            self.options = cmd.options

    @property
    def cmds(self, rex=re.compile(rf'{jobutils.FLAG_JOBNAME} +\w+')):
        """
        Get the parameterized commands.

        :param rex `re.Pattern`: the jobname regular express
        :return list: each value is one command.
        """
        if not (self.args and self.cmd and self.PARAM in self.cmd):
            return [self.cmd] if self.cmd else []
        cmd = f"{rex.sub('', self.cmd)} {jobutils.FLAG_NAME} {self.label}"
        cmds = [
            f'{cmd.replace(self.PARAM, x)} {jobutils.FLAG_JOBNAME} {self.label}_{x}'
            for x in self.args
        ]
        return cmds

    @property
    @functools.cache
    def args(self):
        """
        See the parent.
        """
        fast = Tag(self.dir, options=self.options).fast
        return super().args if fast is None else fast

    @property
    def label(self, rex=re.compile(rf'-(\w*) \{PARAM}')):
        """
        Get the xlabel from the header of the parameter file.

        :param rex `re.Pattern`: the regular express to search param label
        :return str: The xlabel.
        """
        label = next(self.cmts, None)
        if not label and self.cmd:
            match = rex.search(self.cmd)
            if match:
                label = match.group(1)
        return label.lower().replace(' ', '_') if label else self.name


class Check(Base):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    def run(self,
            sub_re=re.compile(rf'(nemd_check) +({check.Cmp.name}) +(\w+)'),
            repl=rf'\1 \2 {{dir}}{os.path.sep}\3'):
        """
        Check the results by execute all operators.

        :param sub_re `re.Pattern`: the regular expression to substitute
        :param repl str: the replacement
        :return str: error message.
        """
        self.log(self.getHeader())
        replacement = repl.format(dir=self.dir)
        tokens = [sub_re.sub(replacement, x) for x in self.args]
        proc = process.Check(tokens, jobname=self.jobname)
        completed = proc.run()
        if not completed.returncode:
            return
        with open(proc.logfile) as fh:
            return fh.read()


class Tag(Base):
    """
    The class create, parses, and update the tag file.
    """

    SLOW = 'slow'
    LABEL = 'label'

    def run(self):
        """
        Main method to run.
        """
        self.setSlow()
        self.setLabel()
        self.write()

    def setSlow(self):
        """
        Set the slow tag with the total job time from the driver log files.
        """
        jobnames = [x.options.JOBNAME for x in self.logs]
        parms = [x.split('_')[-1] for x in jobnames]
        times = [timeutils.delta2str(x.task_time) for x in self.logs]
        slow = [y for x in zip(parms, times) for y in x]
        if not slow:
            return
        self.tags[self.SLOW] = slow

    @property
    @functools.cache
    def logs(self):
        """
        Set the log readers.
        """
        jobs = jobutils.Job(dir=self.dir).getJobs()
        return [logutils.Reader(x.logfile) for x in jobs if x.logfile]

    @property
    @functools.cache
    def tags(self):
        """
        The tags.

        :return dict: the tags
        """
        return {x[0]: x[1:] for x in (x.split() for x in self.args)}

    def setLabel(self):
        """
        Set the label of the job.
        """
        labels = self.tags.get(self.LABEL, [])
        labels += [x.options.NAME for x in self.logs]
        if not labels:
            return
        self.tags[self.LABEL] = list(set(labels))

    def write(self):
        """
        Write the tag file.
        """
        lines = [symbols.SPACE.join([x] + y) for x, y in self.tags.items()]
        self.log(self.getHeader(lines))
        with open(self.infile, 'w') as fh:
            for line in lines:
                fh.write(f"{line}\n")

    @property
    def selected(self):
        """
        Select the operators by the options.

        :return bool: Whether the test is selected.
        """
        return self.labeled and (self.fast is None or bool(self.fast))

    @property
    @functools.cache
    def fast(self):
        """
        Get the fast parameters.

        :return set: parameters filtered by slow
        """
        if self.options.slow is None:
            return
        # [['slow', 'traj', '00:00:01'], ['slow', '9', '00:00:04']]
        params = self.tags.get(self.SLOW)
        if not params:
            return
        params = np.reshape(params, (-1, 2))
        params = pd.Series(params[:, 1], index=params[:, 0])
        params = params.map(lambda x: timeutils.str2delta(x).total_seconds())
        return set(params[params <= self.options.slow].index)

    @property
    def labeled(self):
        """
        Whether the test is labeled with the specified labels.

        :return bool: Whether the test is labeled.
        """
        if self.options.label is None:
            return True
        for label in self.tags.get(self.LABEL, []):
            for selected in self.options.label:
                if label.startswith(selected):
                    return True
        return False


TASKS = [Cmd, Check, Tag]
