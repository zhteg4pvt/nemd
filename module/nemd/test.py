# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides test related classes to parse files for command, parameter,
checking, and labels.
"""
import functools
import os
import pathlib
import re

import numpy as np
import pandas as pd

from nemd import check
from nemd import jobutils
from nemd import logutils
from nemd import process
from nemd import symbols
from nemd import timeutils


class Base(logutils.Base):
    """
    The base class to parse a test file.
    """
    POUND = symbols.POUND

    def __init__(self, dirname, **kwargs):
        """
        :param dirname str: the path containing the file
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        super().__init__(**kwargs)
        self.dirname = pathlib.Path(dirname)
        self.infile = self.dirname / self.name

    def getHeader(self, args=None):
        """
        Return the header.

        :param args list: the arguments to generate the message.
        :return str: the header
        """
        msg = symbols.SEMICOLON.join(args or self.args)
        return f"{self.dirname.name}: {msg}" if msg else self.dirname.name

    @functools.cached_property
    def args(self):
        """
        Return the argument.

        :return list: the argument in the files. (non-comment lines)
        """
        return [x for x in self.raw if not x.startswith(self.POUND)]

    @functools.cached_property
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
        super().__init__(cmd.dirname, **kwargs)
        self.cmd = cmd
        self.first = next(iter(cmd.args), None)
        if self.options is None:
            self.options = cmd.options

    @property
    def cmds(self, rex=re.compile(rf'{jobutils.FLAG_JOBNAME} +\w+')):
        """
        Get the parameterized commands.

        :param rex `re.Pattern`: the jobname regular express
        :return list: each value is one command.
        """
        if not (self.args and self.first and self.PARAM in self.first):
            return self.cmd.args if self.first else []
        cmd = f"{rex.sub('', self.first)} {jobutils.FLAG_NAME} {self.label}"
        cmds = [
            f'{cmd.replace(self.PARAM, x)} {jobutils.FLAG_JOBNAME} {self.label}_{x}'
            for x in self.args
        ]
        return cmds

    @functools.cached_property
    def args(self):
        """
        See the parent.
        """
        fast = Tag(self.dirname, options=self.options).fast
        return super().args if fast is None else fast

    @property
    def label(self, rex=re.compile(rf'-(\w*) \{PARAM}')):
        """
        Get the xlabel from the header of the parameter file.

        :param rex `re.Pattern`: the regular express to search param label
        :return str: The xlabel.
        """
        label = next(self.cmts, None)
        if not label and self.first:
            match = rex.search(self.first)
            if match:
                label = match.group(1)
        return label.replace(' ', '_') if label else self.name


class Check(Base):
    """
    The class to execute the operators in addition to the parsing a file.
    """

    def run(self,
            sub_re=re.compile(rf'(nemd_check) +({check.Cmp.name}) +(\w+)'),
            repl=rf'\1 \2 {{dirname}}{os.path.sep}\3'):
        """
        Check the results by execute all operators.

        :param sub_re `re.Pattern`: the regular expression to substitute
        :param repl str: the replacement
        :return str: error message.
        """
        self.log(self.getHeader())
        replacement = repl.format(dirname=self.dirname)
        tokens = [sub_re.sub(replacement, x) for x in self.args]
        proc = process.Check(tokens, jobname=self.name)
        proc.run()
        return proc.err.strip('\n')

    @functools.cached_property
    def raw(self, default='nemd_check collect finished dropna=False'):
        """
        see parent.

        :param default str: the default check cmd.
        """
        return super().raw if os.path.isfile(self.infile) else [default]


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
        if self.collected.empty:
            return
        task_time = self.collected.task_time.map(
            lambda x: timeutils.Delta(x).toStr())
        self.tags[self.SLOW] = task_time.reset_index().values.flatten()

    @functools.cached_property
    def collected(self):
        """
        Return the collected log data.

        :return 'pd.DataFrame': the collected data.
        """
        return logutils.Reader.collect('task_time', 'NAME')

    @functools.cached_property
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
        labels += [x for x in self.collected.NAME]
        if not labels:
            return
        self.tags[self.LABEL] = list(set(labels))

    def write(self):
        """
        Write the tag file.
        """
        lines = [[x, *y] for x, y in self.tags.items()]
        lines = [symbols.SPACE.join(map(str, x)) for x in lines]
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

    @functools.cached_property
    def fast(self, func=lambda x: timeutils.Delta.fromStr(x).total_seconds()):
        """
        Get the fast parameters.

        :param func func: convert str to total seconds
        :return set: parameters filtered by slow
        """
        if self.options.slow is None:
            return
        # [['slow', 'traj', '00:00:01'], ['slow', '9', '00:00:04']]
        params = self.tags.get(self.SLOW)
        if not params:
            return
        params = np.reshape(params, (-1, 2))
        params = pd.Series(params[:, 1], index=params[:, 0]).map(func)
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
