# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Under jobcontrol:
 1) Job executes without cmd
 2) Agg aggregates without cmd
 3) Cmd generates the cmd for execution
"""
import functools
import os
import re
import types

import flow

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import symbols

MESSAGE = 'message'


class Job(jobutils.Job, logutils.Base):
    """
    Non-cmd job.
    """
    PREREQ = 'prereq'
    OUT = MESSAGE

    def __init__(self, *jobs, name=None, options=None, logger=None, **kwargs):
        """
        :param jobs list' of 'signac.job.Job': signac jobs
        :param name str: the job name
        :param options 'argparse.Namespace': commandline options
        :param logger 'logging.Logger':  print to this logger
        """
        jobutils.Job.__init__(self,
                              jobname=name if name else self.default,
                              job=jobs[0])
        logutils.Base.__init__(self, logger=logger)
        self.jobs = jobs
        self.name = name
        self.options = options
        self.logger = logger
        self.proj = self.job.project
        self.doc = self.proj.doc if self.agg else self.job.doc

    @classmethod
    @property
    def default(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '_'.join([x.lower() for x in words])

    @classmethod
    @property
    def agg(cls):
        """
        Whether this is an aggregator class.

        :return bool: True when this is an aggregator.
        """
        return cls.__name__.endswith('Agg')

    @classmethod
    def getOpr(cls,
               name=None,
               cmd=None,
               with_job=None,
               aggregator=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        :param name str: the job name
        :param cmd bool: whether the function returns a command to run
        :param with_job bool: whether execute in job dir with context management
        :param aggregator 'flow.aggregates.aggregator': job collection criteria
        :return 'types.SimpleNamespace': the name, operation, and class
        """
        if name is None:
            name = cls.default
        if cmd is None:
            cmd = cls.OUT == jobutils.OUTFILE
        if with_job is None:
            with_job = not cls.agg
        if aggregator is None and cls.agg:
            aggregator = flow.aggregator()

        # Operator
        func = functools.partial(cls.runOpr, name=name, **kwargs)
        opr = flow.FlowProject.operation(name=name,
                                         func=func,
                                         cmd=cmd,
                                         with_job=with_job,
                                         aggregator=aggregator)
        # Post conditions
        post = functools.partial(cls.postOpr, name=name, **kwargs)
        opr = flow.FlowProject.post(lambda *x: post(*x))(opr)
        return types.SimpleNamespace(name=name, opr=opr, cls=cls)

    @classmethod
    def runOpr(cls, *args, **kwargs):
        """
        The main opterator. (execute this function after pre-conditions are met)

        :return str: the cmd.
        """
        obj = cls(*args, **kwargs)
        obj.run()
        if obj.OUT == MESSAGE and obj.out is None:
            obj.out = False
        return obj.getCmd()

    def run(self):
        """
        Main method to run.
        """
        self.out = False

    @property
    def out(self):
        """
        The output.

        :return str: the output.
        """
        return self.data.get(self.OUT)

    @out.setter
    def out(self, value):
        """
        Set output.

        :param value str: the output.
        """
        self.data[self.OUT] = value
        self.write()

    def getCmd(self, *arg, **kwargs):
        """
        Get command line str.

        :return str: the command as str
        """
        pass

    @classmethod
    def postOpr(cls, *args, **kwargs):
        """
        Whether the job is completed or not.

        :return bool: True if the post-conditions are met
        """
        return cls(*args, **kwargs).post()

    def post(self):
        """
        The job is considered completed when the out is set.

        :return bool: whether the post-conditions are met.
        """
        return self.out is not None

    def clean(self):
        """
        Clean the output.
        """
        try:
            os.remove(self.file)
        except FileNotFoundError:
            pass

    def getJobs(self, **kwargs):
        """
        Get all json jobs.

        :return 'Job' list: the json jobs (within one parameter set for Job;
            across all parameter sets for Agg)
        """
        return [
            y for x in self.jobs
            for y in super(Job, self).getJobs(job=x, **kwargs)
        ]


class Agg(Job):
    """
    Non-cmd aggregator.
    """


class Cmd(Job):
    """
    Cmd job.
    """
    FILE = None
    ParserClass = parserutils.Driver
    PRE_RUN = jobutils.NEMD_RUN
    SEP = symbols.SPACE
    ARGS_TMPL = None
    OUT = jobutils.OUTFILE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = list(map(str, self.doc.get(symbols.ARGS, [])))

    @classmethod
    @property
    def default(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        return jobutils.get_name(cls.FILE) if cls.FILE else super().default

    def run(self):
        """
        Get the job arguments to construct the command.
        """
        self.addfiles()
        self.rmUnknown()
        self.setName()

    def addfiles(self):
        """
        Add the outfiles from previous jobs to the input arguments of this job.
        """
        if self.ARGS_TMPL is None:
            return
        try:
            pre_jobs = self.doc[self.PREREQ][self.jobname]
        except KeyError:
            return
        self.args = self.ARGS_TMPL + self.args
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for pre_job in pre_jobs:
            index = self.args.index(None)
            self.args[index] = jobutils.Job(jobname=pre_job).getFile()

    def rmUnknown(self):
        """
        Remove unknown arguments.
        """
        parser = self.ParserClass(self.FILE)
        _, unknown = parser.parse_known_args(self.args)
        try:
            first = next(i for i, x in enumerate(unknown) if x.startswith('-'))
        except StopIteration:
            # All unknowns are positional arguments
            first = len(unknown)
        for arg in unknown[:first]:
            self.args.remove(arg)
        unknown = unknown[first:]
        for idx, val in enumerate(unknown, 1):
            if not val.startswith('-'):
                continue
            # Remove the current flag and followed values
            left = unknown[idx:]
            try:
                lidx = next(i for i, x in enumerate(left) if x.startswith('-'))
            except StopIteration:
                lidx = len(left)
            flag_index = self.args.index(val)
            for index in reversed(range(flag_index, flag_index + lidx + 1)):
                self.args.pop(index)

    def setName(self):
        """
        Set the jobname flag in the arguments.
        """
        return jobutils.set_arg(self.args, jobutils.FLAG_JOBNAME, self.jobname)

    def getCmd(self, prefix=PRE_RUN, write=True):
        """
        Get command line str.

        :param prefix str: the prefix command to run before the args
        :param write bool: whether to write the command to a file
        :return str: the command as str
        """
        if self.FILE is not None:
            self.args.insert(0, self.FILE)
        if prefix:
            self.args.insert(0, prefix)
        cmd = self.SEP.join(self.args)
        if write:
            with open(f"{self.jobname}_cmd", 'w') as fh:
                fh.write(cmd)
        return cmd
