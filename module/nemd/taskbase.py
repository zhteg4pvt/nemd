# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Under jobcontrol:
 1) Base executes non-cmd operation
 2) Agg operates a non-cmd aggregation
 3) Job generates the cmd
"""
import functools
import re
import types

import flow

from nemd import DEBUG
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import symbols


class Base(logutils.Base):
    """
    The base to create cmd and non-cmd jobs as normal jobs and aggregators.
    """
    AGG = False
    PREREQ = 'prereq'

    def __init__(self, *jobs, name=None, options=None, logger=None, **kwargs):
        """
        :param jobs list' of 'signac.contrib.job.Job': signac jobs
        :param name str: the subjob jobname, different from the workflow jobname
        :param options 'argparse.Namespace': commandline options
        :param driver 'module': imported driver module
        :param logger 'logging.Logger':  print to this logger
        """
        super().__init__(logger=logger)
        self.jobs = jobs
        self.name = name
        self.options = options
        self.logger = logger
        self.job = self.jobs[0]
        self.proj = self.job.project
        self.doc = self.proj.doc if self.AGG else self.job.doc
        self.jobname = name if name else self.default_name

    @classmethod
    @property
    def default_name(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '_'.join([x.lower() for x in words])

    @classmethod
    def getOpr(cls,
               name=None,
               cmd=False,
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
            name = cls.default_name
        if with_job is None:
            with_job = not cls.AGG
        if aggregator is None and cls.AGG:
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
        The main opterator (function) for a job task executed after
        pre-conditions are met.

        :return 'cls': the instantiated.
        """
        obj = cls(*args, **kwargs)
        obj.run()
        return obj

    def run(self):
        """
        Main method to run.
        """
        pass

    @classmethod
    def postOpr(cls, *args, **kwargs):
        """
        The job is considered finished when the post-conditions return True.

        :return bool: True if the post-conditions are met
        """
        return cls(*args, **kwargs).post()

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return True


class Job(Base):
    """
    The non-cmd normal jobs.
    """

    def run(self):
        """
        Main method to run.

        :raise Exception: execution error are raised in debug mode.
        """
        try:
            self.execute()
        except Exception as err:
            if DEBUG:
                raise err
            self.message = str(err)
        else:
            self.message = False

    def execute(self):
        """
        Main method to execute.
        """
        pass

    @property
    def message(self):
        """
        The message of the job.

        :return str: the message of the job.
        """
        return self.doc.get(symbols.MESSAGE, {}).get(self.jobname)

    @message.setter
    def message(self, value):
        """
        Set message of the job.

        :param value str: the message of the job.
        """
        self.doc.setdefault(symbols.MESSAGE, {})
        self.doc[symbols.MESSAGE].update({self.jobname: value})

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.jobname in self.doc.get(symbols.MESSAGE, {})

    def clean(self):
        """
        Clean the previous job including the post criteria.
        """
        if symbols.MESSAGE not in self.doc:
            return
        self.doc[symbols.MESSAGE].pop(self.jobname, None)


class Agg(Job):
    """
    Non-cmd aggregator job.
    """
    AGG = True


class Cmd(Job):
    """
    Cmd normal job.
    """
    FILE = None
    ParserClass = parserutils.Driver
    PRE_RUN = jobutils.NEMD_RUN
    SEP = symbols.SPACE
    ARGS_TMPL = None
    OUTFILE = jobutils.OUTFILE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = list(map(str, self.doc.get(symbols.ARGS, [])))

    @classmethod
    @property
    def default_name(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        return jobutils.get_name(
            cls.FILE) if cls.FILE else super().default_name

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
            self.args[index] = self.doc[self.OUTFILE][pre_job]

    def rmUnknown(self):
        """
        Remove unknown arguments instead of keeping known so that the same flag
        across different tasks can be used multiple times.
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
        Set the jobname flag in the arguments. (self.jobname is usually defined
        on creating the workflow)
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

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.outfile is not None

    @property
    def outfile(self):
        """
        The outfile of the job.

        :return str: the message of the job.
        """
        return self.doc.get(self.OUTFILE, {}).get(self.jobname)

    @outfile.setter
    def outfile(self, value):
        """
        Set outfile of the job.

        :param value str: the message of the job.
        """
        self.doc.setdefault(self.OUTFILE, {})
        self.doc[self.OUTFILE][self.jobname] = value

    def clean(self):
        """
        Clean the previous job including the post criteria.
        """
        for key in [self.OUTFILE, jobutils.OUTFILES]:
            if key not in self.doc:
                continue
            self.doc[key].pop(self.jobname, None)

    @classmethod
    def getOpr(cls, *args, cmd=True, **kwargs):
        """
        Get the cmd operator. (see parent for details)
        """
        return super().getOpr(*args, cmd=cmd, **kwargs)

    @classmethod
    def runOpr(cls, *args, **kwargs):
        """
        The operator to get cmd. (see parent for details)

        :return str: the command to run.
        """
        obj = super().runOpr(*args, **kwargs)
        return obj.getCmd()
