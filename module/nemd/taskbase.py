# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Under jobcontrol:
 1) Base executes non-cmd operation
 2) Agg operates a non-cmd aggregation
 3) Job generates the cmd
"""
import functools

import flow

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import symbols


class Base(logutils.Base):
    """
    The base class for all jobs in the workflow. This class can be subclassed
    to create cmd and non-cmd jobs used as normal task jobs and aggregate jobs.
    """
    FILE = None
    ARGS = symbols.ARGS
    MESSAGE = 'message'
    SUFFIX = 'Job'

    def __init__(self, job, name=None, options=None, logger=None, **kwargs):
        """
        :param job 'signac.contrib.job.Job': the signac job instance
        :param name str: the subjob jobname, different from the workflow jobname
        :param options 'argparse.Namespace': commandline options
        :param driver 'module': imported driver module
        :param logger 'logging.Logger':  print to this logger
        """
        super().__init__(logger=logger)
        self.job = job
        self.name = name
        self.options = options
        self.logger = logger
        self.doc = self.job.doc
        self.jobname = name if name else self.default_name

    @classmethod
    @property
    def default_name(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        name = cls.__name__.removesuffix(cls.SUFFIX)
        return jobutils.get_name(cls.FILE, name=name)

    def run(self):
        """
        Main method to run.
        """
        self.message = False

    @property
    def message(self):
        """
        The message of the job.

        :return str: the message of the job.
        """
        return self.doc.get(self.MESSAGE, {}).get(self.jobname)

    @message.setter
    def message(self, value):
        """
        Set message of the job.

        :param value str: the message of the job.
        """
        self.doc.setdefault(self.MESSAGE, {})
        self.doc[self.MESSAGE].update({self.jobname: value})

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.message is False

    def clean(self):
        """
        Clean the previous job including the post criteria.
        """
        if self.MESSAGE not in self.doc:
            return
        self.doc[self.MESSAGE].pop(self.jobname, None)

    @classmethod
    def getOpr(cls,
               name=None,
               cmd=False,
               with_job=True,
               aggregator=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        :param name str: the job name
        :param cmd bool: whether the function returns a command to run
        :param with_job bool: whether execute in job dir with context management
        :param aggregator 'flow.aggregates.aggregator': job collection criteria
        :return 'function': the operation to execute
        """
        if name is None:
            name = cls.default_name

        # Operator
        kwargs.update({'name': name})
        func = functools.partial(cls.runOpr, **kwargs)
        func.__name__ = name
        func = flow.FlowProject.operation(name=name,
                                          func=func,
                                          cmd=cmd,
                                          with_job=with_job,
                                          aggregator=aggregator)
        # Post conditions
        post = functools.partial(cls.postOpr, **kwargs)
        func = flow.FlowProject.post(lambda *x: post(*x))(func)
        return func

    @classmethod
    def runOpr(cls, *args, **kwargs):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.

        :param BaseClass 'Base': the class to initiate and run
        :return str: the command to run a task.
        """
        obj = cls(*args, **kwargs)
        obj.run()
        return obj.getCmd() if isinstance(obj, Job) else None

    @classmethod
    def postOpr(cls, *args, **kwargs):
        """
        The job is considered finished when the post-conditions return True.

        :return bool: True if the post-conditions are met
        """
        return cls(*args, **kwargs).post()


class Agg(Base):
    """
    Non-cmd aggregator job.
    """
    SUFFIX = __qualname__

    def __init__(self, *jobs, **kwargs):
        """
        :param jobs list' of 'signac.contrib.job.Job': signac jobs to aggregate
        """
        super().__init__(jobs[0], **kwargs)
        self.jobs = jobs
        self.doc = self.job.project.doc

    @classmethod
    def getOpr(cls,
               *args,
               name=None,
               with_job=False,
               aggregator=flow.aggregator(),
               **kwargs):
        """
        Get the aggregator operator. (see parent for details)
        """
        if name is None:
            name = cls.default_name
        name = f"{name}{symbols.POUND_SEP}agg"
        return super().getOpr(*args,
                              name=name,
                              with_job=with_job,
                              aggregator=aggregator,
                              **kwargs)


class Job(Base):
    """
    Cmd job.
    """
    ParserClass = parserutils.Driver
    PRE_RUN = jobutils.NEMD_RUN
    SEP = symbols.SPACE
    PREREQ = 'prereq'
    ARGS_TMPL = None
    OUTFILE = jobutils.OUTFILE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = list(map(str, self.doc.get(self.ARGS, [])))

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
