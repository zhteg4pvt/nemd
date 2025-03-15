# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Under jobcontrol:
 1) Job executes without cmd
 2) Agg aggregates without cmd
 3) Cmd generates the cmd for execution
"""
import functools
import re
import types

import flow

from nemd import jobutils
from nemd import parserutils
from nemd import symbols

STATUS = 'status'


class Job(jobutils.Job):
    """
    Non-cmd job.
    """
    PREREQ = 'prereq'
    OUT = STATUS

    def __init__(self,
                 *jobs,
                 options=None,
                 status=None,
                 logger=None,
                 **kwargs):
        """
        :param jobs list' of 'signac.job.Job': signac jobs
        :param options 'argparse.Namespace': commandline options
        :param status dict: the post status of all jobs
        :param logger 'logutils.Logger': the logger to print message in the post
        """
        super().__init__(job=jobs[0] if jobs else None, **kwargs)
        self.jobs = jobs
        self.options = options
        self.status = status
        self.logger = logger
        self.doc = self.job.project.doc if self.job else None
        if self.job and self.agg:
            self.job = self.job.project

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
               jobname=None,
               cmd=None,
               with_job=None,
               aggregator=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        :param jobname str: the job name
        :param cmd bool: whether the function returns a command to run
        :param with_job bool: whether execute in job dir with context management
        :param aggregator 'flow.aggregates.aggregator': job collection criteria
        :return 'types.SimpleNamespace': the name, operation, and class
        """
        if jobname is None:
            jobname = cls.default
        if cmd is None:
            cmd = cls.OUT == jobutils.OUTFILE
        if with_job is None:
            with_job = not cls.agg
        if aggregator is None and cls.agg:
            aggregator = flow.aggregator()

        # Operator
        func = functools.partial(cls.runOpr, jobname=jobname, **kwargs)
        opr = flow.FlowProject.operation(name=jobname,
                                         func=func,
                                         cmd=cmd,
                                         with_job=with_job,
                                         aggregator=aggregator)
        # Post conditions
        post = functools.partial(cls.postOpr, jobname=jobname, **kwargs)
        opr = flow.FlowProject.post(lambda *x: post(*x))(opr)
        return types.SimpleNamespace(jobname=jobname, opr=opr, cls=cls)

    @classmethod
    def runOpr(cls, *args, **kwargs):
        """
        The main opterator. (execute this function after pre-conditions are met)

        :return str: the cmd.
        """
        obj = cls(*args, **kwargs)
        obj.run()
        if obj.OUT == STATUS and obj.out is None:
            obj.out = True
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
    def postOpr(cls, *args, status=None, logger=None, **kwargs):
        """
        Whether the job is completed or not.

        :param status dict: the post status of all jobs
        :param logger 'logutils.Logger': the logger to print message
        :return bool: True if the post-conditions are met
        """
        return cls(*args, status=status, logger=logger, **kwargs).post()

    def post(self):
        """
        The job is considered completed when the out is set.

        :return bool: whether the post-conditions are met.
        """
        key = self.jobname
        if not self.agg and self.job:
            key = (key, self.job.id)
        status = self.status.get(key) if self.status else None
        if status:
            return True
        out = self.out
        if self.status is not None:
            self.status[key] = out
        if self.logger and self.OUT == STATUS and isinstance(out, str):
            header = '' if self.agg else f"%s in %s: " % key
            self.logger.log(header + out.strip())
        return bool(out)

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

    def log(self, msg):
        """
        Save this message into the job json.

        :param msg str: the msg to be saved
        """
        self.out = msg if self.out is None else '\n'.join([self.out, msg])


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
        self.args = []
        if not self.job:
            return
        self.args += list(map(str, self.doc.get(symbols.ARGS, [])))
        self.args += [y for x in self.job.statepoint.items() for y in x]

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
            jobnames = self.doc[self.PREREQ][self.jobname]
        except KeyError:
            return
        self.args = self.ARGS_TMPL + self.args
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for jobname in jobnames:
            idx = self.args.index(None)
            self.args[idx] = jobutils.Job(jobname, job=self.job).getFile()

    def rmUnknown(self):
        """
        Remove unknown arguments.
        """
        parser = self.ParserClass()
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
