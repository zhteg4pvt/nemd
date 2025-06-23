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

from nemd import builtinsutils
from nemd import jobutils
from nemd import symbols

STATUS = 'status'


class Job(builtinsutils.Object):
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
                 jobname=None,
                 **kwargs):
        """
        :param jobs list' of 'signac.job.Job': signac jobs
        :param options 'argparse.Namespace': commandline options
        :param status dict: the post status of all jobs
        :param logger 'logutils.Logger': the logger to print message in the post
        """
        super().__init__(**kwargs)
        self.jobs = jobs
        self.options = options
        self.status = status
        self.logger = logger
        self.jobname = jobname or self.name
        job = self.jobs[0] if self.jobs else None
        self.doc = job.project.doc if job else None
        dirname = (job.project if self.agg else job).fn('') if job else None
        self.job = jobutils.Job(jobname=self.jobname, dirname=dirname)

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
            jobname = cls.name
        if cmd is None:
            cmd = cls.OUT != STATUS
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
        self.out = True

    @property
    def out(self):
        """
        The output.

        :return str: the output.
        """
        return self.job.get(self.OUT)

    @out.setter
    def out(self, value):
        """
        Set output.

        :param value str: the output.
        """
        self.job[self.OUT] = value
        self.job.write()

    def getCmd(self, *arg, **kwargs):
        """
        Get command line str.

        :return str: the command as str
        """
        pass

    @classmethod
    def postOpr(cls, *jobs, status=None, logger=None, **kwargs):
        """
        Whether the job is completed or not.

        :param jobs 'signac.job.Job': the signac jobs
        :param status dict: the post status of all jobs
        :param logger 'logutils.Logger': the logger to print message
        :return bool: True if the post-conditions are met
        """
        return cls(*jobs, status=status, logger=logger, **kwargs).post()

    def post(self):
        """
        The job is considered completed when the out is set.

        :return bool: whether the post-conditions are met.
        """
        if self.status is None:
            return bool(self.out)
        key = (self.jobname, self.job.dirname)
        if self.status.get(key):
            return True
        if self.options and self.options.DEBUG:
            print(self.jobname, self.out)
        self.status[key] = self.out
        if self.logger and self.OUT == STATUS and isinstance(self.out, str):
            header = '' if self.agg else f"%s in %s: " % key
            self.logger.log(header + self.out.strip())
        return bool(self.out)

    def getJobs(self):
        """
        Get all json jobs.

        :return 'Job' list: the json jobs (within one parameter set for Job;
            across all parameter sets for Agg)
        """
        return [y for x in self.jobs for y in jobutils.Job.search(x.fn(''))]

    def log(self, msg):
        """
        Save this message into the job json.

        :param msg str: the msg to be saved
        """
        self.out = msg if self.out is None else '\n'.join([self.out, msg])

    def clean(self):
        """
        Clean the json file.
        """
        self.job.clean()


class Agg(Job):
    """
    Non-cmd aggregator.
    """


class Cmd(Job):
    """
    Cmd job.
    """
    FILE = None
    ParserClass = None
    RUN = jobutils.NEMD_RUN
    SEP = symbols.SPACE
    TMPL = None
    OUT = '_outfile'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = []
        if not self.jobs:
            return
        self.args += self.doc[symbols.ARGS]
        self.args += [y for x in self.jobs[0].statepoint.items() for y in x]

    def run(self):
        """
        Get the job arguments to construct the command.
        """
        self.addfiles()
        self.rmUnknown()
        self.addQuot()
        self.setName()

    def addfiles(self):
        """
        Add the outfiles from previous jobs to the input arguments of this job.
        """
        if self.TMPL is None:
            return
        try:
            jobnames = self.doc[self.PREREQ][self.jobname]
        except KeyError:
            return
        self.args = self.TMPL + self.args
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for jobname in jobnames:
            file = jobutils.Job(jobname, dirname=self.job.dirname).outfile
            self.args[self.args.index(None)] = file

    def rmUnknown(self):
        """
        Remove unknown arguments.
        """
        if not self.ParserClass:
            return
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

    def addQuot(self):
        """
        Add quotations to words containing special characters.
        """
        self.args = [self.quote(x) for x in self.args]

    @staticmethod
    def quote(arg,
              single=re.compile("^'.*'$"),
              double=re.compile('^".*"$'),
              spec=re.compile(r"[@!#%^&*()<?|}{:\[\]]")):
        """
        Quote if the unquoted argument containing special characters.

        :param arg str: the argument to quote
        :param single 're.Pattern': match single-quoted text
        :param double 're.Pattern': match double-quoted text
        :param spec 're.Pattern': search text with special characters
        :return str: the quoted word
        """
        return f'"{arg}"' if not single.match(arg) and not double.match(
            arg) and spec.search(arg) else arg

    def setName(self):
        """
        Set the jobname flag in the arguments.
        """
        jobutils.set_arg(self.args, jobutils.FLAG_JOBNAME, self.jobname)

    def getCmd(self, prefix=RUN, write=True):
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
