# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Under jobcontrol:
 1) Job executes without cmd.
 2) Agg aggregates without cmd.
 3) Cmd generates the cmd for execution.

Dynamic Distribution Roadmap:

*** new process for each job ***
1) Main process starts Multiprocessing.Processes of no-prerequisite jobs.
2) Main periodically checks the exitcode of all processes.
    a) successful jobs run the post(), and report the status.
    b) new prerequisite-passed jobs run the setUp(), and start new processes.
3) All processes succeed or any fail.

NOTE: keep active process number <= max number of simultaneous jobs
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.exitcode

*** fixed number of processes ***
1) Main process puts no-prerequisite jobs to eligible multiprocessing.Queue().
2) Start Multiprocessing.Process of max number of simultaneous jobs with each periodically:
    a) get one job from the queue.
    b) execute the job.
    c) put the job id to the finished multiprocessing.Queue().
    d) sleep.
3) Main process gets the job from the finished queue, runs the post(), and
    reports the status.
4) Main process searches new prerequisite-passed jobs, runs the setUp(), and puts
    them to the eligible queue.
5) Main process joins the processes when all status are passed.

https://superfastpython.com/multiprocessing-manager-example/ (The complete example)
https://www.youtube.com/watch?v=EI1gLCvdX_U (Queue section)
"""
import collections
import functools
import multiprocessing
import os
import pathlib
import re
import types

import flow
import pandas as pd

from nemd import builtinsutils
from nemd import jobutils
from nemd import symbols

STATUS = 'status'


class Status(collections.defaultdict):
    """
    The job status class.
    """
    INITIATED = 'Initiated'
    PASSED = 'Passed'
    COLUMNS = [INITIATED, PASSED]

    def __init__(self, *args, name=None, **kwargs):
        """
        :param name: the project jobname.
        """
        super().__init__(*args, **kwargs)
        self.view = pd.DataFrame(columns=self.COLUMNS,
                                 index=pd.Index([], name='Dirname'))
        self.file = name and (pathlib.Path().cwd() /
                              f"{name}_status{symbols.LOG_EXT}")

    def set(self, job, value):
        """
        Set the job status with view updated and written.

        :param job 'jobutils.Job': the job with name in folder.
        :param value 'str': the status to set.
        """
        self[job.dirname][job.jobname] = value
        try:
            row = self.view.loc[job.dirname]
        except KeyError:
            row = [set() for _ in range(len(self.COLUMNS))]
            row = pd.Series(row, index=self.COLUMNS, name=job.dirname)
            self.view = pd.concat((self.view, row.to_frame().T),
                                  ignore_index=False)
        status = self.PASSED if value else self.INITIATED
        for column, jobnames in row.items():
            if column == status:
                jobnames.add(job.jobname)
            else:
                jobnames.discard(job.jobname)
        if not self.file or "daemon" in multiprocessing.current_process(
        )._config:
            # self._status not synced inside a multiprocessing.Pool
            return
        view = self.view.map(lambda x: ' '.join(x))
        view.index = view.index.map(lambda x: x.relative_to(self.file.parent))
        self.file.write_text(view.to_markdown())


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
        :param status 'collections.defaultdict': the post status of all jobs
        :param logger 'logutils.Logger': the logger to print message in the post
        """
        super().__init__(**kwargs)
        self.jobs = jobs
        self.options = options
        self._status = Status(dict) if status is None else status
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
        # FIXME: non-cmd should initialize status outside multiprocessing.Pool.
        obj.status = obj.out
        obj.run()
        if obj.OUT == STATUS and obj.out is None:
            obj.out = True
        return obj.getCmd()

    @property
    def status(self):
        """
        Get the status.

        :return str: the status.
        """
        return self._status[self.job.dirname].get(self.jobname)

    @status.setter
    def status(self, value):
        """
        Set the job status.

        :param value str: the status.
        """
        self._status.set(self.job, value)

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
        if self.status:
            return True
        if not self.out:
            return False
        # FIXME: every job should execute post outside the pool immediately on
        #  finish instead of waiting others. (See Dynamic Distribution Roadmap)
        self.status = self.out
        if self.logger and self.OUT == STATUS and isinstance(self.out, str):
            self.logger.log(self.out if self.agg else \
                            f"{self.jobname} in {self.job.dirname}: {self.out}")
        return True

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
        self._status[self.job.dirname].pop(self.jobname, None)


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
        self.args = jobutils.Args([])
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
        self.setCpu()

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
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for jobname in jobnames:
            file = jobutils.Job(jobname, dirname=self.job.dirname).outfile
            self.args.insert(0, os.path.relpath(file, os.curdir))

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
        for idx, arg in enumerate(self.args):
            self.args[idx] = self.quote(arg)

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
        self.args.set(jobutils.FLAG_JOBNAME, self.jobname)

    def setCpu(self):
        """
        Set the cpu number.
        """
        self.args.set(jobutils.FLAG_CPU, str(self.options.CPU[1]))

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
