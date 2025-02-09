# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This task base module defines the Base class to execute non-cmd jobs, the Job
class to generate the cmd, the AggJob class to execute a non-cmd aggregator, and
the Task class to hold and prepare a regular cmd job and an aggregator job.
"""
import functools
import re
import types

import flow
import pandas as pd

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import symbols
from nemd import timeutils


class Base(logutils.Base):
    """
    The base class for all jobs in the workflow. This class can be subclassed
    to create cmd and non-cmd jobs depending on whether the job returns a cmd
    to run in the shell or not. In terms of the workflow, the subclassed jobs
    can be used as normal task jobs or aggregate jobs.
    """

    ARGS = symbols.ARGS
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    MESSAGE = 'message'
    DATA_EXT = '.csv'
    PRE_RUN = None
    FILE = None

    def __init__(self, job, jobname=None, logger=None, **kwargs):
        """
        :param job 'signac.contrib.job.Job': the signac job instance
        :param name str: the subjob jobname, different from the workflow jobname
        :param driver 'module': imported driver module
        :param logger 'logging.Logger':  print to this logger
        """
        super().__init__(logger=logger)
        self.job = job
        self.jobname = jobname if jobname else self.name
        self.logger = logger
        self.doc = self.job.doc

    @classmethod
    @property
    def name(cls):
        """
        The default jobname.

        :return str: the default jobname
        """
        return jobutils.get_name(cls.FILE) if cls.FILE else symbols.NAME

    @property
    @functools.cache
    def args(self):
        return list(map(str, self.doc.get(self.ARGS, [])))

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.message is False

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

        :value str: the message of the job.
        """
        self.doc.setdefault(self.MESSAGE, {})
        self.doc[self.MESSAGE].update({self.jobname: value})

    def clean(self):
        """
        Clean the previous job including the post criteria.
        """
        if self.MESSAGE not in self.doc:
            return
        self.doc[self.MESSAGE].pop(self.jobname, None)


class Job(Base):
    """
    The class to set up a run_nemd driver job in a workflow.

    NOTE: this is a cmd job.
    """

    SPECIAL_CHAR_RE = re.compile("[@!#%^&*()<>?|}{:]")
    QUOTED_RE = re.compile('^".*"$|^\'.*\'$')
    PRE_RUN = jobutils.NEMD_RUN
    SEP = symbols.SPACE
    PREREQ = 'prereq'
    OUTFILE = jobutils.OUTFILE
    FILE = None

    def getCmd(self, prefix=PRE_RUN, write=True):
        """
        Get command line str.

        :param prefix str: the prefix command to run before the args
        :param write bool: whether to write the command to a file
        :return str: the command as str
        """
        cmd = self.getArgs()
        if self.FILE is not None:
            cmd.insert(0, self.FILE)
        if prefix:
            cmd.insert(0, prefix)
        cmd = self.SEP.join(cmd)
        if write:
            with open(f"{self.jobname}_cmd", 'w') as fh:
                fh.write(cmd)
        return cmd

    def getArgs(self):
        """
        Get the job arguments to construct the command.

        :return list: the command line arguments
        """
        args = self.addfiles()
        self.rmUnknown(args)
        self.setName(args)
        return [self.quote(x) for x in args]

    def addfiles(self):
        """
        Add the outfiles from previous jobs to the input arguments of this job.

        :return list: the updated arguments list
        """
        try:
            pre_jobs = self.doc[self.PREREQ][self.jobname]
        except KeyError:
            return self.args
        try:
            args = self.ARGS_TMPL[:]
        except AttributeError:
            return self.args
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for pre_job in pre_jobs:
            index = args.index(None)
            args[index] = self.doc[self.OUTFILE][pre_job]
        return args + self.args

    def rmUnknown(self, args):
        """
        Remove unknown arguments instead of keeping known so that the same flag
        across different tasks can be used multiple times.

        :param args list: the command line arguments before removing unknowns
        """
        parser = self.get_parser()
        _, unknown = parser.parse_known_args(args)
        try:
            first = next(i for i, x in enumerate(unknown) if x.startswith('-'))
        except StopIteration:
            # All unknowns are positional arguments
            first = len(unknown)
        for arg in unknown[:first]:
            args.remove(arg)
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
            flag_index = args.index(val)
            for index in reversed(range(flag_index, flag_index + lidx + 1)):
                args.pop(index)

    @classmethod
    @functools.cache
    def get_parser(cls, descr=None):
        """
        The user-friendly command-line parser.

        :param descr str: the descr of the program
        :return 'argparse.ArgumentParser': argparse figures out how to parse
          those out of sys.argv.
        """
        parser = parserutils.ArgumentParser(cls.FILE, descr=descr)
        cls.add_arguments(parser, positional=True)
        parser.add_job_arguments()
        return parser

    def add_arguments(self, *args, **kwargs):
        """
        Add job specific arguments to the parser. This method can be overridden
        by the subclass to add more arguments.

        :param parser 'argparse.ArgumentParser': the parser to add arguments to
        :param positional bool: whether to add positional arguments
        """
        pass

    def setName(self, args):
        """
        Set the jobname flag in the arguments. (self.jobname is usually defined on
        creating the workflow)

        :param args list: the command line arguments before setting the jobname
        """
        return jobutils.set_arg(args, self.FLAG_JOBNAME, self.jobname)

    @classmethod
    def quote(cls, arg):
        """
        Quote the unquoted argument that contains special characters.

        :param arg str: the argument to quote
        """
        if cls.SPECIAL_CHAR_RE.search(arg) and not cls.QUOTED_RE.match(arg):
            return f'"{arg}"'
        return arg

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

        :value str: the message of the job.
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


class AggJob(Base):
    """
    The class to run a non-cmd aggregator job in a workflow.
    """

    MS_FMT = '%M:%S'
    MS_LMT = '59:59'
    DELTA_LMT = timeutils.str2delta(MS_LMT, fmt=MS_FMT)
    TIME = symbols.TIME.lower()
    ID = symbols.ID
    NAME = symbols.NAME

    def __init__(self, *jobs, options=None, **kwargs):
        """
        :param jobs list' of 'signac.contrib.job.Job': signac jobs to aggregate
        :param options 'argparse.Namespace': commandline options
        """
        super().__init__(jobs[0], **kwargs)
        self.jobs = jobs
        self.options = options
        self.project = self.job.project

    def run(self):
        """
        Report the total task timing and timing details grouped by name.
        """
        if not self.job:
            return
        info = []
        for job in self.jobs:
            for filename in job.doc.get(jobutils.LOGFILE, {}).values():
                try:
                    log = logutils.LogReader(job.fn(filename))
                except FileNotFoundError:
                    continue
                log.run()
                info.append([log.options.NAME, log.task_time, job.id])
        info = pd.DataFrame(info, columns=[self.NAME, self.TIME, self.ID])
        # Group the jobs by the labels
        data, grouped = {}, info.groupby(self.NAME)
        for key, dat in sorted(grouped, key=lambda x: x[1].size, reverse=True):
            val = dat.drop(columns=self.NAME)
            val.sort_values(self.TIME, ascending=False, inplace=True)
            ave = val.time.mean()
            ave = pd.DataFrame([[ave, 'ave']], columns=[self.TIME, self.ID])
            val = pd.concat([ave, val]).reset_index(drop=True)
            val = val.apply(lambda x: f'{self.delta2str(x.time)} {x.id[:3]}',
                            axis=1)
            data[key[:8]] = val
        data = pd.DataFrame(data)
        if data.empty:
            self.log('No job founds.')
            self.message = False
            return
        total_time = timeutils.delta2str(info.time.sum())
        self.log(logutils.LogReader.TOTOAL_TIME + total_time)
        self.log(data.fillna('').to_markdown(index=False))
        self.message = False

    @property
    def message(self):
        """
        The message of the agg job.

        :return str: the message of the job.
        """
        return self.project.doc.get(self.MESSAGE, {}).get(self.jobname)

    @message.setter
    def message(self, value):
        """
        Set message of the agg job.

        :value str: the message of the job.
        """
        self.project.doc.setdefault(self.MESSAGE, {})
        self.project.doc[self.MESSAGE][self.jobname] = value

    @classmethod
    def delta2str(cls, delta):
        """
        Delta time to string with upper limit.

        :param delta: the time delta object
        :type delta: 'datetime.timedelta'
        :return str: the string representation of the time delta with upper limit
        """
        if pd.isnull(delta):
            return str(delta)
        if delta > cls.DELTA_LMT:
            return cls.MS_LMT
        return timeutils.delta2str(delta, fmt=cls.MS_FMT)

    def clean(self):
        """
        Clean the previous job including the post criteria.
        """
        if self.MESSAGE not in self.project.doc:
            return
        self.project.doc[self.MESSAGE].pop(self.jobname, None)


class Task:
    """
    The class holding a task job and an aggregator job.

    The post method is used to check if the job is finished, and a True return
    tasks the job out of the queue without executing it.
    The pre method is used to check if the job is eligible for submission. The
    current job is submitted only when its pre method returns the True and the
    post methods of all its prerequisite jobs return True as well.
    The operator method is used to execute the job, and is called on execution.
    The aggregator method is used to aggregate the results of the task jobs.
    """

    JobClass = Job
    AggClass = AggJob

    @classmethod
    def pre(cls, *args, **kwargs):
        """
        Set and check pre-conditions before starting the job.

        :return bool: True if the pre-conditions are met
        """
        return True

    @classmethod
    def operator(cls, *args, write=True, **kwargs):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.

        :param write bool: whether to write the command to a file
        :return str: the command to run a task.
        """
        kwargs.setdefault(symbols.NAME, cls.name)
        obj = cls.JobClass(*args, **kwargs)
        if hasattr(obj, 'getCmd'):
            return obj.getCmd(write=write)
        obj.run()

    @classmethod
    def post(cls, *args, **kwargs):
        """
        The job is considered finished when the post-conditions return True.

        :return bool: True if the post-conditions are met
        """
        kwargs.setdefault(symbols.NAME, cls.name)
        obj = cls.JobClass(*args, **kwargs)
        return obj.post()

    @classmethod
    def aggregator(cls, *args, **kwargs):
        """
        The aggregator job task.
        """
        kwargs.setdefault(symbols.NAME, cls.name)
        obj = cls.AggClass(*args, **kwargs)
        obj.run()

    @classmethod
    def aggPost(cls, *args, **kwargs):
        """
        Post-condition for aggregator task.

        :return bool: True if the post-conditions are met
        """
        kwargs.setdefault(symbols.NAME, cls.agg_name)
        obj = cls.AggClass(*args, **kwargs)
        return obj.post()

    @classmethod
    def getOpr(cls,
               cmd=None,
               with_job=True,
               jobname=None,
               attr='operator',
               pre=False,
               post=None,
               aggregator=None,
               logger=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        NOTE: post-condition must be provided so that the current job can check
        submission eligibility using post-condition of its prerequisite job.
        On the other hand, the absence of pre-condition means the current is
        eligible for submission as long as its prerequisite jobs are completed.

        :param cmd: Whether the aggregator function returns a command to run
        :type cmd: bool
        :param with_job: perform the execution in job dir with context management
        :type with_job: bool
        :param jobname str: the taskname
        :param attr: the attribute name of a staticmethod method or callable function
        :type attr: str or types.FunctionType
        :param pre: add pre-condition for the aggregator if True
        :type pre: bool
        :param post: add post-condition for the aggregator if True
        :type post: bool
        :param aggregator: the criteria to collect jobs
        :type aggregator: 'flow.aggregates.aggregator'
        :param logger:  print to this logger
        :type logger: 'logging.Logger'
        :return: the operation to execute
        :rtype: 'function'
        :raise ValueError: the function cannot be found
        """
        if jobname is None:
            jobname = cls.name
        if cmd is None:
            cmd = hasattr(cls.JobClass, 'getCmd')
        if post is None:
            post = cls.post
        if pre is None:
            pre = cls.pre
        if isinstance(attr, str):
            opr = getattr(cls, attr)
        elif isinstance(attr, types.FunctionType):
            opr = attr
        else:
            raise ValueError(f"{attr} is not a callable function or str.")

        # Pass jobname, taskname, and logging function
        kwargs.update({'jobname': jobname})
        if logger:
            kwargs['logger'] = logger
        func = functools.update_wrapper(functools.partial(opr, **kwargs), opr)
        func.__name__ = jobname
        func = flow.FlowProject.operation(cmd=cmd,
                                          func=func,
                                          with_job=with_job,
                                          name=jobname,
                                          aggregator=aggregator)
        # Add FlowProject decorators (pre / post conditions)
        if post:
            fp_post = functools.partial(post, jobname=jobname)
            fp_post = functools.update_wrapper(fp_post, post)
            func = flow.FlowProject.post(lambda *x: fp_post(*x))(func)
        if pre:
            fp_pre = functools.partial(pre, jobname=jobname)
            fp_pre = functools.update_wrapper(fp_pre, pre)
            func = flow.FlowProject.pre(lambda *x: fp_pre(*x))(func)
        return func

    @classmethod
    def getAgg(cls,
               cmd=False,
               with_job=False,
               jobname=None,
               attr='aggregator',
               post=None,
               **kwargs):
        """
        Get and register an aggregator job task that collects task outputs.

        :param cmd: Whether the aggregator function returns a command to run
        :type cmd: bool
        :param with_job: Whether chdir to the job dir
        :type with_job: bool
        :param jobname str: the name of this aggregator job task.
        :param attr: the attribute name of a staticmethod method or callable function
        :type attr: str or types.FunctionType
        :param post: add post-condition for the aggregator if True
        :type post: bool
        :return: the operation to execute
        :rtype: 'function'
        """
        if jobname is None:
            jobname = cls.agg_name
        if post is None:
            post = cls.aggPost
        return cls.getOpr(aggregator=flow.aggregator(),
                          cmd=cmd,
                          with_job=with_job,
                          jobname=jobname,
                          attr=attr,
                          post=post,
                          **kwargs)

    @classmethod
    @property
    def name(cls):
        """
        Return the default name of the task.

        :param name: the name of the operation.
        :return str: the default name of the operation.
        """
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '_'.join([x.lower() for x in words])

    @classmethod
    @property
    def agg_name(cls):
        """
        Return the default agg name of the operation.

        :return str: the default name of the agg operation.
        """
        return f"{cls.name}{symbols.POUND_SEP}agg"
