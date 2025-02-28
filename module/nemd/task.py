# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This task module provides Job subclasses to generate the cmd (*-driver.py) and
AggJob subclasses to execute a non-cmd aggregator.
"""
import collections
import functools
import os
import re
import shlex
import shutil
import types

import pandas as pd

from nemd import DEBUG
from nemd import analyzer
from nemd import jobutils
from nemd import lammpsfix
from nemd import parserutils
from nemd import symbols
from nemd import taskbase
from nemd import test


class MolBldrJob(taskbase.Job):
    """
    Class to run the molecule builder.
    """
    FILE = 'mol_bldr_driver.py'
    ParserClass = parserutils.MolBldr


class AmorpBldrJob(taskbase.Job):
    """
    Class to run the amorphous builder.
    """
    FILE = 'amorp_bldr_driver.py'
    ParserClass = parserutils.AmorpBldr


class XtalBldrJob(taskbase.Job):
    """
    Class to run the crystal builder.
    """
    FILE = 'xtal_bldr_driver.py'
    ParserClass = parserutils.XtalBldr


class LammpsJob(taskbase.Job):
    """
    Class to run the lammps simulation.
    """
    FILE = 'lammps_driver.py'
    ParserClass = parserutils.Lammps
    ARGS_TMPL = [None]


class LogJob(taskbase.Job):
    """
    Class to run lammps log driver.
    """

    FILE = 'lmp_log_driver.py'
    ARGS_TMPL = [None]
    ParserClass = parserutils.Log

    def addfiles(self, match_re=re.compile(lammpsfix.READ_DATA_RE)):
        """
        Set arguments to analyze the log file.

        :param match_re 're.Pattern': the re to search data file
        """
        super().addfiles()
        # Set the args with the data file from the log file
        data_file = self.getMatch(match_re=match_re).group(1)
        self.args += [parserutils.Log.FLAG_DATA_FILE, data_file]

    def getMatch(self, match_re=lammpsfix.READ_DATA_RE):
        """
        Get the regular expression match.

        :param match_re 're.Pattern': the re to search pattern
        :return 're.Match': the found match
        """
        with open(self.args[0], 'r') as fh:
            matches = (match_re.match(line) for line in fh)
            return next(x for x in matches if x)


class TrajJob(LogJob):
    """
    Class to run lammps traj driver.
    """
    FILE = 'lmp_traj_driver.py'
    ParserClass = parserutils.Traj

    def addfiles(self, match_re=re.compile(lammpsfix.DUMP_RE)):
        """
        Set arguments to analyze the custom dump file.

        :param match_re 're.Pattern': the re to search trajectory file
        """
        super().addfiles()
        # Set the args with the trajectory file from the log file
        self.args[0] = self.getMatch(match_re=match_re).group(2)


class CmdJob(taskbase.Job):
    """
    The class to set up a job cmd so that the test can run normal nemd jobs from
    the cmd line.
    """
    NAME = 'cmd'
    CPU_RE = re.compile(f"{jobutils.FLAG_CPU} +\d*")
    SEP = f"{symbols.RETURN}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmd = test.Cmd(job=self.job)
        self.args = self.param.getCmds()

    def getCmd(self, **kwargs):
        """
        Get command line str.

        :return str: the command as str
        """
        msg = f"# {os.path.basename(self.job.statepoint[jobutils.FLAG_DIR])}"
        if self.cmd.comment:
            msg += f": {self.cmd.comment}"
        return super().getCmd(prefix=f'echo "{msg}"', **kwargs)

    def run(self):
        """
        Get arguments that form command lines.
        """
        self.quote()
        self.numCpu()
        self.setName()
        self.setDebug()
        self.setScreen()

    def quote(self, rex=re.compile("[^'\"][@!#%^&*()<>?|}{:\[\]][^'\"]")):
        """
        Add quotes for str with special characters.

        :param rex 're.Pattern': search unquoted text with special characters
        :return str: the quoted command
        """
        for idx, cmd in enumerate(self.args):
            # shlex.split("echo 'h(i)';echo wa", posix=False) splits by ;
            splitted = shlex.split(cmd, posix=False)
            quoted = [f'"{x}"' if rex.match(x) else x for x in splitted]
            self.args[idx] = symbols.SPACE.join(quoted)

    @property
    @functools.cache
    def param(self):
        """
        Return the parameter object.

        :return `test.Param`: the parameters.
        """
        return test.Param(job=self.job, cmd=self.cmd)

    def numCpu(self):
        """
        Set the cpu number.
        """
        # No CPU specified for the workflow: 1 cpu per subjob for efficiency
        default = f"{jobutils.FLAG_CPU} 1"
        try:
            index = self.original.index(jobutils.FLAG_CPU)
        except ValueError:
            cpu_num = None
        else:
            cpu_num = f"{jobutils.FLAG_CPU} {self.original[index + 1]}"

        for idx, cmd in enumerate(self.args):
            if not jobutils.FLAG_CPU in cmd:
                self.args[idx] = f"{cmd} {cpu_num if cpu_num else default}"
                continue
            if cpu_num:
                self.args[idx] = self.CPU_RE.sub(cpu_num, cmd)

    def setName(self):
        """
        Set the cmd job names when the jobname flag is not specified. (name is
        obtained from the python filename)
        """
        for idx, cmd in enumerate(self.args):
            if jobutils.FLAG_JOBNAME in cmd:
                continue
            match = test.FILE_RE.match(cmd)
            if not match:
                continue
            name = match.group(1)
            cmd += f" {jobutils.FLAG_JOBNAME} {name}"
            self.args[idx] = cmd

    def setDebug(self):
        """
        Set the screen output.
        """
        value = jobutils.get_arg(self.original, jobutils.FLAG_DEBUG)
        if value is None:
            return
        is_debug = parserutils.type_bool(value or 'on')
        for idx, cmd in enumerate(self.args):
            if is_debug and jobutils.FLAG_DEBUG not in cmd:
                self.args[idx] = f"{cmd} {jobutils.FLAG_DEBUG}"
            elif not is_debug and jobutils.FLAG_DEBUG in cmd:
                self.args[idx] = cmd.replace(jobutils.FLAG_DEBUG, "")

    def setScreen(self):
        """
        Set the screen output.
        """
        scn = jobutils.get_arg(self.original, jobutils.FLAG_SCREEN)
        if not scn and DEBUG:
            return
        if scn and jobutils.JOB in scn:
            return
        for idx, cmd in enumerate(self.args):
            self.args[idx] = f"{cmd} > /dev/null"

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        try:
            outfiles = self.doc[self.OUTFILE]
        except KeyError:
            return False
        if self.param.args is None:
            return bool(outfiles)
        return len(outfiles) >= len(self.param.args)

    def clean(self):
        """
        Clean the previous jobs including the outfiles and the workspace.

        Note: The jobnames of cmd tasks are determined by the cmd content.
        """
        self.doc[jobutils.OUTFILE] = {}
        self.doc[jobutils.OUTFILES] = {}
        workspace = self.job.fn(symbols.WORKSPACE)
        if os.path.isdir(workspace):
            shutil.rmtree(workspace)


class TagJob(taskbase.Base):
    """
    This job class generates a new tag file (or updates the existing one).
    """

    def run(self):
        """
        Main method to run.
        """
        test.Tag(job=self.job).run()
        self.message = False


class CheckJob(TagJob):
    """
    The job class to parse the check file, run the operators, and set the job
    message.
    """

    def run(self):
        """
        Main method to run.

        :raise ValueError: errors in debug mode are raised.
        """
        err = test.Check(job=self.job).run()
        self.message = err if err else False
        if DEBUG and self.message:
            raise ValueError(self.message)

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.jobname in self.doc.get(self.MESSAGE, {})


class LogAgg(taskbase.AggJob):
    """
    The aggregator job for analyzers.
    """
    AnalyzerAgg = analyzer.Agg

    def run(self):
        """
        Main method to run the aggregator job.
        """
        options = types.SimpleNamespace(
            JOBNAME=self.options.JOBNAME,
            INTERAC=self.options.INTERAC,
            name=self.jobname.split(symbols.POUND_SEP)[0],
            dir=os.path.relpath(self.project.workspace, self.project.path))
        self.log(f"{len(self.jobs)} jobs found for aggregation.")
        for task in self.options.task:
            anlz = self.AnalyzerAgg(task=task,
                                    groups=self.groups,
                                    options=options,
                                    logger=self.logger)
            anlz.run()
        self.message = False

    @property
    @functools.cache
    def groups(self):
        """
        Group jobs by the statepoints so that the jobs within one group only
        differ by the FLAG_SEED.

        return list of tuples: each tuple contains parameters (pandas.Series),
        grouped jobs (signac.job.Job)
        """
        jobs = collections.defaultdict(list)
        series = {}
        for job in self.jobs:
            statepoint = dict(job.statepoint)
            statepoint.pop(jobutils.FLAG_SEED, None)
            params = {
                x[1:] if x.startswith('-') else x: y
                for x, y in statepoint.items()
            }
            params = pd.Series(params).sort_index()
            if params.empty:
                return [tuple([params, self.jobs])]
            values = params.str.split(symbols.COLON, expand=True).iloc[:, -1]
            key = tuple(float(x) if x.isdigit() else x for x in values)
            series[key] = params
            jobs[key].append(job)
        keys = sorted(series.keys())
        for idx, key in enumerate(keys):
            series[key].index.name = idx
        return [tuple([series[x], jobs[x]]) for x in keys]


class TestAgg(taskbase.AggJob):
    """
    The class to run a non-cmd aggregator over jobs filtered by the specified
    ids and labels.
    """

    def run(self):
        """
        Main method to run.
        """
        self.filterIds()
        self.filterLabels()
        super().run()

    def filterIds(self):
        """
        Filter the jobs by the specified ids.
        """
        if self.options is None or len(self.options.id) == 0:
            return
        dirs = [x.statepoint[jobutils.FLAG_DIR] for x in self.jobs]
        ids = [int(os.path.basename(x)) for x in dirs]
        self.jobs = [y for x, y in zip(ids, self.jobs) if x in self.options.id]

    def filterLabels(self):
        """
        Filter the jobs by the specified labels.
        """
        if self.options is None or self.options.label is None:
            return
        tags = [test.Tag(job=x, options=self.options) for x in self.jobs]
        self.jobs = [y for x, y in zip(tags, self.jobs) if x.selected()]


class MolBldr(taskbase.Task):
    """
    Class for the molecule builder.
    """

    JobClass = MolBldrJob


class AmorpBldr(taskbase.Task):
    """
    Class for the amorphous builder.
    """

    JobClass = AmorpBldrJob


class XtalBldr(taskbase.Task):
    """
    Class for the crystal builder.
    """

    JobClass = XtalBldrJob


class Lammps(taskbase.Task):
    """
    Class for the lammps driver.
    """

    JobClass = LammpsJob


class LmpLog(taskbase.Task):
    """
    Class for the lammps log analyzer.
    """

    JobClass = LogJob
    AggClass = LogAgg


class LmpTraj(LmpLog):
    """
    Class for the lammps trajectory analyzer.
    """

    JobClass = TrajJob


class Cmd(taskbase.Task):
    """
    The class to run commands in a cmd file.
    """

    JobClass = CmdJob


class Check(taskbase.Task):
    """
    Class to check the results by executing the operators in the check file.
    """

    JobClass = CheckJob


class Tag(taskbase.Task):
    """
    Class to generate a new tag file (or updates the existing one).
    """

    JobClass = TagJob


class Test(taskbase.Task):
    """
    The task class to hold the aggregator over filtered jobs.
    """
    AggClass = TestAgg
