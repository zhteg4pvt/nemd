# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This task module provides Job subclasses to generate the cmd (*-driver.py) and
Agg subclasses to execute a non-cmd aggregator.
"""
import collections
import functools
import os
import re
import shlex
import shutil
import types

import pandas as pd

from nemd import analyzer
from nemd import jobutils
from nemd import lammpsfix
from nemd import logutils
from nemd import parserutils
from nemd import symbols
from nemd import taskbase
from nemd import test
from nemd import timeutils


class MolBldr(taskbase.Cmd):
    """
    Class to run the molecule builder.
    """
    FILE = 'mol_bldr_driver.py'
    ParserClass = parserutils.MolBldr


class AmorpBldr(taskbase.Cmd):
    """
    Class to run the amorphous builder.
    """
    FILE = 'amorp_bldr_driver.py'
    ParserClass = parserutils.AmorpBldr


class XtalBldr(taskbase.Cmd):
    """
    Class to run the crystal builder.
    """
    FILE = 'xtal_bldr_driver.py'
    ParserClass = parserutils.XtalBldr


class Lammps(taskbase.Cmd):
    """
    Class to run the lammps simulation.
    """
    FILE = 'lammps_driver.py'
    ParserClass = parserutils.Lammps
    ARGS_TMPL = [None]


class LmpLog(taskbase.Cmd):
    """
    Class to run lammps log driver.
    """

    FILE = 'lmp_log_driver.py'
    ARGS_TMPL = [None]
    ParserClass = parserutils.LmpLog

    def addfiles(self, match_re=re.compile(lammpsfix.READ_DATA_RE)):
        """
        Set arguments to analyze the log file.

        :param match_re 're.Pattern': the re to search data file
        """
        super().addfiles()
        # Set the args with the data file from the log file
        data_file = self.getMatch(match_re=match_re).group(1)
        self.args += [parserutils.LmpLog.FLAG_DATA_FILE, data_file]

    def getMatch(self, match_re=lammpsfix.READ_DATA_RE):
        """
        Get the regular expression match.

        :param match_re 're.Pattern': the re to search pattern
        :return 're.Match': the found match
        """
        with open(self.args[0], 'r') as fh:
            matches = (match_re.match(line) for line in fh)
            return next(x for x in matches if x)


class LmpTraj(LmpLog):
    """
    Class to run lammps traj driver.
    """
    FILE = 'lmp_traj_driver.py'
    ParserClass = parserutils.LmpTraj

    def addfiles(self, match_re=re.compile(lammpsfix.DUMP_RE)):
        """
        Set arguments to analyze the custom dump file.

        :param match_re 're.Pattern': the re to search trajectory file
        """
        super().addfiles()
        # Set the args with the trajectory file from the log file
        self.args[0] = self.getMatch(match_re=match_re).group(2)


class Cmd(taskbase.Cmd):
    """
    The class to set up a job cmd so that the test can run normal nemd jobs from
    the cmd line.
    """
    NAME = 'cmd'
    CPU_RE = re.compile(f"{jobutils.FLAG_CPU} +\d*")
    DEBUG_RE = re.compile(f"{jobutils.FLAG_DEBUG}( +(True|False))?")
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
        self.addfiles()
        self.addQuot()
        self.numCpu()
        self.setDebug()
        self.setScreen()

    def addQuot(self):
        """
        Add quotations to words containing special characters.
        """
        for idx, cmd in enumerate(self.args):
            # shlex.split("echo 'h(i)';echo wa", posix=False) splits by ;
            words = shlex.split(cmd, posix=False)
            quoted = [self.quote(x) for x in words]
            self.args[idx] = symbols.SPACE.join(quoted)

    def numCpu(self):
        """
        Set the cpu number.
        """
        for idx, cmd in enumerate(self.args):
            if not jobutils.FLAG_CPU in cmd:
                # CPU not defined for the sub-job: 1 cpu for efficiency
                self.args[idx] = f"{cmd} {jobutils.FLAG_CPU} 1"
            elif self.options.CPU is not None and len(self.options.CPU) == 2:
                # CPU defined for the sub-job, but users forced a different
                flag_num = f"{jobutils.FLAG_CPU} {self.options.CPU[1]}"
                self.args[idx] = self.CPU_RE.sub(flag_num, cmd)

    def setDebug(self):
        """
        Set the screen output.
        """
        if self.options.DEBUG is None:
            return
        debug_bool = f"{jobutils.FLAG_DEBUG} {self.options.DEBUG}"
        for idx, cmd in enumerate(self.args):
            if jobutils.FLAG_DEBUG in cmd:
                self.args[idx] = self.DEBUG_RE.sub(debug_bool, cmd)
            else:
                self.args[idx] = f"{cmd} {debug_bool}"

    def setScreen(self):
        """
        Set the screen output.
        """
        if self.options.screen is None and self.options.DEBUG:
            return
        if self.options.screen and jobutils.JOB in self.options.screen:
            return
        for idx, cmd in enumerate(self.args):
            self.args[idx] = f"{cmd} > /dev/null"

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        jobs = self.getJobs()
        if self.param.args is None:
            return bool(jobs)
        return len(jobs) >= len(self.param.args)

    @property
    @functools.cache
    def param(self):
        """
        Return the parameter object.

        :return `test.Param`: the parameters.
        """
        return test.Param(job=self.job, cmd=self.cmd, options=self.options)

    def clean(self):
        """
        Clean the previous jobs including the outfiles and the workspace.
        """
        # Remove all jobs as the cmd task jobnames are unknown.
        for job in self.getJobs():
            job.clean()
        try:
            shutil.rmtree(self.job.fn(jobutils.WORKSPACE))
        except FileNotFoundError:
            pass


class Check(taskbase.Job):
    """
    The job class to parse the check file, run the operators, and set the job
    message.
    """

    def run(self):
        """
        Main method to execute.
        """
        try:
            test.Check(job=self.job, options=self.options).run()
        except test.CheckError as err:
            self.out = str(err)


class Tag(Check):
    """
    This job class generates a new tag file (or updates the existing one).
    """

    def run(self):
        """
        Main method to execute.
        """
        test.Tag(job=self.job, options=self.options).run()


class LmpLogAgg(taskbase.Agg):
    """
    The aggregator job for analyzers.
    """
    AnalyzerAgg = analyzer.Agg

    def run(self):
        """
        Main method to run the aggregator job.
        """
        options = types.SimpleNamespace(JOBNAME=self.options.JOBNAME,
                                        INTERAC=self.options.INTERAC,
                                        name=self.jobname.removesuffix('_agg'),
                                        dir=jobutils.WORKSPACE)
        self.log(f"{len(self.jobs)} jobs found for aggregation.")
        for task in self.options.task:
            anlz = self.AnalyzerAgg(task=task,
                                    groups=self.groups,
                                    options=options,
                                    logger=self)
            anlz.run()

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


class TimeAgg(taskbase.Agg):
    """
    The class to run a non-cmd aggregator job in a workflow.
    """
    MS_FMT = '%M:%S'
    MS_LMT = '59:59'
    DELTA_LMT = timeutils.str2delta(MS_LMT, fmt=MS_FMT)
    TIME = symbols.TIME.lower()

    def run(self):
        """
        Report the total task timing and timing details grouped by name.
        """
        if not self.jobs:
            return
        jobs = [x for x in self.getJobs() if x.logfile]
        rdrs = [logutils.Reader(x.logfile) for x in jobs]
        info = [[x.options.NAME[:8], x.task_time] for x in rdrs]
        info = pd.DataFrame(info, columns=[symbols.NAME, self.TIME])
        info[symbols.ID] = [x.job.id[:3] for x in jobs]
        total_time = timeutils.delta2str(info.time.sum())
        self.log(logutils.Reader.TOTAL_TIME + total_time)
        grouped = info.groupby(symbols.NAME)
        data = {
            x: y.apply(lambda x: f'{self.delta2str(x.time)} {x.id}', axis=1)
            for x, y in grouped[[self.TIME, symbols.ID]]
        }
        data = {x: sorted(y, reverse=True) for x, y in data.items()}
        sorted_keys = sorted(data, key=lambda x: len(data[x]), reverse=True)
        ave = grouped.time.mean().apply(lambda x: f"{self.delta2str(x)} (ave)")
        data = {x: [ave.loc[x], *data[x]] for x in sorted_keys}
        data = pd.DataFrame.from_dict(data, orient='index').transpose()
        self.log(data.fillna('').to_markdown(index=False))

    @classmethod
    def delta2str(cls, delta):
        """
        Delta time to string with upper limit.

        :param delta 'datetime.timedelta': the time delta object
        :return str: the string representation of the delta time (< 1 hour)
        """
        if pd.isnull(delta):
            return str(delta)
        if delta > cls.DELTA_LMT:
            return cls.MS_LMT
        return timeutils.delta2str(delta, fmt=cls.MS_FMT)


class TestAgg(TimeAgg):
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
