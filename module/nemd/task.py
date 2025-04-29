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
from nemd import osutils
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
    TMPL = [None]


class LmpLog(taskbase.Cmd):
    """
    Class to run lammps log driver.
    """

    FILE = 'lmp_log_driver.py'
    TMPL = [None]
    ParserClass = parserutils.LmpLog

    def addfiles(self):
        """
        Set arguments to analyze the log file.

        :param match_re 're.Pattern': the re to search data file
        """
        super().addfiles()
        # Set the args with the data file from the log file
        data_file = self.getMatch().group(1)
        self.args += [parserutils.LmpLog.FLAG_DATA_FILE, data_file]

    def getMatch(self, match_re=re.compile(lammpsfix.READ_DATA_RE)):
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
    The class to parse file, setup cmd, and run job.
    """
    CPU_RE = re.compile(fr"{jobutils.FLAG_CPU} +\d*")
    DEBUG_RE = re.compile(f"{jobutils.FLAG_DEBUG}( +(True|False))?")
    SEP = f"{symbols.RETURN}"

    def run(self):
        """
        Get arguments that form command lines.
        """
        self.setArgs()
        self.addfiles()
        self.addQuot()
        self.numCpu()
        self.setDebug()
        self.setScreen()

    def setArgs(self):
        """
        Set the arguments.
        """
        self.args = self.param.cmds

    @property
    @functools.cache
    def param(self):
        """
        The param object.

        :return `test.Param`: the param object.
        """
        return test.Param(self.cmd, options=self.options)

    @property
    @functools.cache
    def cmd(self):
        """
        The cmd object.

        :return `test.Cmd`: the cmd object.
        """
        return test.Cmd(self.jobs[0].statepoint[jobutils.FLAG_DIRNAME])

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
        args = [] if self.doc is None else self.doc[symbols.ARGS]
        cpu = jobutils.get_arg(args, jobutils.FLAG_CPU, 1)
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            if jobutils.FLAG_CPU not in cmd:
                # CPU not defined for the sub-job: 1 cpu for efficiency
                self.args[idx] = f"{cmd} {jobutils.FLAG_CPU} {cpu}"
                continue
            if self.options.CPU:
                # CPU defined in the cmd file, but users forced a different
                flag_num = f"{jobutils.FLAG_CPU} {cpu}"
                self.args[idx] = self.CPU_RE.sub(flag_num, cmd)
                continue
            # Use the CPU defined in the cmd file

    def setDebug(self):
        """
        Set the screen output.
        """
        if self.options.DEBUG is None:
            return
        debug_bool = f"{jobutils.FLAG_DEBUG} {self.options.DEBUG}"
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            if jobutils.FLAG_DEBUG in cmd:
                self.args[idx] = self.DEBUG_RE.sub(debug_bool, cmd)
            else:
                self.args[idx] = f"{cmd} {debug_bool}"

    def setScreen(self):
        """
        Set the screen output.
        """
        if self.options.screen != symbols.OFF or self.options.DEBUG:
            return
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            self.args[idx] = f"{cmd} > /dev/null"

    @property
    def out(self):
        """
        The output.

        :return bool: True when all output files are set.
        """
        return len(self.getJobs()) >= max(1, len(self.param.args))

    def clean(self):
        """
        Clean the previous jobs including the outfiles and the workspace.
        """
        # Remove all jobs as the cmd task jobnames are unknown.
        for job in self.getJobs():
            job.clean()
        for job in self.jobs:
            try:
                shutil.rmtree(job.fn(jobutils.WORKSPACE))
            except FileNotFoundError:
                pass

    def getCmd(self, **kwargs):
        """
        Get command line str.

        :return str: the command as str
        """
        return super().getCmd(prefix=self.cmd.prefix, **kwargs)


class Check(taskbase.Job):
    """
    The job class to parse the check file, run the operators, and set the job
    message.
    """
    TestClass = test.Check

    def run(self):
        """
        Main method to execute.
        """
        with osutils.chdir(self.job.dirname):
            dirname = self.jobs[0].statepoint[jobutils.FLAG_DIRNAME]
            obj = self.TestClass(dirname, options=self.options)
            self.out = obj.run() or True


class Tag(Check):
    """
    This job class generates a new tag file (or updates the existing one).
    """

    TestClass = test.Tag


class LmpAgg(taskbase.Agg):
    """
    Analyzer aggregator.
    """
    AnalyzerAgg = analyzer.Agg

    def run(self):
        """
        Main method to run the aggregator job.
        """
        self.log(f"{len(self.jobs)} jobs found for aggregation.")
        options = dict(NAME=self.jobname.removesuffix('_agg'))
        options = types.SimpleNamespace(**{**vars(self.options), **options})
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
        Group jobs by the statepoint so that the jobs within one group only
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
    Report the time.
    """
    TIME = symbols.TIME.lower()

    def run(self):
        """
        Report the total task timing and timing details grouped by name.
        """
        if not self.jobs:
            return
        jobs = [x for x in self.getJobs() if x.logfile]
        rdrs = [logutils.Reader(x.logfile) for x in jobs]
        info = [[x.options.NAME[:9], x.task_time] for x in rdrs]
        info = pd.DataFrame(info, columns=[symbols.NAME, self.TIME])
        pardirs = [os.path.basename(os.path.dirname(x.file)) for x in jobs]
        info[symbols.ID] = [x[:3] for x in pardirs]
        total_time = timeutils.delta2str(info.time.sum())
        self.log(logutils.Reader.TOTAL_TIME + total_time)
        grouped = info.groupby(symbols.NAME)
        data = {
            x: y.apply(lambda x: f'{self.delta2str(x.time)} {x.id}', axis=1)
            for x, y in grouped[[self.TIME, symbols.ID]]
        }
        data = {x: sorted(y, reverse=True) for x, y in data.items()}
        sorted_keys = sorted(data, key=lambda x: len(data[x]), reverse=True)
        ave = grouped.time.mean().apply(lambda x: f"{self.delta2str(x)} ave")
        data = {x: [ave.loc[x], *data[x]] for x in sorted_keys}
        data = pd.DataFrame.from_dict(data, orient='index').transpose()
        self.log(data.fillna('').to_markdown(index=False))

    @classmethod
    def delta2str(cls, delta, fmt='%M:%S'):
        """
        Delta time to string with upper limit.

        :param delta 'datetime.timedelta': the time delta object
        :param fmt str: the format to print the time
        :return str: the string representation of the delta time (< 1 hour)
        """
        return timeutils.delta2str(delta, fmt=fmt)


class TestAgg(TimeAgg):
    """
    Report the time of jobs filtered by ids and labels.
    """

    def run(self):
        """
        Main method to run.
        """
        self.filter()
        super().run()

    def filter(self):
        """
        Filter the jobs by the ids and labels.
        """
        if not any([self.options.id, self.options.label, self.options.slow]):
            return
        jobs = {x.statepoint[jobutils.FLAG_DIRNAME]: x for x in self.jobs}
        if self.options.id:
            jobs = {
                x: y
                for x, y in jobs.items()
                if int(os.path.basename(x)) in self.options.id
            }
        if self.options.label or self.options.slow:
            jobs = {
                x: y
                for x, y in jobs.items()
                if test.Tag(x, options=self.options).selected
            }
        self.jobs = list(jobs.values())
