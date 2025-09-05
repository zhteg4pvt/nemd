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
from nemd import envutils
from nemd import jobutils
from nemd import lmpin
from nemd import logutils
from nemd import parserutils
from nemd import rdkitutils
from nemd import structure
from nemd import symbols
from nemd import taskbase
from nemd import test
from nemd import timeutils

FLAG_DIRNAME = jobutils.FLAG_DIRNAME


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


class LmpLog(taskbase.Cmd):
    """
    Class to run lammps log driver.
    """
    FILE = 'lmp_log_driver.py'
    ParserClass = parserutils.LmpLog

    def addfiles(self):
        """
        Set arguments to analyze the log file.

        :param rex 're.Pattern': the re to search data file
        """
        super().addfiles()
        # Set the args with the data file from the log file
        data_file = self.getMatch().group(1)
        self.args += [parserutils.LmpLog.FLAG_DATA_FILE, data_file]

    def getMatch(self, rex=lmpin.SinglePoint.READ_DATA_RE):
        """
        Get the regular expression match.

        :param rex 're.Pattern': the re to search pattern
        :return 're.Match': the found match
        """
        with open(self.args[0], 'r') as fh:
            matches = (rex.match(line) for line in fh)
            return next(x for x in matches if x)


class LmpTraj(LmpLog):
    """
    Class to run lammps traj driver.
    """
    FILE = 'lmp_traj_driver.py'
    ParserClass = parserutils.LmpTraj

    def addfiles(self,
                 rex=re.compile(r"dump 1 all (?:custom|xtc) (\d*) ([\w.]*)")):
        """
        Set arguments to analyze the custom dump file.

        :param rex 're.Pattern': the re to search trajectory file
        """
        super().addfiles()
        # Set the args with the trajectory file from the log file
        self.args[0] = self.getMatch(rex=rex).group(2)


class Cmd(taskbase.Cmd):
    """
    The class to parse file, setup cmd, and run job.
    """
    DEBUG_RE = re.compile(f"{jobutils.FLAG_DEBUG}( +(True|False))?")
    SEP = f"{symbols.RETURN}"
    PERFORMANCE = 'performance'

    def run(self):
        """
        Get arguments that form command lines.
        """
        self.setArgs()
        self.addQuot()
        self.numCpu()
        self.setDebug()
        self.setMem()
        self.setScreen()
        self.exit()

    def setArgs(self):
        """
        Set the arguments.
        """
        self.args = self.param.cmds

    @functools.cached_property
    def param(self):
        """
        The param object.

        :return `test.Param`: the param object.
        """
        return test.Param(self.cmd, options=self.options)

    @functools.cached_property
    def cmd(self):
        """
        The cmd object.

        :return `test.Cmd`: the cmd object.
        """
        return test.Cmd(self.job.fn(self.jobs[0].statepoint[FLAG_DIRNAME]))

    def addQuot(self):
        """
        Add quotations to words containing special characters.
        """
        for idx, cmd in enumerate(self.args):
            # shlex.split("echo 'h(i)';echo wa", posix=False) splits by ;
            words = shlex.split(cmd, posix=False)
            quoted = [self.quote(x) for x in words]
            self.args[idx] = symbols.SPACE.join(quoted)

    def numCpu(self, rex=re.compile(fr"{jobutils.FLAG_CPU} +\d*")):
        """
        See parent.

        :param rex `re.Pattern`: the regular expression to search cpu.
        """
        flag_cpu = f"{jobutils.FLAG_CPU} {self.options.CPU[1]}"
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            if jobutils.FLAG_CPU not in cmd:
                # CPU not defined in the cmd file
                self.args[idx] = f"{cmd} {flag_cpu}"
            elif self.options.CPU.forced:
                # CPU defined in the cmd file, but users forced one
                self.args[idx] = rex.sub(f"{flag_cpu}", cmd)
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

    def setMem(self):
        """
        Set the memory interval.
        """
        if self.options.name != self.PERFORMANCE:
            return
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            self.args[idx] = f"{envutils.MEM_INTVL}=1 {cmd}"

    def setScreen(self):
        """
        Set the screen output.
        """
        if self.options.screen == jobutils.JOB or self.options.DEBUG:
            return
        for idx, cmd in enumerate(self.args):
            if jobutils.NEMD_RUN not in cmd:
                continue
            self.args[idx] = f"{cmd} > /dev/null"

    def exit(self):
        """
        Ignore the previous non-zero return code.
        """
        self.args.append('exit 0')

    @property
    def out(self):
        """
        The output.

        :return bool: True when all output files are set.
        """
        return len([x for x in self.getJobs()
                    if x.logfile]) >= max(1, len(self.param.args))

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
        self._status.pop(self.job.dirname, None)

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
        dirname = self.jobs[0].statepoint[FLAG_DIRNAME]
        self.out = self.TestClass(dirname, options=self.options).run() or True


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
            try:
                Anlz = next(x for x in self.AnalyzerAgg.ANLZ if x.name == task)
            except StopIteration:
                continue
            anlz = self.AnalyzerAgg(Anlz,
                                    groups=self.groups,
                                    options=options,
                                    logger=self)
            anlz.run()

    @functools.cached_property
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
                return [types.SimpleNamespace(parm=params, jobs=self.jobs)]
            values = params.str.split(symbols.COLON, expand=True).iloc[:, -1]
            key = tuple(float(x) if x.isdigit() else x for x in values)
            series[key] = params
            jobs[key].append(job)
        keys = sorted(series.keys())
        for idx, key in enumerate(keys):
            series[key].index.name = idx
        return [
            types.SimpleNamespace(parm=series[x], jobs=jobs[x]) for x in keys
        ]


class TimeAgg(taskbase.Agg):
    """
    Report the time.
    """
    TIME = symbols.TIME.lower()

    def run(self):
        """
        Report the total task timing and timing details grouped by name.
        """
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
    Report the time of filtered jobs.
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
        jobs = {x.statepoint[FLAG_DIRNAME]: x for x in self.jobs}
        if self.options.id:
            jobs = {
                x: y
                for x, y in jobs.items()
                if os.path.basename(x) in self.options.id
            }
        if self.options.label or self.options.slow:
            jobs = {
                x: y
                for x, y in jobs.items()
                if test.Tag(y.fn(x), options=self.options).selected
            }
        self.jobs = list(jobs.values())


class Reader(logutils.Reader):
    """
    A builder log reader customized for substructure.
    """

    def getSubstruct(self, smiles):
        """
        Get the value of a substructure from the log file.

        :param smiles str: the substructure smiles
        :return str: the value of the substructure
        """
        for line in self.lines:
            if not line.startswith(smiles):
                continue
            # e.g. 'CCCC dihedral angle: 73.50 deg'
            return line.split(symbols.COLON)[-1].split()[0]


class AnalyzerAgg(analyzer.Agg):
    """
    Customized for substructures.
    """

    def merge(self):
        """
        Modify the result substructure column so that the name includes the
        structure smiles and geometry type.
        """
        super().merge()
        substruct = self.data.index.str.split(expand=True)
        has_value = self.data.index[0] != substruct[0]
        smiles = substruct[0][0] if has_value else substruct[0]
        # Set the name of the substructure column (e.g. CC Bond (Angstrom))
        name = f"{smiles} {structure.Mol.MolFromSmiles(smiles).name}"
        if has_value:
            # result.substruct contains the values  (e.g. CC: 2)
            self.data.index = pd.Index([x[1] for x in substruct], name=name)
            logutils.Reader.sort(self.data)
            return
        # result.substruct contains the smiles (e.g. CCCC)
        # Read the reported value from the log (e.g. dihedral angle: 73.50 deg)
        for job in jobutils.Job.search(self.groups[0].jobs[0].fn('')):
            reader = Reader(job.logfile)
            if reader.options.NAME != MolBldr.name:
                continue
            values = reader.getSubstruct(smiles)
            self.data.index = pd.Index([values], name=name)
