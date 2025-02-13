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

    @staticmethod
    def add_arguments(parser, **kwargs):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        """
        parser.add_polym_arguments()
        parser.add_bldr_arguments()
        parser.add_md_arguments()
        parser.suppress(buffer=f"{symbols.DEFAULT_CUT * 4}",
                        mol_num=[1],
                        temp=0,
                        timestep=1,
                        press=1,
                        relax_time=0,
                        prod_time=0,
                        prod_ens=lammpsfix.NVE)


class AmorpBldrJob(taskbase.Job):
    """
    Class to run the amorphous builder.
    """
    FILE = 'amorp_bldr_driver.py'
    FLAG_DENSITY = '-density'
    FLAG_METHOD = '-method'
    GRID = 'grid'
    PACK = 'pack'
    GROW = 'grow'
    METHODS = [GRID, PACK, GROW]

    @classmethod
    def add_arguments(cls, parser, **kwargs):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        """
        parser.add_polym_arguments()
        parser.add_bldr_arguments()
        parser.add_argument(
            cls.FLAG_DENSITY,
            metavar='g/cm^3',
            type=functools.partial(parserutils.type_ranged_float,
                                   bottom=0,
                                   included_bottom=False,
                                   top=30),
            default=0.5,
            help=f'The density used for {cls.PACK} and {cls.GROW} cells.')
        parser.add_argument(
            cls.FLAG_METHOD,
            choices=cls.METHODS,
            default=cls.GROW,
            help=f'place molecules into the space {cls.GRID}; {cls.PACK} '
            f'molecules with random rotation and translation; {cls.GROW} '
            'molecules by rotating rigid fragments.')
        parser.add_md_arguments()
        parser.suppress([parserutils.FLAG_SUBSTRUCT])


class XtalBldrJob(taskbase.Job):
    """
    Class to run the crystal builder.
    """
    FILE = 'xtal_bldr_driver.py'
    FLAG_SCALED_RANGE = '-scaled_range'

    @classmethod
    def add_arguments(cls, parser, **kwargs):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        """
        parser.add_xtal_arguments()
        parser.add_bldr_arguments()
        parser.add_md_arguments()
        parser.suppress(parserutils.BLDR_FLAGS)
        parser.set_defaults(force_field=[symbols.SW])


class LammpsJob(taskbase.Job):
    """
    Class to run the lammps simulation.
    """
    FILE = 'lammps_driver.py'
    ARGS_TMPL = [None]
    FLAG_INSCRIPT = 'inscript'
    FLAG_LOG = '-log'
    FLAG_DATA_FILE = '-data_file'

    @classmethod
    def add_arguments(cls, parser, positional=False):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        :param positional bool: whether to add positional arguments
        """
        if positional:
            parser.add_argument(cls.FLAG_INSCRIPT,
                                metavar=cls.FLAG_INSCRIPT.upper(),
                                type=parserutils.type_file,
                                help='Read input from this file.')
        parser.add_argument(jobutils.FLAG_SCREEN,
                            default=symbols.NONE,
                            help='Where to send screen output.')
        parser.add_argument(jobutils.FLAG_LOG,
                            metavar=jobutils.FLAG_LOG[1:].upper(),
                            help='Print logging information into this file.')
        parser.add_argument(cls.FLAG_DATA_FILE,
                            metavar=cls.FLAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='Data file to get force field information')
        if positional:
            parser.validators.add(cls.Validator)

    class Validator(parserutils.Validator):

        def run(self):
            """
            When not provided, try to locate the data file based on the input script.

            :raises FileNotFoundError: if data file is required but doesn't exist.
            """
            if self.options.data_file:
                return

            with open(self.options.inscript, 'r') as fh:
                contents = fh.read()
            matched = re.search(lammpsfix.READ_DATA_RE, contents)
            if not matched:
                return
            # try to find data file in the current dir and in the input script dir
            data_file = matched.group(1)
            if not os.path.isfile(data_file):
                dirname = os.path.dirname(self.options.inscript)
                data_file = os.path.join(dirname, data_file)

            if not os.path.isfile(data_file):
                raise FileNotFoundError(f"No data file {data_file} found.")

            self.options.data_file = data_file

    def rmUnknown(self, args):
        """
        In addition to the default behavior, remove the screen flag if the
        values are not in the choices.

        :param args list: the command line arguments before removing unknowns
        """
        try:
            index = len(args) - 1 - args[::-1].index(jobutils.FLAG_SCREEN)
        except ValueError:
            super().rmUnknown(args)
            return

        start, end = index + 1, len(args)
        try:
            end = next(x for x in range(start, end) if args[x].startswith('-'))
        except StopIteration:
            pass
        to_rm = [x for x in range(start, end) if x not in self.SCREEN_CHOICES]
        for idx in reversed(to_rm):
            args.pop(idx)
        if len(to_rm) == end - start:
            args.pop(index)
        super().rmUnknown(args)


class LogJob(taskbase.Job):
    """
    Class to run lammps log driver.
    """

    FILE = 'lmp_log_driver.py'
    ARGS_TMPL = [None]
    FLAG_DATA_FILE = '-data_file'
    FLAG_LAST_PCT = '-last_pct'
    FLAG_SLICE = '-slice'
    TASK_CHOICES = analyzer.THERMO.keys()
    TASK_HELP = 'Searches, combines and averages thermodynamic info.'
    LAST_FRM = analyzer.THERMO.keys()
    FLAG = 'log'
    HELP = 'LAMMPS log file to analyze.'

    def addfiles(self):
        """
        Set arguments to analyze the log file.
        """
        args = super().addfiles()
        log_file = args.pop(0)
        return [log_file, self.FLAG_DATA_FILE,
                self.getDataFile(log_file)] + args

    def getDataFile(self, log_file):
        """
        Set the args with the data file from the log file.
        """
        with open(log_file, 'r') as fh:
            for line in fh:
                match = re.match(lammpsfix.READ_DATA_RE, line)
                if not match:
                    continue
                return match.group(1)

    @classmethod
    def add_arguments(cls, parser, positional=False):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        :param positional bool: whether to add positional arguments
        """
        if positional:
            parser.add_argument(cls.FLAG,
                                metavar=cls.FLAG.upper(),
                                type=parserutils.type_file,
                                help=cls.HELP)
        else:
            parser.set_defaults(task=[symbols.TOTENG])
        parser.add_argument(cls.FLAG_DATA_FILE,
                            metavar=cls.FLAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='The file of the structure and force field.')
        parser.add_argument(parserutils.FLAG_TASK,
                            type=str.lower,
                            choices=cls.TASK_CHOICES,
                            nargs='+',
                            help=cls.TASK_HELP)
        parser.add_argument(
            cls.FLAG_LAST_PCT,
            type=LastPct.type,
            default=LastPct(0.2),
            help=f"{', '.join(cls.LAST_FRM)} average results from this last "
            "percentage to the end.")
        parser.add_argument(cls.FLAG_SLICE,
                            metavar='START END STEP',
                            action=parserutils.Slice,
                            nargs='+',
                            help="Slice the input data before the analysis by "
                            "END, START END, or START END STEP.")


class LastPct(float):
    """
    Class to validate the last percentage argument and get the start index of
    the input data.
    """

    def getSidx(self, data, buffer=0):
        """
        Get the start index of the data.

        :param data tuple, or numpy.ndarray: on which the length is determined
        :param buffer int: the buffer step to be added to the start index
        :return int: the start index
        """
        num = len(data)
        sidx = min(max(num - 1, 0), round(num * (1 - self)))
        return max(0, sidx - buffer) if buffer else sidx

    @classmethod
    def type(cls, arg):
        """
        Check whether the argument can be converted to a percentage.

        :param arg str: the input argument.
        :return `cls`: the customized last percentage
        """
        value = parserutils.type_ranged_float(arg, include_top=False, top=1)
        return cls(value)


class TrajJob(LogJob):
    """
    Class to run lammps traj driver.
    """
    FILE = 'lmp_traj_driver.py'
    TASK_CHOICES = analyzer.TRAJ.keys()
    TASK_DEFAULT = analyzer.Density.NAME
    TASK_HELP = ', '.join(x.__doc__.strip().lower()
                          for x in analyzer.TRAJ.values())
    LAST_FRM = [x.NAME for x in [analyzer.MSD, analyzer.RDF]]
    FLAG = 'traj'
    HELP = 'Custom dump file to analyze.'

    def addfiles(self):
        """
        Set arguments to analyze the custom dump file.
        """
        args = super().addfiles()
        log_file = args.pop(0)
        return [self.getTrajFile(log_file)] + args

    def getTrajFile(self, log_file):
        """
        Set the args with the trajectory file from the log file.
        """
        with open(log_file, 'r') as fh:
            for line in fh:
                match = re.match(lammpsfix.DUMP_RE, line)
                if not match:
                    continue
                return match.group(2)

    @classmethod
    def add_arguments(cls, parser, positional=False):
        """
        Add job specific arguments to the parser.

        :param parser ArgumentParser: the parse to add arguments
        :param positional bool: whether to add positional arguments
        """
        super().add_arguments(parser, positional=positional)
        parser.set_defaults(task=[cls.TASK_DEFAULT])
        parser.add_argument('-sel', help=f'The element of the selected atoms.')
        if positional:
            parser.validators.add(cls.TrajValidator)

    class TrajValidator(parserutils.Validator):

        def run(self):
            """
            Validate the command options.

            :raise ValueError: no data file with data-requested tasks.
            """
            tasks = set(self.options.task)
            data_rqd_tasks = tasks.intersection(analyzer.DATA_RQD)
            if data_rqd_tasks and not self.options.data_file:
                raise ValueError(f"Please specify {TrajJob.FLAG_DATA_FILE} to"
                                 f" run {', '.join(data_rqd_tasks)} tasks.")


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
        self.jargs = self.job.doc[self.ARGS]

    def getCmd(self, write=True, **kwargs):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :return str: the command as str
        """
        msg = os.path.basename(self.job.statepoint[parserutils.FLAG_DIR])
        if self.cmd.comment:
            msg = f"{msg}: {self.cmd.comment}"
        return super().getCmd(prefix=f"echo \'# {msg}\'", write=write)

    def getArgs(self):
        """
        Get arguments that form command lines.

        :return list of str: each str is a command
        """
        self.setQuot()
        self.numCpu()
        self.setName()
        self.setDebug()
        self.setScreen()
        return self.args

    def setQuot(self):
        """
        Add quotes for str with special characters.

        Note: shlex.split("echo 'h(i)';echo wa", posix=False) splits by ;

        :return str: the quoted command
        """
        for idx, cmd in enumerate(self.args):
            quoted = map(super().quote, shlex.split(cmd, posix=False))
            self.args[idx] = symbols.SPACE.join(quoted)

    @property
    @functools.cache
    def args(self):
        """
        Return the arguments out of the cmd file.

        :return list of str: each str is a command
        """
        return self.param.getCmds()

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
            index = self.jargs.index(jobutils.FLAG_CPU)
        except ValueError:
            cpu_num = None
        else:
            cpu_num = f"{jobutils.FLAG_CPU} {self.jargs[index + 1]}"

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
            if self.FLAG_JOBNAME in cmd:
                continue
            match = test.FILE_RE.match(cmd)
            if not match:
                continue
            name = match.group(1)
            cmd += f" {self.FLAG_JOBNAME} {name}"
            self.args[idx] = cmd

    def setDebug(self):
        """
        Set the screen output.
        """
        value = jobutils.get_arg(self.jargs, jobutils.FLAG_DEBUG)
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
        scn = jobutils.get_arg(self.jargs, jobutils.FLAG_SCREEN)
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


class CheckJob(taskbase.Base):
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
        if DEBUG:
            raise ValueError(err)

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.jobname in self.doc.get(self.MESSAGE, {})


class TagJob(CheckJob):
    """
    This job class generates a new tag file (or updates the existing one).
    """

    def run(self):
        """
        Main method to run.
        """
        test.Tag(job=self.job).run()
        self.message = False


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
            jobname=self.options.jobname,
            interactive=self.options.interactive,
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
            statepoint.pop(parserutils.FLAG_SEED, None)
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
        dirs = [x.statepoint[parserutils.FLAG_DIR] for x in self.jobs]
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
