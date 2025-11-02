# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
parser utilities.
"""
import argparse
import functools
import os
import pathlib
import random

import numpy as np
import pandas as pd

from nemd import analyzer
from nemd import builtinsutils
from nemd import cru
from nemd import envutils
from nemd import is_debug
from nemd import jobutils
from nemd import lmpin
from nemd import ml
from nemd import rdkitutils
from nemd import sw
from nemd import symbols

FLAG_NAME = '-name'
FLAG_CPU = jobutils.FLAG_CPU


class Base:
    """
    Typer Base.
    """

    def __init__(self, arg):
        """
        :param arg str: original input argument.
        """
        self.arg = arg
        self.typed = None

    @classmethod
    def partial(cls, *args, **kwargs):
        """
        Return a partial object.

        return 'functools.partial': the partial object.
        """
        return functools.partial(cls.type, *args, **kwargs)

    @classmethod
    def type(cls, *args, **kwargs):
        """
        Type and check.

        :return any: the typed.
        """
        obj = cls(*args, **kwargs)
        obj.run()
        return obj.typed

    def run(self):
        """
        Type and check.
        """
        self.typed = self.arg

    @staticmethod
    def error(msg):
        """
        Raise ArgumentTypeError.

        :param msg str: the message to raise.
        :raise ArgumentTypeError: raise the error.
        """
        raise argparse.ArgumentTypeError(msg)


class Path(Base):
    """
    Type namepath and check existence.
    """
    FILE = 'file'
    DIRECTORY = 'directory'

    def __init__(self, *args, atype='path'):
        """
        :param atype str: the type of the namepath.
        """
        super().__init__(*args)
        self.atype = atype

    def run(self):
        """
        Check existence.

        :raise ArgumentTypeError: if the requested type of namepath not found.
        """
        self.typed = pathlib.Path(self.arg)
        match self.atype:
            case self.FILE:
                existed = self.typed.is_file()
            case self.DIRECTORY:
                existed = self.typed.is_dir()
            case _:
                existed = self.typed.exists()
        if not existed:
            self.error(f'{self.typed} is not an existing {self.atype}')

    @classmethod
    def typeFile(cls, *args, atype=FILE):
        """
        Check file existence.

        :return 'PosixPath': the existing file path.
        """
        return cls.type(*args, atype=atype)

    @classmethod
    def typeDir(cls, *args, atype=DIRECTORY):
        """
        Check directory existence.

        :return 'PosixPath': the existing directory path.
        """
        return cls.type(*args, atype=atype)


class Bool(Base):
    """
    Bool typer.
    """

    def run(self):
        """
        Check and convert to a boolean.
        """
        match self.arg.lower():
            case 'y' | 'yes' | 't' | 'true' | 'on' | '1':
                self.typed = True
            case '' | 'n' | 'no' | 'f' | 'false' | 'off' | '0':
                self.typed = False
            case _:
                self.error(f'Cannot convert {self.arg} to a boolean')


class Range(Base):
    """
    Numeric Range.
    """

    def __init__(self,
                 *args,
                 bot=None,
                 top=None,
                 incl_bot=True,
                 incl_top=True):
        """
        :param bot `float``: the lower bound of the range.
        :param top `float`: the upper bound of the range.
        :param incl_bot bool: whether to allow the lower bound.
        :param incl_top bool: whether to allow the upper bound.
        """
        super().__init__(*args)
        self.bot = bot
        self.top = top
        self.incl_bot = incl_bot
        self.incl_top = incl_top

    def run(self):
        """
        Check whether the float is within the range.
        """
        if self.bot is not None:
            if self.typed < self.bot:
                self.error(f'{self.typed} < {self.bot}')
            if not self.incl_bot and self.typed == self.bot:
                self.error(f'{self.typed} == {self.bot}')
        if self.top is not None:
            if self.typed > self.top:
                self.error(f'{self.typed} > {self.top}')
            if not self.incl_top and self.typed == self.top:
                self.error(f'{self.typed} == {self.top}')


class Int(Range):
    """"
    Integer typer.
    """

    def run(self):
        """
        Type int and check range.
        """
        try:
            self.typed = int(self.arg)
        except ValueError:
            self.error(f'Cannot convert {self.arg} to an integer')
        super().run()

    @classmethod
    def typeSlice(cls, arg):
        """
        Check and convert to an integer or None.

        :param arg str: the input argument.
        :return `int`: the converted integer or None.
        """
        return None if arg.lower() == 'none' else cls.type(arg)

    @classmethod
    def typeNonnegative(cls, *args, bot=0, **kwargs):
        """
        Check and convert to a nonnegative integer. (see init)

        :return `int: the converted positive integer.
        """
        return cls.type(*args, bot=bot, **kwargs)

    @classmethod
    def typePositive(cls, *args, incl_bot=False, **kwargs):
        """
        Check and convert to a positive integer. (see init)

        :return `int: the converted positive integer.
        """
        return cls.typeNonnegative(*args, incl_bot=incl_bot, **kwargs)

    @classmethod
    def typeSeed(cls, *args, bot=0, top=symbols.MAX_INT32, **kwargs):
        """
        Check, convert, and set a random seed. (see init)

        :return `int`: the converted random seed.
        """
        seed = cls.type(*args, bot=bot, top=top, **kwargs)
        np.random.seed(seed)
        random.seed(seed)
        return seed


class Float(Range):
    """"
    Integer typer.
    """

    def run(self):
        """
        Type float and check range.
        """
        try:
            self.typed = float(self.arg)
        except ValueError:
            self.error(f'Cannot convert {self.arg} to a float')
        super().run()

    @classmethod
    def typeNonnegative(cls, *args, bot=0, **kwargs):
        """
        Check and convert to a non-negative float. (see init)

        :return `float`: the converted non-negative value.
        """
        return cls.type(*args, bot=bot, **kwargs)

    @classmethod
    def typePositive(cls, *args, incl_bot=False, **kwargs):
        """
        Check and convert to a positive float. (see init)

        :return `float`: the converted positive value.
        """
        return cls.typeNonnegative(*args, incl_bot=incl_bot, **kwargs)


class Smiles(Base):
    """
    Type smiles and check.
    """

    def __init__(self, *args, allow_reg=True, canonize=True):
        """
        :param allow_reg bool: whether to allow regular molecule (without wildcard).
        :param canonize bool: whether to canonize the SMILES.
        """
        super().__init__(*args)
        self.allow_reg = allow_reg
        self.canonize = canonize

    def run(self):
        """
        Check whether the smiles can be converted to molecule (and/or)
        constitutional repeating units.
        """
        try:
            mol = cru.Mol.MolFromSmiles(self.arg,
                                        allow_reg=self.allow_reg,
                                        united=False)
        except (ValueError, TypeError) as err:
            self.error(f"Invalid SMILES: {self.arg}\n{err}")
        try:
            mol.run()
        except cru.MoietyError as err:
            # 1) regular molecule(s) found but allow_reg=False
            # 2) constitutional repeating units and regular molecules are mixed
            # 3) constitutional repeating units are insufficient to build a polymer
            self.error(str(err))
        self.typed = mol.getSmiles(canonize=self.canonize)


class LastPct(float):
    """
    The last percentage class to get the start index.
    """

    @classmethod
    def type(cls, arg):
        """
        Check whether the argument can be converted to a percentage.

        :param arg str: the input argument.
        :return `cls`: the customized last percentage
        """
        return cls(Float.typePositive(arg, top=1))

    def getSidx(self, data, buffer=0):
        """
        Get the start index of the data.

        :param data list: on which the length is determined
        :param buffer int: the buffer step to be added to the start index
        :return int: the start index
        """
        num = len(data)
        sidx = min(max(num - 1, 0), round(num * (1 - self)))
        return max(0, sidx - buffer) if buffer else sidx


class Cpu(list):
    """
    Cpu numbers for total and per-job.
    """

    def __init__(self, *args, forced=None):
        """
        :param forced bool: whether the cpu number if forced by users.
        """
        super().__init__(*args)
        self.forced = forced
        if forced is None:
            self.forced = bool(self)

    def set(self, jobs):
        """
        Set the cpu number per job.

        :param list: the jobs.
        """
        if len(self) == 1:
            # Evenly distribute among subjobs if only total cpu specified
            self.append(round(self[0] / len(jobs)) or 1)


class Action(argparse.Action):
    """
    Action on multiple values after the type check.
    """

    def __call__(self, parser, options, values, option_string=None):
        """
        Call this on parsing the arguments.

        :param parser `argparse.ArgumentParser`: the parser object.
        :param options 'argparse.Namespace': partially parsed arguments.
        :param values list: the values to be parsed.
        :param option_string str: the option string (e.g., -flag)
        """
        try:
            setattr(options, self.dest, self.doTyping(*values))
        except argparse.ArgumentTypeError as err:
            parser.error(f"{err} ({self.dest})")

    def doTyping(self, *args):
        """
        Check the input values.

        :return tuple: the modified values.
        """
        return args

    def error(self, msg):
        """
        Raise ArgumentTypeError with the message.

        :raise argparse.ArgumentTypeError: the raised error
        """
        raise argparse.ArgumentTypeError(msg)


class LmpLogAction(Action):
    """
    Action on a force field name optionally followed by a water model name.
    """

    def doTyping(self, *args):
        """
        Return the real tasks with 'all' replaced.

        :return tuple: real task names.
        """
        return tuple(LmpLog.TASKS[:-1]) if symbols.ALL in args else args


class ForceFieldAction(Action):
    """
    Action on a force field name optionally followed by a water model name.
    """

    def doTyping(self, name, *args):
        """
        Check the force field and water model.

        :param name str: the force field name
        :return tuple: force field name, additional arguments
        """
        match name:
            case symbols.SW:
                if args and not sw.get_file(*args):
                    self.error(f"Choose from {sw.NAME_ELEMENTS} sub-lists")
            case symbols.OPLSUA:
                if not args:
                    args = [symbols.TIP3P]
                if args[0] not in symbols.WMODELS:
                    self.error(f"Choose water models from {symbols.WMODELS}")
            case _:
                self.error(f"Choose force field from {symbols.FF_NAMES}")
        return name, *args


class SliceAction(Action):
    """
    Action on slice integers: 1) END; 2) START STEP; 3) START END STEP
    """

    def doTyping(self, *args):
        """
        Check the slice arguments.

        :param args list of str: the arguments for the slice function.
            (1: END; 2: START, END; 3: START, END, STEP)
        :return tuple: start, stop, and step
        """
        if len(args) > 3:
            self.error(f"More than 3 argument found.")
        return args


class StructAction(Action):
    """
    Action on smile str optionally followed by a float.
    """

    def doTyping(self, smiles, value=None):
        """
        Check the slice arguments.

        :param smiles str: the smiles str to select a substructure.
        :param value str: the target value for the substructure to be set.
        :return tuple: the smiles str, (and the target value).
        """
        smiles = Smiles.type(smiles)
        return (smiles, Float.type(value)) if value is not None else (smiles, )


class ThreeAction(Action):
    """
    Action that allows three values.
    """

    def doTyping(self, *args):
        """
        Check and return the first three arguments.

        :return tuple: the first three arguments
        """
        if len(args) == 1:
            return args * 3
        if len(args) == 2:
            self.error(f"{args} contains two values")
        return args[:3]


class Valid:
    """
    Validate cross-flag arguments after parse_args().
    """

    def __init__(self, options):
        """
        param options 'argparse.Namespace': Command line options.
        """
        self.options = options

    def run(self):
        """
        Validate cross-flag values.
        """
        pass


class CpuValid(Valid):
    """
    Class to valid the cpu parameters.
    """

    def run(self, cpu=1):
        """
        See parent.
        """
        if self.options.CPU:
            if len(self.options.CPU) > 2:
                raise ValueError(f'More than 2 arguments found. ({FLAG_CPU})')
            self.options.CPU = Cpu(self.options.CPU)
            return
        if not self.options.DEBUG:
            # Debug mode: 1 cpu to avoid threading -> stdout available for pdb
            # Production mode: 75% of cpu count as total to avoid overloading
            cpu = max([round(os.cpu_count() * 0.75), 1])
        self.options.CPU = Cpu([cpu], forced=False)


class MdValid(Valid):
    """
    Class to valid the damp parameters.
    """

    def run(self):
        """
        Main method to run the validation.
        """
        if self.options.tdamp is None:
            self.options.tdamp = self.options.timestep * 100
        if self.options.pdamp is None:
            self.options.pdamp = self.options.timestep * 1000


class MolValid(Valid):
    """
    Class to validate molecule related arguments after parse_args().
    """

    def run(self):
        """
        Main method to run the validation.
        """
        self.cruNum()
        self.molNum()

    def cruNum(self):
        """
        Validate (or set) the number of repeat units.

        :raise ValueError: if cru_num number is different from the cru number.
        """
        if self.options.cru_num is None:
            self.options.cru_num = [1] * len(self.options.cru)
            return
        if len(self.options.cru_num) == len(self.options.cru):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.cru_num)} cru found.')

    def molNum(self):
        """
        Validate (or set) the number of molecules.
        """
        if self.options.mol_num is None:
            self.options.mol_num = [1] * len(self.options.cru_num)
            return
        if len(self.options.mol_num) == len(self.options.cru_num):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.mol_num)} molecules found.')


class LmpValid(Valid):
    """
    Class to validate lammps related arguments after parse_args().
    """

    def run(self):
        """
        When not provided, try to locate the data file based on the input script.

        :raises FileNotFoundError: if data file is required but doesn't exist.
        """
        if self.options.data_file:
            return

        with open(self.options.inscript, 'r') as fh:
            contents = fh.read()
        matched = lmpin.Script.READ_DATA_RE.search(contents)
        if not matched:
            return
        data_file = matched.group(1)
        if not os.path.isfile(data_file):
            # Data file not found relative to the current directory
            dirname = os.path.dirname(self.options.inscript)
            data_file = os.path.join(dirname, data_file)

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"No data file {data_file} found.")

        self.options.data_file = data_file


class TrajValid(Valid):
    """
    Class to validate trajectory related arguments after parse_args().
    """

    def run(self):
        """
        Validate the command options.

        :raise ValueError: tasks request data file but not provided.
        """
        tasks = set(self.options.task)
        rqd_data = tasks.intersection([x.name for x in analyzer.DATA_RQD])
        if rqd_data and not self.options.data_file:
            raise ValueError(f"Please specify {Lammps.FLAG_DATA_FILE} to"
                             f" run {', '.join(rqd_data)} tasks.")


class RegValid(Valid):
    """
    Class to validate Regression arguments.
    """
    LOGIT = ml.Reg.LOGIT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def run(self):
        """
        Validate the command options.
        """
        self.read()
        self.columns()
        self.method()

    def read(self):
        """
        Read the input data.

        :raise ValueError: read_csv errors
        """
        try:
            self.data = pd.read_csv(self.options.data)
        except pd.errors.EmptyDataError as err:
            raise ValueError(f"{err} ({self.options.data})")

    def columns(self):
        """
        Validate columns.

        :raise ValueError: not enough columns
        """
        if self.data.shape[1] < 2:
            raise ValueError(f"X, y columns not found")
        if self.data.select_dtypes(include=['number']).shape[1] < 2:
            raise ValueError(f"Less than two columns after excluding "
                             f"non-numeric values")

    def method(self):
        """
        Validate method.
        """
        if self.LOGIT in self.options.method and \
                not self.data.iloc[:, -1].isin([0, 1]).all().all():
            self.options.method.remove(self.LOGIT)


class Driver(argparse.ArgumentParser, builtinsutils.Object):
    """
    Parser with job arguments.
    """
    FLAG_INTERAC = jobutils.FLAG_INTERAC
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    FLAG_PYTHON = jobutils.FLAG_PYTHON
    FLAG_DEBUG = jobutils.FLAG_DEBUG
    JFLAGS = [FLAG_INTERAC, FLAG_JOBNAME, FLAG_PYTHON, FLAG_CPU, FLAG_DEBUG]

    def __init__(self,
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                 descr=None,
                 valids=None,
                 **kwargs):
        """
        :param formatter_class 'ArgumentDefaultsHelpFormatter': the formatter
            class to display the customized help message
        :param descr str: the script description displayed as the help message.
        :param valids set: across-argument valids after parse_args()
        """
        if descr is not None:
            kwargs.update(description=descr)
        super().__init__(formatter_class=formatter_class, **kwargs)
        self.valids = set() if valids is None else valids
        self.setUp()
        self.add(self, positional=True)
        self.addJob()

    @classmethod
    def setUp(self):
        """
        Set up the parser.
        """
        pass

    @classmethod
    def add(cls, *args, **kwargs):
        """
        Add arguments to the parser.
        """
        pass

    def addJob(self):
        """
        Add job control related flags.
        """
        if self.FLAG_JOBNAME in self.JFLAGS:
            default = envutils.Env().jobname or self.name
            self.add_argument(jobutils.FLAG_NAME,
                              default=default,
                              help=argparse.SUPPRESS)
            self.add_argument(self.FLAG_JOBNAME,
                              default=default,
                              help='Name output files.')
        if self.FLAG_INTERAC in self.JFLAGS:
            self.addBool(self.FLAG_INTERAC,
                         default=envutils.Env().interac,
                         help='Pause for user input.')
        if self.FLAG_PYTHON in self.JFLAGS:
            self.add_argument(self.FLAG_PYTHON,
                              default=envutils.Mode(),
                              type=Int.partial(bot=-1, top=2),
                              help='0: native; 1: compiled; 2: cached.')
        if FLAG_CPU in self.JFLAGS:
            self.add_argument(FLAG_CPU,
                              metavar='INT',
                              type=Int.typePositive,
                              nargs='+',
                              help='Total number of CPUs [CPUs per task].')
            self.valids.add(CpuValid)
        if self.FLAG_DEBUG in self.JFLAGS:
            self.addBool(
                self.FLAG_DEBUG,
                default=is_debug(),
                help='True allows additional printing and output files; '
                'False disables the mode.')

    def addBool(self, flag, default=None, help=None):
        """
        Add bool type argument: both [-flag] and [-flag True] set True;
        [-flag False] sets False.

        :param flag str: the flag to dd
        :param default any: the default
        :param help str: help message
        """
        self.add_argument(flag,
                          default=default,
                          nargs='?',
                          const=True,
                          type=Bool.type,
                          choices=[True, False],
                          help=help)

    @classmethod
    def addSeed(cls, parser, **kwargs):
        """
        Get a random seed, set the state, and add to the parser.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        default = np.random.randint(0, symbols.MAX_INT32)
        Int.typeSeed(default)
        parser.add_argument(jobutils.FLAG_SEED,
                            type=Int.typeSeed,
                            default=default,
                            help='Set random state.')

    def suppress(self, flags=None, **kwargs):
        """
        Supress the help messages of specified arguments.

        :param flags list: the arguments to be suppressed.
        :param kwargs dict: are the arguments to be suppressed, and the values
            are the default values to be used.
        """
        if flags is None:
            flags = []
        defaults = {x.lstrip('-'): y for x, y in kwargs.items()}
        self.set_defaults(**defaults)
        flags = {*flags, *kwargs.keys()}
        for axn in self._actions:
            if not flags.intersection(axn.option_strings):
                continue
            axn.help = argparse.SUPPRESS

    def parse_args(self, args=None, **kwargs):
        """
        Parse the command line arguments and perform the validations.

        :param args list: command line arguments.
        :rtype 'argparse.Namespace': the parsed arguments.
        """
        options = super().parse_args(args=args, **kwargs)
        for Valid in self.valids:
            val = Valid(options)
            try:
                val.run()
            except (ValueError, FileNotFoundError) as err:
                self.error(err)
        return options


class Bldr(Driver):
    """
    Parser with builder arguments.
    """
    FLAG_SUBSTRUCT = '-substruct'

    @classmethod
    def addBldr(cls, parser, **kwargs):
        """
        Set up the builder arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        parser.add_argument(cls.FLAG_SUBSTRUCT,
                            metavar='SMILES (VALUE)',
                            nargs='+',
                            action=StructAction,
                            help='set or measure the substructure geometry.')
        parser.add_argument(
            '-force_field',
            nargs='+',
            action=ForceFieldAction,
            default=symbols.OPLSUA_TIP3P,
            help=f'The force field type: 1) {symbols.OPLSUA} '
            f'[{symbols.PIPE.join(symbols.WMODELS)}]; 2) {symbols.SW}')


class MolBase(Bldr):
    """
    Parser with basic molecular arguments.
    """
    FLAG_CRU = 'cru'
    FLAG_CRU_NUM = '-cru_num'
    FLAG_MOL_NUM = '-mol_num'
    FLAG_BUFFER = '-buffer'

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the basic molecular arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param seed `bool`: whether this is a seeding job in the workflow
        """
        parser.add_argument(cls.FLAG_CRU,
                            metavar=cls.FLAG_CRU.upper(),
                            type=Smiles.type,
                            nargs='+',
                            help='SMILES of the constitutional repeat units.')
        parser.add_argument(
            cls.FLAG_CRU_NUM,
            type=Int.typePositive,
            nargs='+',
            help='Number of constitutional repeat unit per polymer')
        parser.add_argument(cls.FLAG_MOL_NUM,
                            type=Int.typePositive,
                            nargs='+',
                            help='Number of molecules in the amorphous cell')
        # The buffer distance between molecules in the grid cell
        parser.add_argument(cls.FLAG_BUFFER,
                            type=Float.typePositive,
                            help=argparse.SUPPRESS)
        parser.valids.add(MolValid)
        cls.addBldr(parser, **kwargs)
        Md.add(parser)


class Md(Driver):
    """
    Parser with molecular dynamics arguments.
    """
    FLAG_TIMESTEP = '-timestep'
    FLAG_STEMP = '-stemp'
    FLAG_TEMP = '-temp'
    FLAG_TDAMP = '-tdamp'
    FLAG_PRESS = '-press'
    FLAG_PDAMP = '-pdamp'
    FLAG_RELAX_TIME = '-relax_time'
    FLAG_PROD_TIME = '-prod_time'
    FLAG_PROD_ENS = '-prod_ens'
    FLAG_STAT = '-stat'
    FLAG_NO_MINIMIZE = '-no_minimize'

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the molecular dynamics arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param positional `bool`: whether add positional arguments.
        """
        parser.add_argument(cls.FLAG_TIMESTEP,
                            metavar='fs',
                            type=Float.typePositive,
                            default=1,
                            help=f'Timestep for the MD simulation.')
        # 'Initialize the atoms with this temperature.'
        parser.add_argument(cls.FLAG_STEMP,
                            metavar='K',
                            type=Float.typePositive,
                            default=10,
                            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_TEMP,
                            metavar='K',
                            type=Float.typeNonnegative,
                            default=300,
                            help=f'The equilibrium temperature target. A zero '
                            f'for single point energy.')
        # Temperature damping parameter in time unit
        parser.add_argument(cls.FLAG_TDAMP,
                            type=Float.typePositive,
                            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_PRESS,
                            metavar='atm',
                            type=float,
                            default=1,
                            help="The equilibrium pressure target.")
        # Pressure damping parameter in time unit
        parser.add_argument(cls.FLAG_PDAMP,
                            type=Float.typePositive,
                            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_RELAX_TIME,
                            metavar='ns',
                            type=Float.typeNonnegative,
                            default=1,
                            help='Relaxation simulation time.')
        parser.add_argument(cls.FLAG_PROD_TIME,
                            metavar='ns',
                            type=Float.typePositive,
                            default=1,
                            help='Production simulation time.')
        parser.add_argument(cls.FLAG_PROD_ENS,
                            choices=lmpin.Script.ENSEMBLES,
                            default=lmpin.Script.NVE,
                            help='Production ensemble.')
        parser.add_argument(cls.FLAG_STAT,
                            choices=lmpin.RampUp.STATS,
                            default=lmpin.RampUp.NOSE_HOOVER,
                            help='Thermostat and barostat style.')
        cls.addSeed(parser)
        # Skip the structure minimization step
        parser.add_argument(cls.FLAG_NO_MINIMIZE,
                            action='store_true',
                            help=argparse.SUPPRESS)
        parser.valids.add(MdValid)


class MolBldr(MolBase):
    """
    Parser with molecule arguments.
    """

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the single-molecule arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        super().add(parser, **kwargs)
        parser.suppress(buffer=f"{symbols.DEFAULT_CUT * 4}",
                        mol_num=[1],
                        temp=0,
                        timestep=1,
                        press=1,
                        relax_time=0,
                        prod_time=0,
                        prod_ens=lmpin.Script.NVE)


class AmorpBldr(MolBase):
    """
    Parser with amorphous-cell arguments.
    """
    FLAG_DENSITY = '-density'
    FLAG_METHOD = '-method'
    GRID = 'grid'
    PACK = 'pack'
    GROW = 'grow'
    METHODS = [GRID, PACK, GROW]

    @classmethod
    def addBldr(cls, parser, **kwargs):
        """
        Set up the amorphous-cell arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        parser.add_argument(
            cls.FLAG_DENSITY,
            metavar='g/cm^3',
            type=Float.partial(bot=0, incl_bot=False, top=30),
            default=0.5,
            help=f'The density used for {cls.PACK} and {cls.GROW} cells.')
        parser.add_argument(
            cls.FLAG_METHOD,
            choices=cls.METHODS,
            default=cls.GROW,
            help=f'place molecules into the space {cls.GRID}; {cls.PACK} '
            f'molecules with random rotation and translation; {cls.GROW} '
            'molecules by rotating rigid fragments.')
        super().addBldr(parser, **kwargs)


class XtalBldr(Bldr):
    """
    Parser with crystal arguments.
    """
    FLAG_DIMENSION = '-dimension'
    FLAG_SCALED_FACTOR = '-scale_factor'
    ONES = (1, 1, 1)

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the crystal arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        # FIXME: support more choices based on crystals.Crystal.builtins
        parser.add_argument(
            FLAG_NAME,
            default='Si',
            choices=['Si'],
            help='Name to retrieve the crystal structure from the database.')
        parser.add_argument(
            cls.FLAG_DIMENSION,
            metavar='INT',
            default=cls.ONES,
            nargs='+',
            type=int,
            action=ThreeAction,
            help='Duplicate the unit cell by these factors to generate the '
            'supercell.')
        parser.add_argument(
            cls.FLAG_SCALED_FACTOR,
            default=cls.ONES,
            nargs='+',
            type=Float.typePositive,
            action=ThreeAction,
            help='Each lattice vector is scaled by the corresponding factor.')
        super().add(parser, **kwargs)
        parser.set_defaults(force_field=[symbols.SW])
        Md.add(parser, **kwargs)


class Lammps(Driver):
    """
    Parser with lammps arguments.
    """
    FLAG_INSCRIPT = 'inscript'
    FLAG_DATA_FILE = '-data_file'

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the molecular dynamics arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        """
        parser.add_argument(cls.FLAG_INSCRIPT,
                            metavar=cls.FLAG_INSCRIPT.upper(),
                            type=Path.typeFile,
                            help='Simulation protocol.')
        parser.add_argument(cls.FLAG_DATA_FILE,
                            type=Path.typeFile,
                            help='Structure and force field.')
        parser.valids.add(LmpValid)


class LmpLog(Lammps):
    """
    Parser with lammps-log arguments.
    """
    FLAG = 'log'
    FLAG_LAST_PCT = '-last_pct'
    FLAG_SLICE = '-slice'
    LAST_FRM = [x.name for x in analyzer.THERMO]
    TASKS = LAST_FRM + [symbols.ALL]
    HELP = 'Searches, combines and averages thermodynamic info.'

    @classmethod
    def add(cls,
            parser,
            positional=False,
            task=analyzer.TotEng.name,
            action=LmpLogAction):
        """
        Add job specific arguments to the parser.

        :param parser argparse.ArgumentParser: the parse to add arguments.
        :param positional `bool`: whether add positional arguments.
        :param task str: the default task.
        :param action `Action`: task action.
        """
        if positional:
            parser.add_argument(cls.FLAG,
                                metavar=cls.FLAG.upper(),
                                type=Path.typeFile,
                                help=f'The {cls.FLAG} file to analyze.')
        parser.add_argument(jobutils.FLAG_TASK,
                            type=str.lower,
                            nargs='+',
                            choices=cls.TASKS,
                            action=action,
                            default=[task],
                            help=cls.HELP)
        parser.add_argument(cls.FLAG_DATA_FILE,
                            type=Path.typeFile,
                            help='Structure and force field.')
        parser.add_argument(
            cls.FLAG_LAST_PCT,
            type=LastPct.type,
            default=LastPct(0.2),
            help=f"{', '.join(cls.LAST_FRM)} average results from this last "
            "percentage to the end.")
        parser.add_argument(
            cls.FLAG_SLICE,
            metavar='START END STEP',
            type=Int.typeSlice,
            action=SliceAction,
            nargs='+',
            help="Slice the input data by END, START END, or START END STEP.")


class LmpTraj(LmpLog):
    """
    Parser with lammps-trajectory arguments.
    """
    FLAG = 'trj'
    TASKS = [x.name for x in analyzer.TRAJ]
    HELP = ', '.join(x.__doc__.strip(' .\n').lower() for x in analyzer.TRAJ)
    LAST_FRM = [x.name for x in [analyzer.MSD, analyzer.RDF]]

    @classmethod
    def add(cls,
            parser,
            positional=False,
            task=analyzer.Density.name,
            action=None):
        """
        See parent.
        """
        super().add(parser, positional=positional, task=task, action=action)
        parser.add_argument('-sel', help=f'The element of the selected atoms.')
        if positional:
            parser.valids.add(TrajValid)
            parser.add_argument(Md.FLAG_TIMESTEP,
                                metavar='fs',
                                type=Float.typePositive,
                                default=1,
                                help='Trajectory timetep.')


class Reg(Driver):
    """
    Parser with ml regression arguments.
    """
    FLAG_DATA = 'data'
    FLAG_METHOD = '-method'
    FLAG_DEGREE = '-degree'
    FLAG_TREE_NUM = '-tree_num'
    FLAG_TEST_SIZE = '-test_size'

    @classmethod
    def add(cls, parser, **kwargs):
        """
        See parent.
        """
        parser.add_argument(cls.FLAG_DATA,
                            type=Path.typeFile,
                            help='The csv file.')
        names = ", ".join([f"{y} ({x})" for x, y in ml.Reg.NAMES.items()])
        parser.add_argument(cls.FLAG_METHOD,
                            default=[ml.Reg.LR],
                            choices=ml.Reg.NAMES,
                            nargs='+',
                            help=f'Regression method: {names}')
        parser.add_argument(cls.FLAG_DEGREE,
                            default=2,
                            type=Int.typePositive,
                            help=f'The max polynomial degree.')
        parser.add_argument(cls.FLAG_TREE_NUM,
                            default=100,
                            type=Int.typePositive,
                            help=f'The number of trees in the forest.')
        parser.add_argument(cls.FLAG_TEST_SIZE,
                            default=0.2,
                            type=Float.partial(top=1,
                                               bot=0,
                                               incl_bot=False,
                                               incl_top=False),
                            help=f'The test size on splitting.')
        parser.valids.add(RegValid)
        cls.addSeed(parser)


class Workflow(Driver):
    """
    Parser with workflow arguments.
    """
    FLAG_STATE_NUM = '-state_num'
    FLAG_CLEAN = '-clean'
    FLAG_JTYPE = '-jtype'
    FLAG_SCREEN = jobutils.FLAG_SCREEN
    SERIAL = 'serial'
    PARALLEL = jobutils.PARALLEL
    JOB = jobutils.JOB
    WFLAGS = [FLAG_STATE_NUM, FLAG_CLEAN, FLAG_JTYPE, FLAG_SCREEN]
    SCREENS = [symbols.OFF, SERIAL, PARALLEL, JOB]

    def __init__(self, *args, conflict_handler='resolve', **kwargs):
        """
        Set up the parser.

        :param conflict_handler str: the action when two actions with the same
            option string
        """
        super().__init__(*args, conflict_handler=conflict_handler, **kwargs)
        self.addWorkflow()

    def addWorkflow(self):
        """
        Add workflow related flags.
        """
        if self.FLAG_STATE_NUM in self.WFLAGS:
            self.add_argument(
                self.FLAG_STATE_NUM,
                default=1,
                type=Int.typePositive,
                help='Total number of the states (e.g., dynamical system).')
        if self.FLAG_JTYPE in self.WFLAGS:
            # Task jobs have to register outfiles to be considered as completed
            # Aggregator jobs collect results from finished task jobs
            self.add_argument(
                self.FLAG_JTYPE,
                nargs='+',
                choices=[symbols.TASK, symbols.AGGREGATOR],
                default=[symbols.TASK, symbols.AGGREGATOR],
                help=f'{symbols.TASK}: run tasks and register files; '
                f'{symbols.AGGREGATOR}: collect results.')
            self.add_argument('-prj_path',
                              type=Path.typeDir,
                              help='Collect jobs from this directory.')
        if self.FLAG_CLEAN in self.WFLAGS:
            self.add_argument(self.FLAG_CLEAN,
                              action='store_true',
                              help='Clean previous workflow results.')
        if self.FLAG_SCREEN in self.WFLAGS:
            self.add_argument(
                self.FLAG_SCREEN,
                choices=self.SCREENS,
                default=symbols.OFF,
                help=f'Screen status: {symbols.OFF}, {self.SERIAL}, '
                f'{self.PARALLEL}, and {self.JOB} details')
