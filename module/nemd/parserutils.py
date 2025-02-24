# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
parser utilities.
"""
import argparse
import functools
import os
import random
import re

from nemd import DEBUG
from nemd import analyzer
from nemd import cru
from nemd import envutils
from nemd import jobutils
from nemd import lammpsfix
from nemd import np
from nemd import rdkitutils
from nemd import sw
from nemd import symbols


def type_file(arg):
    """
    Check file existence.

    :param arg str: the input argument.
    :return str: the existing namepath of the file.
    :raise ArgumentTypeError: if the file is not found.
    """
    if os.path.isfile(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} not found.')


def type_dir(arg):
    """
    Check directory existence.

    :param arg str: the input argument.
    :return str: the existing directory path.
    :raise ArgumentTypeError: if the directory doesn't exist.
    """
    if os.path.isdir(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} is not an existing directory.')


def type_bool(arg):
    """
    Check and convert to a boolean.

    :param arg str: the input argument.
    :return str: the coverted boolean value.
    :raise ArgumentTypeError: if the arg is not a valid boolean value.
    """
    match arg.lower():
        case 'y' | 'yes' | 't' | 'true' | 'on' | '1':
            return True
        case 'n' | 'no' | 'f' | 'false' | 'off' | '0':
            return False
        case _:
            raise argparse.ArgumentTypeError(f'{arg} is not a valid boolean.')


def type_float(arg):
    """
    Check and convert to a float.

    :param arg str: the input argument.
    :return `float`: the converted float value.
    :raise ArgumentTypeError: argument cannot be converted to a float.
    """
    try:
        return float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Cannot convert {arg} to a float.')


def type_int(arg):
    """
    Check and convert to an integer.

    :param arg str: the input argument.
    :return `int:: the converted integer.
    :raise ArgumentTypeError: argument cannot be converted to an integer.
    """
    try:
        return int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Cannot convert {arg} to an integer')


def type_ranged(value,
                bottom=-symbols.MAX_INT32,
                top=symbols.MAX_INT32,
                included_bottom=True,
                include_top=True):
    """
    Check whether the float is within the range.

    :param value `float`: the value to be checked.
    :param bottom `float``: the lower bound of the range.
    :param top `float`: the upper bound of the range.
    :param included_bottom bool: whether the lower bound is allowed
    :param include_top bool: whether the upper bound is allowed
    :return `float`: the checked value.
    :raise ArgumentTypeError: argument is not within the range.
    """
    if included_bottom and value < bottom:
        raise argparse.ArgumentTypeError(f'{value} < {bottom}')
    if not included_bottom and value <= bottom:
        raise argparse.ArgumentTypeError(f'{value} <= {bottom}')
    if include_top and value > top:
        raise argparse.ArgumentTypeError(f'{value} > {top}')
    if not include_top and value >= top:
        raise argparse.ArgumentTypeError(f'{value} >= {top}')
    return value


def type_ranged_float(arg, **kwargs):
    """
    Check and convert to a float within the range.

    :param arg str: the input argument.
    :return `float`: the converted value within the range.
    """
    value = type_float(arg)
    return type_ranged(value, **kwargs)


def type_nonnegative_float(arg):
    """
    Check and convert to a non-negative float.

    :param arg str: the input argument.
    :return `float`: the converted non-negative value.
    """
    return type_ranged_float(arg, bottom=0)


def type_positive_float(arg, **kwargs):
    """
    Check and convert to a positive float.

    :param arg str: the input argument.
    :return `float`: the converted positive value.
    """
    return type_ranged_float(arg, bottom=0, included_bottom=False, **kwargs)


def type_positive_int(arg):
    """
    Check and convert to a positive integer.

    :param arg str: the input argument.
    :return `int: the converted positive integer.
    """
    value = type_int(arg)
    type_ranged(value, bottom=1)
    return value


def type_nonnegative_int(arg):
    """
    Check and convert to a nonnegative integer.

    :param arg str: the input argument.
    :return `int: the converted positive integer.
    """
    value = type_int(arg)
    type_ranged(value, bottom=0)
    return value


def type_screen(arg):
    """
    Set TQDM_DISABLE environmental variable.

    :param arg str: the input argument.
    :return `str: the input argument.
    """
    if arg == jobutils.PROGRESS:
        os.environ['TQDM_DISABLE'] = ''
    return arg


def type_random_seed(arg):
    """
    Check, convert, and set a random seed.

    :param arg str: the input argument.
    :return `int`: the converted random seed.
    """
    value = type_int(arg)
    type_ranged(value, bottom=0, top=symbols.MAX_INT32)
    np.random.seed(value)
    random.seed(value)
    return value


def type_smiles(arg):
    """
    Check and convert a smiles to a molecule.

    :param arg str: the input argument.
    :return `rdkit.Chem.rdchem.Mol: the converted molecule.
    :raise ArgumentTypeError: argument cannot be converted to a valid molecule.
    """
    try:
        value = rdkitutils.MolFromSmiles(arg)
    except (ValueError, TypeError) as err:
        raise argparse.ArgumentTypeError(f"Invalid SMILES: {arg}\n{err}")
    return value


def type_cru_smiles(arg, allow_reg=True, canonize=True):
    """
    Check whether the smiles can be converted to constitutional repeating units.

    :param arg str: the input argument
    :param allow_reg bool: whether to allow regular molecule (without wildcard).
    :param canonize bool: whether to canonize the SMILES.
    :return `rdkit.Chem.rdchem.Mol: the converted molecule.
    :raise ArgumentTypeError: when unable to build a molecule from the smiles
    """
    mol = cru.Mol(type_smiles(arg), allow_reg=allow_reg)
    try:
        mol.run()
    except cru.MoietyError as err:
        # 1) regular molecule(s) found but allow_reg=False
        # 2) constitutional repeating units and regular molecules are mixed
        # 3) constitutional repeating units are insufficient to build a polymer
        raise argparse.ArgumentTypeError(str(err))
    return mol.getSmiles(canonize=canonize)


class LastPct(float):
    """
    The last percentage class to get the start index.
    """

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

    @classmethod
    def type(cls, arg):
        """
        Check whether the argument can be converted to a percentage.

        :param arg str: the input argument.
        :return `cls`: the customized last percentage
        """
        value = type_positive_float(arg, include_top=False, top=1)
        return cls(value)


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
            parser.error(f"{err} ({option_string})")

    def doTyping(self, *args):
        """
        Check the input values.

        :return *args list: the modified values.
        """
        return args

    def error(self, msg):
        """
        Raise ArgumentTypeError with the message.

        :raise argparse.ArgumentTypeError: the raised error
        """
        raise argparse.ArgumentTypeError(msg)


class ForceFieldAction(Action):
    """
    Action on a force field name optionally followed by a water model name.
    """

    def doTyping(self, name, *args):
        """
        Check the force field and water model.

        :param name str: the force field name
        :return list: force field name, additional arguments
        """
        match name:
            case symbols.SW:
                if args and not sw.get_file(args):
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
        :return list of int: start, stop, and step
        """
        sliced = slice(*args)
        start = 0 if sliced.start is None else sliced.start
        step = 1 if sliced.step is None else sliced.step
        if step == 0 or start > sliced.stop:
            self.error(f"{args} invalid for slice.")
        return start, sliced.stop, step


class StructAction(Action):
    """
    Action on smile str optionally followed by a float.
    """

    def doTyping(self, smiles, value=None):
        """
        Check the slice arguments.

        :param smiles str: the smiles str to select a substructure.
        :param value str: the target value for the substructure to be set.
        :return str, float: the smiles str, and the target value.
        """
        type_smiles(smiles)
        if value is not None:
            value = type_float(value)
        return smiles, value


class ThreeAction(Action):
    """
    Argparse action that allow three values.
    """

    def doTyping(self, *args):
        """
        Check and return the first three arguments.

        :return tuple: the first three arguments
        """
        if len(args) == 1:
            return args * 3
        if len(args) == 2:
            self.error(f"{self.option_strings[0]} expects three values.")
        return args[:3]


class Validator:
    """
    Class to check cross-flag arguments after parse_args().
    """

    def __init__(self, options):
        """
        param options 'argparse.Namespace': Command line options.
        """
        self.options = options

    def run(self):
        """
        Check cross-flag values.
        """
        pass


class MolValidator(Validator):
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

        :raise ValueError: if the number of cru_num is not equal to the number
            of cru.
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


class LmpValidator(Validator):
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


class TrajValidator(Validator):
    """
    Class to validate trajectory related arguments after parse_args().
    """

    def run(self):
        """
        Validate the command options.

        :raise ValueError: no data file with data-requested tasks.
        """
        tasks = set(self.options.task)
        data_rqd_tasks = tasks.intersection(analyzer.DATA_RQD)
        if data_rqd_tasks and not self.options.data_file:
            raise ValueError(f"Please specify {Lammps.FLAG_DATA_FILE} to"
                             f" run {', '.join(data_rqd_tasks)} tasks.")


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    A customized formatter for the argument help message.
    """

    def add_usage(self, usage, actions, groups, prefix='Usage: '):
        """
        See parent class for details.
        :rtype:
        """
        super().add_usage(usage, actions, groups, prefix)


class Driver(argparse.ArgumentParser):
    """
    Parser with job arguments.
    """
    FLAG_INTERAC = jobutils.FLAG_INTERAC
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    FLAG_PYTHON = jobutils.FLAG_PYTHON
    FLAG_CPU = jobutils.FLAG_CPU
    FLAG_DEBUG = jobutils.FLAG_DEBUG
    JFLAGS = [FLAG_INTERAC, FLAG_JOBNAME, FLAG_PYTHON, FLAG_CPU, FLAG_DEBUG]

    def __init__(self,
                 file='name_seed.py',
                 formatter_class=ArgumentDefaultsHelpFormatter,
                 descr=None,
                 validators=None,
                 delay=False,
                 **kwargs):
        """
        :param file str: script filename which defines the default jobname.
        :param formatter_class 'ArgumentDefaultsHelpFormatter': the formatter
            class to display the customized help message
        :param descr str: the script description displayed as the help message.
        :param validators set: across-argument validators after parse_args()
        :param delay bool: delay the setup.
        """
        if descr is not None:
            kwargs.update(description=descr)
        super().__init__(formatter_class=formatter_class, **kwargs)
        self.file = file
        self.delay = delay
        self.validators = set() if validators is None else validators
        if self.delay:
            return
        self.setUp()
        self.add(self, append=False)
        self.addJob()

    @classmethod
    def setUp(self):
        """
        Set up the parser.
        """
        pass

    @classmethod
    def add(cls, parser, append=True):
        """
        Add arguments to the parser.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param append `bool`: whether this appends to a previous parser.
        """
        pass

    def addJob(self):
        """
        Add job control related flags.
        """
        if self.FLAG_JOBNAME in self.JFLAGS:
            self.add_argument(jobutils.FLAG_NAME,
                              default=self.name,
                              help=argparse.SUPPRESS)
            envutils.set_jobname_default(self.name)
            self.add_argument(self.FLAG_JOBNAME,
                              dest=self.FLAG_JOBNAME[1:],
                              default=self.name,
                              help='Name output files.')
        if self.FLAG_INTERAC in self.JFLAGS:
            self.add_argument(self.FLAG_INTERAC,
                              dest=self.FLAG_INTERAC[1:],
                              action='store_true',
                              help='Pause for user input.')
        if self.FLAG_PYTHON in self.JFLAGS:
            self.add_argument(self.FLAG_PYTHON,
                              default=envutils.CACHE_MODE,
                              dest=self.FLAG_PYTHON[1:],
                              choices=envutils.PYTHON_MODES,
                              help='0: native; 1: compiled; 2: cached.')
        if self.FLAG_CPU in self.JFLAGS:
            self.add_argument(
                self.FLAG_CPU,
                type=type_positive_int,
                nargs='+',
                dest=self.FLAG_CPU[1:],
                help='Total number of CPUs (the number for one task).')
        if self.FLAG_DEBUG in self.JFLAGS:
            self.add_argument(
                self.FLAG_DEBUG,
                default=DEBUG,
                nargs='?',
                const=True,
                type=type_bool,
                choices=[symbols.ON, symbols.OFF],
                dest=self.FLAG_DEBUG[1:],
                help=f'{symbols.ON}: allow additional printing and output files'
                f'; {symbols.OFF}: disable the mode.')

    @property
    @functools.cache
    def name(self):
        """
        Return the default jobname.

        :return str: the default jobname.
        """
        return jobutils.get_name(self.file)

    @staticmethod
    @functools.cache
    def getSeed():
        """
        Get a random seed and set the default state with it.

        :return int: the default random seed.
        """
        value = np.random.randint(0, symbols.MAX_INT32)
        return type_random_seed(value)

    def suppress(self, to_suppress=None, **kwargs):
        """
        Supress the help messages of specified arguments.

        :param to_suppress: the arguments to be suppressed. For dict, the keys
            are the arguments to be suppressed, and the values are the default
            values to be used.
        :type to_suppress list, tuple, or set
        """
        flags = set() if to_suppress is None else set(to_suppress)
        self.set_defaults(**{x: y for x, y in kwargs.items()})
        for axn in self._actions:
            if axn.dest in kwargs or flags.intersection(axn.option_strings):
                axn.help = argparse.SUPPRESS
                continue

    def parse_args(self, args=None, **kwargs):
        """
        Parse the command line arguments and perform the validations.

        :param args list: command line arguments.
        :rtype 'argparse.Namespace': the parsed arguments.
        """
        options = super().parse_args(args=args, **kwargs)
        for Validator in self.validators:
            val = Validator(options)
            try:
                val.run()
            except ValueError as err:
                self.error(err)
        return options


class Bldr(Driver):
    """
    Parser with builder arguments.
    """
    FlAG_FORCE_FIELD = '-force_field'
    FLAG_SUBSTRUCT = '-substruct'

    @classmethod
    def addBldr(cls, parser, append=True):
        """
        Set up the builder arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param append `bool`: whether this appends to a previous parser.
        """
        parser.add_argument(jobutils.FLAG_SEED,
                            metavar=jobutils.FLAG_SEED[1:].upper(),
                            type=type_random_seed,
                            default=cls.getSeed(),
                            help='Set random state.')
        parser.add_argument(cls.FLAG_SUBSTRUCT,
                            metavar='SMILES (VALUE)',
                            nargs='+',
                            action=StructAction,
                            help='set or measure the substructure geometry.')
        parser.add_argument(
            cls.FlAG_FORCE_FIELD,
            metavar=cls.FlAG_FORCE_FIELD[1:].upper(),
            action=ForceFieldAction,
            nargs='+',
            default=symbols.OPLSUA_TIP3P,
            help=f'The force field type:\n'
            f'1) {symbols.OPLSUA} [{symbols.PIPE.join(symbols.WMODELS)}] '
            f'2) {symbols.SW}')


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
                            type=type_cru_smiles,
                            nargs='+',
                            help='SMILES of the constitutional repeat units.')
        parser.add_argument(
            cls.FLAG_CRU_NUM,
            metavar=cls.FLAG_CRU_NUM[1:].upper(),
            type=type_positive_int,
            nargs='+',
            help='Number of constitutional repeat unit per polymer')
        parser.add_argument(cls.FLAG_MOL_NUM,
                            metavar=cls.FLAG_MOL_NUM[1:].upper(),
                            type=type_positive_int,
                            nargs='+',
                            help='Number of molecules in the amorphous cell')
        # The buffer distance between molecules in the grid cell
        parser.add_argument(cls.FLAG_BUFFER,
                            metavar=cls.FLAG_BUFFER[1:].upper(),
                            type=type_positive_float,
                            help=argparse.SUPPRESS)
        parser.validators.add(MolValidator)
        cls.addBldr(parser)
        super().add(parser, **kwargs)
        Md.add(parser, append=True)


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
    FLAG_NO_MINIMIZE = '-no_minimize'
    FLAG_RIGID_BOND = '-rigid_bond'
    FLAG_RIGID_ANGLE = '-rigid_angle'

    @classmethod
    def add(cls, parser, append=True):
        """
        Set up the molecular dynamics arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param append `bool`: whether this appends to a previous parser.
        """
        parser.add_argument(cls.FLAG_TIMESTEP,
                            metavar='fs',
                            type=type_positive_float,
                            default=1,
                            help=f'Timestep for the MD simulation.')
        # 'Initialize the atoms with this temperature.'
        parser.add_argument(cls.FLAG_STEMP,
                            metavar='K',
                            type=type_positive_float,
                            default=10,
                            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_TEMP,
                            metavar=cls.FLAG_TEMP[1:].upper(),
                            type=type_nonnegative_float,
                            default=300,
                            help=f'The equilibrium temperature target. A zero '
                            f'for single point energy.')
        parser.add_argument(
            cls.FLAG_TDAMP,
            metavar=cls.FLAG_TDAMP[1:].upper(),
            type=type_positive_float,
            default=100,
            # Temperature damping parameter (x timestep to get the param)
            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_PRESS,
                            metavar=cls.FLAG_PRESS[1:].upper(),
                            type=float,
                            default=1,
                            help="The equilibrium pressure target.")
        # Pressure damping parameter (x timestep to get the param)
        parser.add_argument(cls.FLAG_PDAMP,
                            metavar=cls.FLAG_PDAMP[1:].upper(),
                            type=type_positive_float,
                            default=1000,
                            help=argparse.SUPPRESS)
        parser.add_argument(cls.FLAG_RELAX_TIME,
                            metavar='ns',
                            type=type_nonnegative_float,
                            default=1,
                            help='Relaxation simulation time.')
        parser.add_argument(cls.FLAG_PROD_TIME,
                            metavar='ns',
                            type=type_positive_float,
                            default=1,
                            help='Production simulation time.')
        parser.add_argument(cls.FLAG_PROD_ENS,
                            choices=lammpsfix.ENSEMBLES,
                            default=lammpsfix.NVE,
                            help='Production ensemble.')
        if not append:
            parser.add_argument(jobutils.FLAG_SEED,
                                metavar=jobutils.FLAG_SEED[1:].upper(),
                                type=type_random_seed,
                                default=cls.getSeed(),
                                help='Set random state.')
        # Skip the structure minimization step
        parser.add_argument(cls.FLAG_NO_MINIMIZE,
                            action='store_true',
                            help=argparse.SUPPRESS)
        # The lengths of these types are fixed during the simulation
        parser.add_argument(cls.FLAG_RIGID_BOND,
                            metavar=cls.FLAG_RIGID_BOND[1:].upper(),
                            type=type_positive_int,
                            nargs='+',
                            help=argparse.SUPPRESS)
        # The angles of these types are fixed during the simulation
        parser.add_argument(cls.FLAG_RIGID_ANGLE,
                            metavar=cls.FLAG_RIGID_ANGLE[1:].upper(),
                            type=type_positive_int,
                            nargs='+',
                            help=argparse.SUPPRESS)


class MolBldr(MolBase):
    """
    Parser with molecule arguments.
    """

    @classmethod
    def add(cls, parser, **kwargs):
        """
        Set up the single-molecule arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param seed `bool`: whether this is a seeding job in the workflow
        """
        super().add(parser, **kwargs)
        parser.suppress(buffer=f"{symbols.DEFAULT_CUT * 4}",
                        mol_num=[1],
                        temp=0,
                        timestep=1,
                        press=1,
                        relax_time=0,
                        prod_time=0,
                        prod_ens=lammpsfix.NVE)


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
        :param seed `bool`: whether this is a seeding job in the workflow
        """
        parser.add_argument(
            cls.FLAG_DENSITY,
            metavar='g/cm^3',
            type=functools.partial(type_ranged_float,
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
        super().addBldr(parser, **kwargs)


class XtalBldr(Bldr):
    """
    Parser with crystal arguments.
    """
    FlAG_NAME = jobutils.FlAG_NAME
    FlAG_DIMENSION = '-dimension'
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
            cls.FlAG_NAME,
            default='Si',
            choices=['Si'],
            help='Name to retrieve the crystal structure from the database.')
        parser.add_argument(
            cls.FlAG_DIMENSION,
            default=cls.ONES,
            nargs='+',
            metavar=cls.FlAG_DIMENSION[1:].upper(),
            type=int,
            action=ThreeAction,
            help='Duplicate the unit cell by these factors to generate the '
            'supercell.')
        parser.add_argument(
            cls.FLAG_SCALED_FACTOR,
            default=cls.ONES,
            nargs='+',
            metavar=cls.FLAG_SCALED_FACTOR[1:].upper(),
            type=type_positive_float,
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
    FLAG_LOG = '-log'
    FLAG_DATA_FILE = '-data_file'

    @classmethod
    def add(cls, parser, append=True):
        """
        Set up the molecular dynamics arguments.

        :param parser `argparse.ArgumentParser`: the parser instance to set up
        :param append `bool`: whether this appends to a previous parser.
        """
        parser.add_argument(cls.FLAG_INSCRIPT,
                            metavar=cls.FLAG_INSCRIPT.upper(),
                            type=type_file,
                            help='Read input from this file.')
        parser.add_argument(cls.FLAG_DATA_FILE,
                            metavar=cls.FLAG_DATA_FILE[1:].upper(),
                            type=type_file,
                            help='Data file to get force field information')
        parser.add_argument(jobutils.FLAG_LOG,
                            metavar=jobutils.FLAG_LOG[1:].upper(),
                            help='Print logging information into this file.')
        parser.add_argument(jobutils.FLAG_SCREEN,
                            default=symbols.NONE,
                            help='Where to send screen output.')
        parser.validators.add(LmpValidator)


class Log(Driver):
    """
    Parser with lammps-log arguments.
    """
    FLAG_DATA_FILE = '-data_file'
    FLAG_LAST_PCT = '-last_pct'
    FLAG_SLICE = '-slice'
    TASKS = [*analyzer.THERMO.keys(), symbols.ALL]
    TASK_HELP = 'Searches, combines and averages thermodynamic info.'
    LAST_FRM = analyzer.THERMO.keys()

    @classmethod
    def add(cls, parser, append=True, task=symbols.TOTENG):
        """
        Add job specific arguments to the parser.

        :param parser argparse.ArgumentParser: the parse to add arguments
        :param append `bool`: whether this appends to a previous parser.
        :param task str: the default task
        """
        if not append:
            parser.add_argument(cls.__name__.lower(),
                                metavar=cls.__name__.upper(),
                                type=type_file,
                                help=f'The {cls.__name__} file to analyze.')
        parser.add_argument(jobutils.FLAG_TASK,
                            type=str.lower,
                            default=[task],
                            choices=cls.TASKS,
                            nargs='+',
                            help=cls.TASK_HELP)
        if not append:
            parser.add_argument(
                cls.FLAG_DATA_FILE,
                metavar=cls.FLAG_DATA_FILE[1:].upper(),
                type=type_file,
                help='The file of the structure and force field.')
        parser.add_argument(
            cls.FLAG_LAST_PCT,
            type=LastPct.type,
            default=LastPct(0.2),
            help=f"{', '.join(cls.LAST_FRM)} average results from this last "
            "percentage to the end.")
        parser.add_argument(
            cls.FLAG_SLICE,
            metavar='START END STEP',
            type=type_nonnegative_int,
            action=SliceAction,
            nargs='+',
            help="SliceAction the input data before the analysis by "
            "END, START END, or START END STEP.")


class Traj(Log):
    """
    Parser with lammps-trajectory arguments.
    """
    TASKS = analyzer.TRAJ.keys()
    TASK_HELP = ', '.join(x.__doc__.strip().lower()
                          for x in analyzer.TRAJ.values())
    LAST_FRM = [x.NAME for x in [analyzer.MSD, analyzer.RDF]]

    @classmethod
    def add(cls, parser, append=True, task=analyzer.Density.NAME):
        """
        Add job specific arguments to the parser.

        :param parser argparse.ArgumentParser: the parse to add arguments
        :param append `bool`: whether this appends to a previous parser.
        :param task str: the default task
        """
        super().add(parser, append=append, task=task)
        parser.add_argument('-sel', help=f'The element of the selected atoms.')
        if not append:
            parser.validators.add(TrajValidator)


class Workflow(Driver):
    """
    Parser with workflow arguments.
    """
    FLAG_STATE_NUM = '-state_num'
    FLAG_CLEAN = '-clean'
    FLAG_JTYPE = '-jtype'
    FLAG_PRJ_PATH = '-prj_path'
    FLAG_SCREEN = jobutils.FLAG_SCREEN
    WFLAGS = [FLAG_STATE_NUM, FLAG_CLEAN, FLAG_JTYPE, FLAG_SCREEN]
    SCREENS = [jobutils.PROGRESS, jobutils.JOB, symbols.OFF]

    def __init__(self, *args, **kwargs):
        """
        Set up the parser.
        """
        super().__init__(*args, **kwargs)
        if self.delay:
            return
        self.addWorkflow()

    def addWorkflow(self):
        """
        Add workflow related flags.
        """
        if self.FLAG_STATE_NUM in self.WFLAGS:
            self.add_argument(
                self.FLAG_STATE_NUM,
                default=1,
                metavar=self.FLAG_STATE_NUM[1:].upper(),
                type=type_positive_int,
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
            self.add_argument(
                self.FLAG_PRJ_PATH,
                type=type_dir,
                help='The aggregator jobs collect jobs from this directory.')
        if self.FLAG_CLEAN in self.WFLAGS:
            self.add_argument(self.FLAG_CLEAN,
                              action='store_true',
                              help='Clean previous workflow results.')
        if self.FLAG_SCREEN in self.WFLAGS:
            os.environ['TQDM_DISABLE'] = '1'
            self.add_argument(
                self.FLAG_SCREEN,
                nargs='+',
                choices=self.SCREENS,
                type=type_screen,
                help=f'{jobutils.PROGRESS}: serialization and parallelization; '
                f'{jobutils.JOB}: cmd process details.; {symbols.OFF}: no '
                f'printing')
