# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
parser utilities.
"""
import argparse
import functools
import os
import random

from nemd import cru
from nemd import envutils
from nemd import jobutils
from nemd import lammpsfix
from nemd import np
from nemd import rdkitutils
from nemd import sw
from nemd import symbols

FLAG_STATE_NUM = '-state_num'
FLAG_CLEAN = '-clean'
FLAG_JTYPE = '-jtype'
FLAG_CPU = jobutils.FLAG_CPU
FLAG_INTERACTIVE = jobutils.FLAG_INTERACTIVE
FLAG_JOBNAME = jobutils.FLAG_JOBNAME
FLAG_DEBUG = jobutils.FLAG_DEBUG
FLAG_PYTHON = jobutils.FLAG_PYTHON
FLAG_PRJ_PATH = '-prj_path'
FLAG_SCREEN = jobutils.FLAG_SCREEN

FLAG_SUBSTRUCT = '-substruct'
FLAG_NO_MINIMIZE = '-no_minimize'
FLAG_RIGID_BOND = '-rigid_bond'
FLAG_RIGID_ANGLE = '-rigid_angle'
BLDR_FLAGS = [
    FLAG_SUBSTRUCT, FLAG_NO_MINIMIZE, FLAG_RIGID_BOND, FLAG_RIGID_ANGLE
]

FLAG_CRU = 'cru'
FLAG_CRU_NUM = '-cru_num'
FLAG_MOL_NUM = '-mol_num'
FLAG_BUFFER = '-buffer'
FLAG_SEED = jobutils.FLAG_SEED

FlAG_NAME = '-name'
FlAG_DIMENSION = '-dimension'
FLAG_SCALED_FACTOR = '-scale_factor'

FLAG_TIMESTEP = '-timestep'
FLAG_STEMP = '-stemp'
FLAG_TEMP = '-temp'
FLAG_TDAMP = '-tdamp'
FLAG_PRESS = '-press'
FLAG_PDAMP = '-pdamp'
FLAG_RELAX_TIME = '-relax_time'
FLAG_PROD_TIME = '-prod_time'
FLAG_PROD_ENS = '-prod_ens'
FlAG_FORCE_FIELD = '-force_field'

INTEGRATION = 'integration'
SCIENTIFIC = 'scientific'
PERFORMANCE = 'performance'

FLAG_ID = 'id'
FLAG_DIR = jobutils.FLAG_DIR
FLAG_TASK = '-task'
FLAG_LABEL = '-label'
CMD = 'cmd'
CHECK = 'check'
TAG = 'tag'

ONES = (1, 1, 1)


def type_file(arg):
    """
    Check whether the argument is an existing file.

    :param arg str: the input argument.
    :return str: the existing namepath of the file.
    :raise ArgumentTypeError: if the file is not found.
    """
    if os.path.isfile(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} not found.')


def type_dir(arg):
    """
    Check whether the argument is an existing directory.

    :param arg str: the input argument.
    :return str: the existing directory path.
    :raise ArgumentTypeError: if the directory doesn't exist.
    """
    if os.path.isdir(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} is not an existing directory.')


def type_bool(arg):
    """
    Check whether the argument can be converted to a boolean value.

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
    Check whether the argument can be converted to a float.

    :param arg str: the input argument.
    :return `float`: the converted float value.
    :raise ArgumentTypeError: argument cannot be converted to a float.
    """
    try:
        return float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Cannot convert {arg} to a float.')


def type_ranged(value,
                bottom=-symbols.MAX_INT32,
                top=symbols.MAX_INT32,
                included_bottom=True,
                include_top=True):
    """
    Check whether the value is within the range.

    :param value `float` or `int`: the value to be checked.
    :param bottom `float` or `int`: the lower bound of the range.
    :param top `float` or `int`: the upper bound of the range.
    :param included_bottom bool: whether the lower bound is allowed
    :param include_top bool: whether the upper bound is allowed
    :return `float` or `int`: the checked value.
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
    Check whether the argument can be converted to a float within the range.

    :param arg str: the input argument.
    :return `float`: the converted value within the range.
    """
    value = type_float(arg)
    return type_ranged(value, **kwargs)


def type_nonnegative_float(arg):
    """
    Check whether the argument can be converted to a non-negative float.

    :param arg str: the input argument.
    :return `float`: the converted non-negative value.
    """
    return type_ranged_float(arg, bottom=0)


def type_positive_float(arg):
    """
    Check whether the argument can be converted to a positive float.

    :param arg str: the input argument.
    :return `float`: the converted positive value.
    """
    return type_ranged_float(arg, bottom=0, included_bottom=False)


def type_int(arg):
    """
    Check whether the argument can be converted to an integer.

    :param arg str: the input argument.
    :return `int:: the converted integer.
    :raise ArgumentTypeError: argument cannot be converted to an integer.
    """
    try:
        return int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Cannot convert {arg} to an integer')


def type_positive_int(arg):
    """
    Check whether the argument can be converted to a positive integer.

    :param arg str: the input argument.
    :return `int: the converted positive integer.
    """
    value = type_int(arg)
    type_ranged(value, bottom=1)
    return value


def type_random_seed(arg):
    """
    Check whether the argument can be converted to a random seed.

    :param arg str: the input argument.
    :return `int: the converted random seed.
    """
    value = type_int(arg)
    return type_ranged(value, bottom=0, top=symbols.MAX_INT32)


def type_smiles(arg):
    """
    Check whether the argument is a smiles that can be converted to molecule.

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
    Check whether the argument is a SMILES that can be converted to
    constitutional repeating unit.

    :param allow_reg bool: whether to allow regular molecule (without wildcard).
    :param canonize bool: whether to canonize the SMILES.
    :return `rdkit.Chem.rdchem.Mol: the converted molecule.
    :raise ArgumentTypeError: 1) regular molecule(s) found but allow_reg=False;
        2) constitutional repeating units and regular molecules are mixed;
        3) constitutional repeating units are insufficient to build a polymer.
    """
    mol = cru.Mol(type_smiles(arg), allow_reg=allow_reg)
    try:
        mol.run()
    except cru.MoietyError as err:
        raise argparse.ArgumentTypeError(str(err))
    return mol.getSmiles(canonize=canonize)


class Action(argparse.Action):
    """
    Argparse action that errors on ArgumentTypeError in doTyping().
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        The action to be taken when the argument is parsed.

        :param parser `ArgumentParser`: the parser object.
        :param 'argparse.Namespace': partially parsed arguments.
        :param values list: the values to be parsed.
        :param option_string str: the option string (flog)
        """
        try:
            setattr(namespace, self.dest, self.doTyping(*values))
        except argparse.ArgumentTypeError as err:
            parser.error(f"{err} ({option_string})")


class ForceField(Action):
    """
    Argparse action that allows a mandatory force field name fallowed by an
    optional water model name.
    """

    def doTyping(self, name, *args):
        """
        Check the validity of the force field and water model.

        :param name str: the force field name
        :return list: force field name, additional arguments
        """
        choices = None
        match name:
            case symbols.SW:
                if args and not sw.get_file(args):
                    choices = f"elements set from {sw.NAME_ELEMENTS} sub-lists"
            case symbols.OPLSUA:
                if not args:
                    args = [symbols.TIP3P]
                if args[0] not in symbols.WMODELS:
                    choices = f"water models from {symbols.WMODELS}"
            case _:
                choices = f"force field from {symbols.FF_NAMES}"
        if choices:
            raise argparse.ArgumentTypeError(f"Please choose {choices}.")
        return name, *args


class Slice(Action):
    """
    Argparse action that allows a mandatory END, and optional START and STEP
    to slice a range of values.
    """

    def doTyping(self, *args):
        """
        Check the validity of the slice arguments.

        :param args list of str: the arguments for the slice function.
            (1: END; 2: START, END; 3: START, END, STEP)
        :return list of int: start, stop, and step
        """
        args = [None if x == 'None' else type_int(x) for x in args]
        sliced = slice(*args)
        start = 0 if sliced.start is None else sliced.start
        stop = sliced.stop
        step = 1 if sliced.step is None else sliced.step
        if start < 0 or (stop is not None and stop < 0) or step <= 0:
            raise argparse.ArgumentTypeError(f"{args} invalid for islice.")
        return start, stop, step


class Struct(Action):
    """
    Action for argparse that allows a mandatory smile str followed by an optional
    VALUE of float type.
    """

    def doTyping(self, smiles, value=None):
        """
        Check the validity of the slice arguments.

        :param smiles str: the smiles str to select a substructure.
        :param value str: the target value for the substructure to be set.
        :return str, float: the smiles str, and the target value.
        """
        type_smiles(smiles)
        if value is not None:
            value = type_float(value)
        return smiles, value


class StructRg(Struct):
    """
    Action for argparse that allows a mandatory smile str followed by optional
    START END, and STEP values of type float, float, and int, respectively.
    """

    def doTyping(self, smiles, start=None, end=None, step=None):
        """
        Check the validity of the smiles string and the range.

        :param smiles str: the smiles str to select a substructure.
        :param start str: the start to define a range.
        :param end str: the end to define a range.
        :param step str: the step to define a range.
        :return str, float, float, float: the smiles str, start, end, and step.
        """
        _, start = super().doTyping(smiles, start)
        if start is None:
            return [smiles, None]
        if end is None or step is None:
            raise argparse.ArgumentTypeError(
                "start, end, and step partially provided.")
        return [smiles, start, type_float(end), type_float(step)]


class Three(Action):
    """
    Argparse action that allow three values.
    """

    def doTyping(self, *args):
        """
        Check the validity of the smiles string and the range.

        :param smiles str: the smiles str to select a substructure.
        :param start str: the start to define a range.
        :param end str: the end to define a range.
        :param step str: the step to define a range.
        :return str, float, float, float: the smiles str, start, end, and step.
        """
        if len(args) == 1:
            return args * 3
        if len(args) == 2:
            raise ValueError(f"{self.option_strings[0]} expects three values.")
        return args[:3]


class Validator:
    """
    Class to validate arguments after parse_args().
    """

    def __init__(self, options):
        """
        param options 'argparse.Namespace': Command line options.
        """
        self.options = options

    def run(self):
        """
        Set the random seed.
        """
        if self.options.seed is None:
            self.options.seed = np.random.randint(0, symbols.MAX_INT32)
        np.random.seed(self.options.seed)
        random.seed(self.options.seed)


class WorkflowValidator(Validator):
    """
    Class to validate workflow related arguments after parse_args().
    """

    def __init__(self, options):
        """
        param options 'argparse.Namespace': Command line options.
        """
        self.options = options

    def run(self):
        """
        Main method to run the validation.
        """
        os.environ['TQDM_DISABLE'] = '' if self.options.screen else '1'


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


class TestValidator(Validator):
    """
    Class to validate the input options.
    """

    def run(self):
        """
        Main method to run the validation.

        :raises ValueError: if the input directory is None.
        """
        if self.options.dir is None:
            self.options.dir = envutils.get_nemd_src('test', self.options.name)
        if not self.options.dir:
            raise ValueError(f'Please define the test dir via {FLAG_DIR}')


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


class ArgumentParser(argparse.ArgumentParser):
    """
    A customized parser that provides additional features.
    """
    JFLAGS = [
        FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_PYTHON, FLAG_CPU, FLAG_DEBUG
    ]

    def __init__(self,
                 file='name_driver.py',
                 formatter_class=ArgumentDefaultsHelpFormatter,
                 descr=None,
                 validators=None,
                 **kwargs):
        """
        :param file str: script filename which defines the default jobname.
        :param formatter_class 'ArgumentDefaultsHelpFormatter': the formatter
            class to display the customized help message
        :param descr str: the script description displayed as the help message.
        :param validators set: across-argument validators after parse_args()
        """
        if descr is not None:
            kwargs.update(description=descr)
        super().__init__(formatter_class=formatter_class, **kwargs)
        self.file = file
        self.validators = set() if validators is None else validators
        self.setUp()

    def setUp(self):
        """
        Set up the parser.
        """
        pass

    def parse_args(self, argv):
        """
        Parse the command line arguments and perform the validations.

        :param argv list: command line arguments.
        :rtype 'argparse.Namespace': the parsed arguments.
        """
        options = super().parse_args(argv)
        for Validator in self.validators:
            val = Validator(options)
            try:
                val.run()
            except ValueError as err:
                self.error(err)
        return options

    @property
    @functools.cache
    def name(self):
        """
        Return the default jobname.

        :return str: the default jobname.
        """
        return jobutils.get_name(self.file)

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

    def add_bldr_arguments(self):
        """
        Add builder flags.
        """
        self.add_argument(
            FLAG_BUFFER,
            metavar=FLAG_BUFFER[1:].upper(),
            type=type_positive_float,
            help='The buffer distance between molecules in the grid cell.')
        self.add_argument(FLAG_NO_MINIMIZE,
                          action='store_true',
                          help='Skip the structure minimization step.')
        self.add_argument(
            FLAG_RIGID_BOND,
            metavar=FLAG_RIGID_BOND[1:].upper(),
            type=type_positive_int,
            nargs='+',
            help='The lengths of these types are fixed during the simulation.')
        self.add_argument(
            FLAG_RIGID_ANGLE,
            metavar=FLAG_RIGID_ANGLE[1:].upper(),
            type=type_positive_int,
            nargs='+',
            help='The angles of these types are fixed during the simulation.')
        self.add_argument(FLAG_SUBSTRUCT,
                          metavar='SMILES (VALUE)',
                          nargs='+',
                          action=Struct,
                          help='set or measure the substructure geometry.')
        self.add_argument(
            FlAG_FORCE_FIELD,
            metavar=FlAG_FORCE_FIELD[1:].upper(),
            action=ForceField,
            nargs='+',
            default=symbols.OPLSUA_TIP3P,
            help=f'The force field type:\n'
            f'1) {symbols.OPLSUA} [{symbols.PIPE.join(symbols.WMODELS)}] '
            f'2) {symbols.SW}')
        self.suppress([FLAG_BUFFER, FLAG_RIGID_BOND, FLAG_RIGID_ANGLE])

    def add_polym_arguments(self):
        """
        Add polymer builder flags.
        """
        self.add_argument(FLAG_CRU,
                          metavar=FLAG_CRU.upper(),
                          type=type_cru_smiles,
                          nargs='+',
                          help='SMILES of the constitutional repeat units.')
        self.add_argument(
            FLAG_CRU_NUM,
            metavar=FLAG_CRU_NUM[1:].upper(),
            type=type_positive_int,
            nargs='+',
            help='Number of constitutional repeat unit per polymer')
        self.add_argument(FLAG_MOL_NUM,
                          metavar=FLAG_MOL_NUM[1:].upper(),
                          type=type_positive_int,
                          nargs='+',
                          help='Number of molecules in the amorphous cell')
        self.validators.add(MolValidator)

    def add_xtal_arguments(self):
        """
        Add crystal builder flags.
        """
        # FIXME: support more choices based on crystals.Crystal.builtins
        self.add_argument(
            FlAG_NAME,
            default='Si',
            choices=['Si'],
            help='Name to retrieve the crystal structure from the database.')
        self.add_argument(
            FlAG_DIMENSION,
            default=ONES,
            nargs='+',
            metavar=FlAG_DIMENSION.upper()[1:],
            type=int,
            action=Three,
            help='Duplicate the unit cell by these factors to generate the '
            'supercell.')
        self.add_argument(
            FLAG_SCALED_FACTOR,
            default=ONES,
            nargs='+',
            metavar=FLAG_SCALED_FACTOR.upper()[1:],
            type=type_positive_float,
            action=Three,
            help='Each lattice vector is scaled by the corresponding factor.')

    def add_md_arguments(self):
        """
        The molecular dynamics flags.
        """
        self.add_argument(FLAG_SEED,
                          metavar=FLAG_SEED[1:].upper(),
                          type=type_random_seed,
                          help='Set random state.')
        self.add_argument(FLAG_TIMESTEP,
                          metavar='fs',
                          type=type_positive_float,
                          default=1,
                          help=f'Timestep for the MD simulation.')
        self.add_argument(
            FLAG_STEMP,
            metavar='K',
            type=type_positive_float,
            default=10,
            # 'Initialize the atoms with this temperature.'
            help=argparse.SUPPRESS)
        self.add_argument(FLAG_TEMP,
                          metavar=FLAG_TEMP[1:].upper(),
                          type=type_nonnegative_float,
                          default=300,
                          help=f'The equilibrium temperature target. A zero '
                          f'for single point energy.')
        self.add_argument(
            FLAG_TDAMP,
            metavar=FLAG_TDAMP[1:].upper(),
            type=type_positive_float,
            default=100,
            # Temperature damping parameter (x timestep to get the param)
            help=argparse.SUPPRESS)
        self.add_argument(FLAG_PRESS,
                          metavar=FLAG_PRESS[1:].upper(),
                          type=float,
                          default=1,
                          help="The equilibrium pressure target.")
        self.add_argument(
            FLAG_PDAMP,
            metavar=FLAG_PDAMP[1:].upper(),
            type=type_positive_float,
            default=1000,
            # Pressure damping parameter (x timestep to get the param)
            help=argparse.SUPPRESS)
        self.add_argument(FLAG_RELAX_TIME,
                          metavar='ns',
                          type=type_nonnegative_float,
                          default=1,
                          help='Relaxation simulation time.')
        self.add_argument(FLAG_PROD_TIME,
                          metavar='ns',
                          type=type_positive_float,
                          default=1,
                          help='Production simulation time.')
        self.add_argument(FLAG_PROD_ENS,
                          choices=lammpsfix.ENSEMBLES,
                          default=lammpsfix.NVE,
                          help='Production ensemble.')
        self.validators.add(Validator)

    def add_test_arguments(self):
        """
        Add test related flags.
        """
        self.add_argument(
            FLAG_ID,
            metavar=FLAG_ID.upper(),
            type=type_positive_int,
            nargs='*',
            help='Select the tests according to these ids.')
        self.add_argument(FlAG_NAME,
                          default=INTEGRATION,
                          choices=[INTEGRATION, SCIENTIFIC, PERFORMANCE],
                          help=f'{INTEGRATION}: reproducible; '
                          f'{SCIENTIFIC}: physical meaningful; '
                          f'{PERFORMANCE}: resource efficient.')
        self.add_argument(FLAG_DIR,
                          metavar=FLAG_DIR[1:].upper(),
                          type=type_dir,
                          help='Search test(s) under this directory.')
        self.add_argument(
            jobutils.FLAG_SLOW,
            type=type_positive_float,
            metavar='SECOND',
            help='Skip (sub)tests marked with time longer than this criteria.')
        self.add_argument(FLAG_LABEL,
                          nargs='+',
                          metavar='LABEL',
                          help='Select the tests marked with the given label.')
        self.add_argument(FLAG_TASK,
                          nargs='+',
                          choices=[CMD, CHECK, TAG],
                          default=[CMD, CHECK],
                          help='cmd: run the commands in cmd file; '
                          'check: check the results; tag: update the tag file')
        self.validators.add(TestValidator)

    def add_job_arguments(self):
        """
        Add job control related flags.
        """
        if FLAG_JOBNAME in self.JFLAGS:
            self.add_argument(jobutils.FLAG_NAME,
                              default=self.name,
                              help=argparse.SUPPRESS)
            envutils.set_default_jobname(self.name)
            self.add_argument(FLAG_JOBNAME,
                              dest=FLAG_JOBNAME[1:].lower(),
                              default=self.name,
                              help='Name output files.')
        if FLAG_INTERACTIVE in self.JFLAGS:
            self.add_argument(FLAG_INTERACTIVE,
                              dest=FLAG_INTERACTIVE[1:].lower(),
                              action='store_true',
                              help='Pause for user input.')
        if FLAG_PYTHON in self.JFLAGS:
            self.add_argument(FLAG_PYTHON,
                              default=envutils.CACHE_MODE,
                              dest=FLAG_PYTHON[1:].lower(),
                              choices=envutils.PYTHON_MODES,
                              help='0: native; 1: compiled; 2: cached.')
        if FLAG_CPU in self.JFLAGS:
            self.add_argument(
                FLAG_CPU,
                type=type_positive_int,
                nargs='+',
                dest=FLAG_CPU[1:].lower(),
                help='Total number of CPUs (the number for one task).')
        if FLAG_DEBUG in self.JFLAGS:
            self.add_argument(FLAG_DEBUG,
                              action='store_true',
                              dest=FLAG_DEBUG[1:].lower(),
                              help='Additional printing and output files.')


class WorkflowParser(ArgumentParser):
    """
    A customized parser that provides additional features.
    """
    JFLAGS = ArgumentParser.JFLAGS[:-1]
    WFLAGS = [FLAG_STATE_NUM, FLAG_CLEAN, FLAG_JTYPE, FLAG_SCREEN, FLAG_DEBUG]

    def add_workflow_arguments(self):
        """
        Add workflow related flags.
        """
        if FLAG_STATE_NUM in self.WFLAGS:
            self.add_argument(
                FLAG_STATE_NUM,
                default=1,
                metavar=FLAG_STATE_NUM[1:].upper(),
                type=type_positive_int,
                help='Total number of the states (e.g., dynamical system).')
        if FLAG_JTYPE in self.WFLAGS:
            # Task jobs have to register outfiles to be considered as completed
            # Aggregator jobs collect results from finished task jobs
            self.add_argument(
                FLAG_JTYPE,
                nargs='+',
                choices=[jobutils.TASK, jobutils.AGGREGATOR],
                default=[jobutils.TASK, jobutils.AGGREGATOR],
                help=f'{jobutils.TASK}: run tasks and register files; '
                f'{jobutils.AGGREGATOR}: collect results.')
            self.add_argument(
                FLAG_PRJ_PATH,
                type=type_dir,
                help='The aggregator jobs collect jobs from this directory.')
        if FLAG_CLEAN in self.WFLAGS:
            self.add_argument(FLAG_CLEAN,
                              action='store_true',
                              help='Clean previous workflow results.')
        if FLAG_SCREEN in self.WFLAGS:
            self.add_argument(
                FLAG_SCREEN,
                nargs='+',
                choices=[
                    jobutils.TQDM, jobutils.PROGRESS, jobutils.JOB, symbols.OFF
                ],
                help=f'Print the serialization {jobutils.TQDM}, parallelization'
                f' {jobutils.PROGRESS}, and {jobutils.JOB} details.')
            self.validators.add(WorkflowValidator)
        if FLAG_DEBUG in self.WFLAGS:
            self.add_argument(
                FLAG_DEBUG,
                nargs='?',
                const=True,
                type=type_bool,
                choices=[symbols.ON, symbols.OFF],
                dest=FLAG_DEBUG[1:].lower(),
                help=f'{symbols.ON}: enable debug mode for the workflow and '
                f'sub-jobs; {symbols.OFF}: disable the mode.')
