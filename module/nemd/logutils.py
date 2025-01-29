# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class creates loggers for modules and scripts, redirects stdout and stderr,
logs messages and options, and parses log files.
"""
import contextlib
import io
import logging
import os
import re
import sys
import types

import wurlitzer

from nemd import IS_DEBUG
from nemd import envutils
from nemd import jobutils
from nemd import psutils
from nemd import symbols
from nemd import timeutils

STATUS_LOG = f'_status{symbols.LOG}'
JOBSTART = 'JobStart:'
FINISHED = 'Finished.'
START = 'start'
END = 'end'
DELTA = 'delta'
COMMAND_OPTIONS = 'Command Options'
COMMAND_OPTIONS_START = f"." * 10 + COMMAND_OPTIONS + f"." * 10
COMMAND_OPTIONS_END = f"." * (20 + len(COMMAND_OPTIONS))
COLON_SEP = f'{symbols.COLON} '
MEMORY = 'memory'
MEMORY_UNIT = 'MB'
MEMORY = f'Peak {MEMORY} usage: %s {MEMORY_UNIT}'


class FileHandler(logging.FileHandler):
    """
    Handler that controls the writing of the newline character.

    https://stackoverflow.com/questions/7168790/suppress-newline-in-python-logging-module
    """

    NO_NEWLINE = '[!n]'
    INFO_FMT = '%(message)s'
    DEBUG_FMT = f'%(asctime)s %(levelname)s {INFO_FMT}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fmt = logging.Formatter(self.DEBUG_FMT if IS_DEBUG else self.INFO_FMT)
        self.setFormatter(fmt)

    def emit(self, record):
        """
        See parent method for documentation.
        """
        newline = not record.msg.endswith(self.NO_NEWLINE)
        pre_newline = self.terminator == '\n'
        self.terminator = '\n' if newline else ''
        record.msg = record.msg.replace(self.NO_NEWLINE, '')
        if not pre_newline:
            record.msg = self.NO_NEWLINE + record.msg
        return super().emit(record)

    def format(self, record):
        """
        See parent method for documentation.
        """
        default = not self.formatter or record.msg.startswith(self.NO_NEWLINE)
        fmt = logging._defaultFormatter if default else self.formatter
        record.msg = record.msg.replace(self.NO_NEWLINE, '')
        return fmt.format(record)


class Logger(logging.Logger):
    """
    A logger for driver so that customer-facing information can be saved.
    """

    def setLevel(self, level=None):
        """
        Set the level of the logger.

        :param level int: the level of the logger
        """
        if level is None:
            level = logging.DEBUG if IS_DEBUG else logging.INFO
        super().setLevel(level)

    def infoJob(self, options):
        """
        Info the job options and the start time.

        :param options 'argparse.Namespace': command-line options
        """
        self.info(COMMAND_OPTIONS_START)
        for key, val in options.__dict__.items():
            if type(val) is list:
                val = symbols.SPACE.join(map(str, val))
            self.info(f"{key}{COLON_SEP}{val}")
        self.info(f"{JOBSTART} {timeutils.ctime()}")
        self.info(COMMAND_OPTIONS_END)

    def log(self, msg, timestamp=False):
        """
        Log message to the logger.

        :param msg str: the message to be printed out
        :param timestamp bool: append time information after the message
        """
        self.info(msg)
        if timestamp:
            self.info(timeutils.ctime())

    def error(self, msg, timestamp=True):
        """
        Print this message and exit the program.

        :param msg str: the msg to be printed
        :param timestamp bool: append time information after the message
        """
        self.log(msg + '\nAborting...', timestamp=timestamp)
        sys.exit(1)

    @classmethod
    def get(cls, name):
        """
        Get the logger.

        :param name: the logger name
        :return 'logging.Logger': the logger
        """
        logger_class = logging.getLoggerClass()
        logging.setLoggerClass(cls)
        logger = logging.getLogger(name)
        logging.setLoggerClass(logger_class)
        return logger


def get_logger(pathname, log=False, file=False):
    """
    Get a module logger to print debug information.

    :param pathname str: pathname based on which logger name is generated
    :param log bool: sets as the log file if True
    :param file bool: set this file as the single output file
    :return 'logging.Logger': the logger
    """
    name = os.path.splitext(os.path.basename(pathname))[0]
    logger = Logger.get(name)
    logger.setLevel()
    if IS_DEBUG and not logger.hasHandlers():
        outfile = name + '.debug'
        logger.addHandler(FileHandler(outfile, mode='w'))
        jobutils.add_outfile(outfile, file=file, log=log)
    return logger


class Script:
    """
    A class to handle the logging in running a script.
    """

    MEMORY = MEMORY % '{value:.4f}'

    def __init__(self, options, log=True, file=False):
        """
        :param options str: the command-line options
        :param log bool: sets as the log file if True
        :param file bool: set this file as the single output file
        """
        self.options = options
        self.memory = None
        self.name = options.jobname
        self.logger = Logger.get(self.name)
        self.outfile = self.name + symbols.LOG
        jobutils.add_outfile(self.outfile, file=file, log=log)

    def __enter__(self):
        """
        Create the logger and start the memory monitoring if requested.

        :return `Logger`: the logger object to print messages
        """
        self.logger.setLevel()
        if not self.logger.hasHandlers():
            self.logger.addHandler(FileHandler(self.outfile, mode='w'))
        self.logger.infoJob(self.options)
        intvl = envutils.get_mem_intvl()
        if intvl is not None:
            self.memory = psutils.Memory(intvl)
            self.memory.thread.start()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the memory monitoring and print the messages.

        :param exc_type `type`: the exception type
        :param exc_val Exception: the exception
        :param exc_tb traceback: the traceback object
        :raise Exception: when in DEBUG mode
        """
        if self.memory:
            self.logger.log(f"{self.MEMORY.format(value=self.memory.result)}")
        if exc_type:
            while exc_tb.tb_next:
                exc_tb = exc_tb.tb_next
            if isinstance(exc_val, SystemExit):
                # log_error calls sys.exit(1)
                return
            self.logger.log(f"File {exc_tb.tb_frame.f_code.co_filename}, "
                            f"line {exc_tb.tb_frame.f_lineno}\n"
                            f"{exc_type.__name__}: {exc_val.args}")
            raise exc_val
        self.logger.log(FINISHED, timestamp=True)


class Base(object):
    """
    A base class with a logger to print logging messages.
    """

    def __init__(self, logger=None):
        """
        :param logger 'logging.Logger': the logger to log messages
        """
        self.logger = logger

    def log(self, msg, **kwargs):
        """
        Print this message into the log file as information.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.log(msg, **kwargs)
        else:
            print(msg)

    def log_debug(self, msg):
        """
        Print this message into the log file in debug mode.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log_warning(self, msg):
        """
        Print this warning message into log file.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.warning(msg)
        else:
            print(msg)

    def log_error(self, msg):
        """
        Print this message and exit the program.

        :param msg str: the msg to be printed
        """
        self.log(msg + '\nAborting...', timestamp=True)
        sys.exit(1)


class Handler(logging.Handler):
    """
    This class saves the records instead of printing them.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
        self.logs = {}

    def handle(self, record):
        key = record.levelname
        self.logs[key] = self.logs.get(key, "") + self.format(record)
        return False


@contextlib.contextmanager
def redirect(*args, logger=None, **kwargs):
    """
    Redirecting all kinds of stdout in Python via wurlitzer
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    :param logger 'logging.Logger': the logger to print the out and err messages.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with wurlitzer.pipes(out, err):
            yield None
    finally:
        if logger is None:
            return
        out = out.getvalue()
        if out:
            logger.warning(out)
        err = err.getvalue()
        if err:
            logger.warning(err)


class LogReader:
    """
    A class to read the log file.
    """

    INFO_SEP = ' INFO '
    TOTOAL_TIME = 'Task Total Timing: '
    MEMORY_RE = re.compile(MEMORY % '(\d+.\d+)')
    TIME_LEN = len(timeutils.ctime())

    def __init__(self, filepath, delay=False):
        """
        Initialize the LogReader object.

        :param filepath str: the log filepath
        :param delay bool: if True, delay the reading of the log file
        """
        self.filepath = filepath
        self.lines = None
        self.options = None
        self.sidx = None
        self.delay = delay
        if self.delay:
            return
        self.run()

    def run(self):
        """
        Run the log reader.
        """
        self.read()
        self.setOptions()

    def read(self):
        """
        Read the log file.
        """
        with open(self.filepath, 'r') as fh:
            self.lines = [
                x.split(self.INFO_SEP)[-1].strip() for x in fh.readlines()
            ]

    def setOptions(self):
        """
        Set the options from the log file.
        """
        block = None
        for idx, line in enumerate(self.lines):
            if line.endswith(COMMAND_OPTIONS_END):
                self.sidx = idx + 1
                break
            if block is not None:
                block.append(line)
            if line.endswith(COMMAND_OPTIONS_START):
                block = []
        options = {}
        for line in block:
            key, val = line.split(COLON_SEP)
            key = key.split()[-1]
            vals = val.split()
            options[key] = val if len(vals) == 1 else vals
        self.options = types.SimpleNamespace(**options)

    @property
    def task_time(self):
        """
        Return the total task time.

        :return 'datetime.timedelta': the task time
        """
        for line in self.lines[self.sidx:]:
            if not line.startswith(self.TOTOAL_TIME):
                continue
            task_time = line.split(self.TOTOAL_TIME)[-1].strip()
            return timeutils.str2delta(task_time)
        # No total task timing found (driver log instead of workflow log)
        return self.time()

    def time(self, dtype=DELTA):
        """
        Return the specific time. (Start, End, or Delta)

        :param dtype str: the starting time on START, the finishing time on END,
            and the time span on DELTA.
        :return: the specific time information
        :rtype: 'datetime.datetime' on START / END; 'datetime.timedelta' on DELTA
        """
        job_start = symbols.SPACE.join(self.options.JobStart)
        stime = timeutils.dtime(job_start)
        if dtype == START:
            return stime
        try:
            dtime = timeutils.dtime(self.lines[-1][-self.TIME_LEN:])
        except ValueError:
            return
        if dtype == END:
            return dtime
        delta = dtime - stime
        return delta

    @property
    def memory(self):
        """
        Return the peak memory usage.

        :return float: the peak memory usage
        """
        for line in self.lines[self.sidx:]:
            match = self.MEMORY_RE.search(line)
            if not match:
                continue
            return float(match.group(1))
