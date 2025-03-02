# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Creates loggers for modules and scripts, log messages and options, parse log
files, and redirect stdout and stderr.
"""
import contextlib
import io
import logging
import os
import re
import sys
import traceback
import types

import wurlitzer

from nemd import DEBUG
from nemd import envutils
from nemd import jobutils
from nemd import psutils
from nemd import symbols
from nemd import timeutils

STDERR = 'stderr'
JOBSTART = 'JobStart:'
FINISHED = 'Finished.'
OPTIONS = 'Options'
COLON_SEP = f'{symbols.COLON} '
PEAK_MEMORY_USAGE = 'Peak memory usage'


@contextlib.contextmanager
def redirect(*args, logger=None, **kwargs):
    """
    Redirecting all kinds of stdout in Python via wurlitzer
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    :param logger 'logging.Logger': the logger to print the out and err messages.
    :return dict: the redirected stdout and stderr
    """
    redirected = {}
    out = {x: io.StringIO() for x in ['stdout', STDERR]}
    try:
        with wurlitzer.pipes(**out):
            yield redirected
    finally:
        out = {x: y.getvalue() for x, y in out.items()}
        out = {x: y for x, y in out.items() if y}
        for key, value in out.items():
            redirected[key] = value
        if logger is None:
            return
        for key, msg in redirected.items():
            if key == STDERR:
                logger.info(f'The following {STDERR} is found:')
            logger.info(msg)


class Handler(logging.Handler):
    """
    This handler saves the records instead of printing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = {}

    def handle(self, record):
        """
        Record as dict key / value without of emitting.
        """
        key = record.levelname
        previous = self.logs.get(key)
        current = self.format(record)
        self.logs[key] = f"{previous}\n{current}" if previous else current
        return False


class Logger(logging.Logger):
    """
    A script logger to save information into a file.
    """
    OPTIONS_START = f'..........{OPTIONS}..........'
    OPTIONS_END = OPTIONS_START.replace(OPTIONS, '.' * len(OPTIONS))

    def __init__(self, *args, delay=False, **kwargs):
        """
        :param delay bool: if True, delay the reading of the log file
        """
        super().__init__(*args, **kwargs)
        if delay:
            return
        self.setUp()

    def setUp(self, ext=symbols.LOG_EXT):
        """
        Set up the logger. (e.g., level, handler)

        :param ext str: the filename extension of the FileHandler
        """
        self.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        basename, name_ext = os.path.splitext(self.name)
        if name_ext.startswith('.py'):
            if not DEBUG:
                return
            ext = '.debug'
        # File handler for driver/workflow in any mode and module in debug mode
        filename = f"{basename}{ext}"
        jobutils.add_outfile(filename)
        hdlr = logging.FileHandler(filename, mode='w')
        hdlr.setFormatter(logging.Formatter('%(message)s'))
        self.addHandler(hdlr)

    def infoJob(self, options):
        """
        Info the job options and the start time.

        :param options 'argparse.Namespace': command-line options
        """
        self.info(self.OPTIONS_START)
        for key, val in options.__dict__.items():
            if type(val) is list:
                val = symbols.SPACE.join(map(str, val))
            self.info(f"{key}{COLON_SEP}{val}")
        self.info(f"{JOBSTART} {timeutils.ctime()}")
        self.info(self.OPTIONS_END)

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
        self.log(msg)
        self.log('Aborting...', timestamp=timestamp)
        sys.exit(1)

    @classmethod
    def get(cls, name):
        """
        Get a module logger to print debug information.

        :param name str: logger name or the python script pathname
        :return 'logging.Logger': the logger
        """
        # Create new or retrieve previous
        logger_class = logging.getLoggerClass()
        logging.setLoggerClass(cls)
        logger = logging.getLogger(os.path.basename(name))
        logging.setLoggerClass(logger_class)
        return logger

    @contextlib.contextmanager
    def oneLine(self, level, separator=' '):
        """
        Print messages within one line to StreamHandler.

        :param level int: the logging level
        :param separator str: the separator between messages.
        :return `function`: the function to print one message as a word followed
            by a seperator within a line.
        """
        assert level != logging.ERROR
        try:
            terminators = {
                x: x.terminator
                for x in self.handlers if isinstance(x, logging.StreamHandler)
            }
            for handler in terminators.keys():
                handler.terminator = separator
            match level:
                case logging.DEBUG:
                    yield self.debug
                case logging.INFO:
                    yield self.log
                case logging.WARNING:
                    yield self.warning
                case logging.CRITICAL:
                    yield self.critical
        finally:
            for handler, terminator in terminators.items():
                handler.terminator = terminator
            if self.isEnabledFor(level):
                self._log(level, '', ())


class Script:
    """
    A class to handle the logging in running a script.
    """
    MEMORY_FMT = f'{PEAK_MEMORY_USAGE}: {{value:.4f}} MB'

    def __init__(self, options, **kwargs):
        """
        :param options str: the command-line options
        """
        self.options = options
        self.kwargs = {**dict(log=True), **kwargs}
        self.logger = None
        self.memory = None

    def __enter__(self):
        """
        Create the logger and start the memory monitoring if requested.

        :return `Logger`: the logger object to print messages
        """
        self.logger = Logger.get(self.options.JOBNAME)
        logfile = os.path.basename(self.logger.handlers[0].baseFilename)
        jobutils.add_outfile(logfile, **self.kwargs)
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
            self.logger.log(self.MEMORY_FMT.format(value=self.memory.result))
        if exc_type:
            if isinstance(exc_val, SystemExit):
                # error calls sys.exit(1)
                return
            self.logger.log(traceback.format_exc())
            raise exc_val
        self.logger.log(FINISHED, timestamp=True)


class Reader:
    """
    A class to read the log file.
    """

    TOTAL_TIME = 'Task Total Timing: '
    MEMORY_RE = re.compile(fr'{PEAK_MEMORY_USAGE}: (\d+.\d+) (\w+)')
    TIME_LEN = len(timeutils.ctime())
    START = 'start'
    END = 'end'
    DELTA = 'delta'

    def __init__(self, namepath, delay=False):
        """
        Initialize the LogReader object.

        :param namepath str: the log namepath
        :param delay bool: if True, delay the reading of the log file
        """
        self.namepath = namepath
        self.lines = None
        self.options = None
        self.delay = delay
        if self.delay:
            return
        self.read()
        self.setOptions()

    def read(self):
        """
        Read the log file.
        """
        with open(self.namepath, 'r') as fh:
            self.lines = [x.strip() for x in fh.readlines()]

    def setOptions(self):
        """
        Set the options from the log file.
        """
        options = {}
        for line in self.cropOptions():
            key, val = line.split(COLON_SEP)
            key = key.split()[-1]
            vals = val.split()
            options[key] = val if len(vals) == 1 else vals
        self.options = types.SimpleNamespace(**options)

    def cropOptions(self):
        """
        Crop the option lines.

        :return list: option lines.
        """
        for idx, line in enumerate(self.lines):
            match line:
                case Logger.OPTIONS_START:
                    start = idx
                case Logger.OPTIONS_END:
                    end = idx
                    break
        else:
            return []
        option_lines = self.lines[start + 1:end]
        self.lines = self.lines[end + 1:]
        return option_lines

    @property
    def task_time(self):
        """
        Return the total task time.

        :return 'datetime.timedelta': the task time
        """
        for line in self.lines:
            if not line.startswith(self.TOTAL_TIME):
                continue
            task_time = line.split(self.TOTAL_TIME)[-1].strip()
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
        if dtype == self.START:
            return stime
        try:
            dtime = timeutils.dtime(self.lines[-1][-self.TIME_LEN:])
        except ValueError:
            return
        if dtype == self.END:
            return dtime
        delta = dtime - stime
        return delta

    @property
    def memory(self):
        """
        Return the peak memory usage.

        :return float: the peak memory usage
        """
        for line in self.lines:
            match = self.MEMORY_RE.search(line)
            if not match:
                continue
            return float(match.group(1))


class Base(object):
    """
    A base class with a logger to print logging messages.
    """

    def __init__(self, logger=None):
        """
        :param logger 'logging.Logger': the logger to log messages
        """
        self.logger = logger

    def debug(self, msg):
        """
        Print this message into the log file in debug mode.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log(self, msg, **kwargs):
        """
        Print this message into the log file as information.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.log(msg, **kwargs)
        else:
            print(msg)

    def warning(self, msg):
        """
        Print this warning message into log file.

        :param msg str: the msg to be printed
        """
        self.log(f"WARNING: {msg}")

    def error(self, msg):
        """
        Print this message and exit the program.

        :param msg str: the msg to be printed
        """
        self.log(msg + '\nAborting...', timestamp=True)
        sys.exit(1)
