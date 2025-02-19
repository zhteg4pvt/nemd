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
import types

import wurlitzer

from nemd import DEBUG
from nemd import envutils
from nemd import jobutils
from nemd import psutils
from nemd import symbols
from nemd import timeutils

JOBSTART = 'JobStart:'
FINISHED = 'Finished.'
OPTIONS = 'Options'
OPTIONS_START = f'..........{OPTIONS}..........'
OPTIONS_END = OPTIONS_START.replace(OPTIONS, '.' * len(OPTIONS))
COLON_SEP = f'{symbols.COLON} '
PEAK_MEMORY_USAGE = 'Peak memory usage'


@contextlib.contextmanager
def redirect(*args, logger=None, **kwargs):
    """
    Redirecting all kinds of stdout in Python via wurlitzer
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    :param logger 'logging.Logger': the logger to print the out and err messages.
    """
    stdout = io.StringIO()
    try:
        with wurlitzer.pipes(stdout=stdout, stderr=wurlitzer.STDOUT):
            yield None
    finally:
        if logger is None:
            return
        out = stdout.getvalue()
        if out:
            logger.warning(out)


class Handler(logging.Handler):
    """
    This handler saves the records instead of printing.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
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
    A logger for driver so that customer-facing information can be saved.
    """

    def __init__(self, *args, **kwargs):
        """
        :param level int: the level of the logger
        """
        kwargs.setdefault('level', logging.DEBUG if DEBUG else logging.INFO)
        super().__init__(*args, **kwargs)

    def infoJob(self, options):
        """
        Info the job options and the start time.

        :param options 'argparse.Namespace': command-line options
        """
        self.info(OPTIONS_START)
        for key, val in options.__dict__.items():
            if type(val) is list:
                val = symbols.SPACE.join(map(str, val))
            self.info(f"{key}{COLON_SEP}{val}")
        self.info(f"{JOBSTART} {timeutils.ctime()}")
        self.info(OPTIONS_END)

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
    def get(cls, name, log=False, file=False, fmt='%(message)s'):
        """
        Get a module logger to print debug information.

        :param name str: logger name or the python script pathname
        :param log bool: sets as the log file if True
        :param file bool: set this file as the single output file
        :param fmt str: the formatter of the handler
        :return 'logging.Logger': the logger
        """
        if DEBUG:
            fmt = f'%(asctime)s %(levelname)s {fmt}'
        isfile = name.endswith('.py')
        if isfile:
            name, _ = os.path.splitext(os.path.basename(name))
        # Either create new or retrieve previous logger
        logger_class = logging.getLoggerClass()
        logging.setLoggerClass(cls)
        logger = logging.getLogger(name)
        logging.setLoggerClass(logger_class)
        if logger.handlers:
            return logger
        # File handler
        if isfile and not DEBUG:
            # No file handler for module logger outside the debug mode
            return logger
        # File handler for driver/workflow in any mode and module in debug mode
        outfile = f"{name}{'.debug' if isfile else symbols.LOG_EXT}"
        jobutils.add_outfile(outfile, file=file, log=log)
        hdlr = logging.FileHandler(outfile, mode='w')
        hdlr.setFormatter(logging.Formatter(fmt))
        logger.addHandler(hdlr)
        return logger

    @contextlib.contextmanager
    def oneLine(self, level, separator=' ', fmt='%(message)s'):
        """
        Print messages within one line to StreamHandler.

        :param level int: the logging level
        :param separator str: the separator between messages.
        :param fmt str: the formatter of one message.
        :
        """
        fmt = logging.Formatter(fmt)
        try:
            stream_handlers = [
                x for x in self.handlers
                if isinstance(x, logging.StreamHandler)
            ]
            terminators = {x: x.terminator for x in stream_handlers}
            for handler in stream_handlers:
                handler.terminator = ''
            if self.isEnabledFor(level):
                self._log(level, '', ())
            for handler in stream_handlers:
                handler.terminator = separator
            formatters = {x: x.formatter for x in stream_handlers}
            for handler in stream_handlers:
                handler.setFormatter(fmt)
            yield self.debug if level == logging.DEBUG else self.log
        finally:
            for handler, terminator in terminators.items():
                handler.terminator = terminator
            if self.isEnabledFor(level):
                self._log(level, '', ())
            for handler, formatter in formatters.items():
                handler.setFormatter(formatter)


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
        self.logger = None
        self.memory = None
        self.kwargs = kwargs
        self.outfile = self.options.jobname + symbols.LOG_EXT
        jobutils.add_outfile(self.outfile, **kwargs)

    def __enter__(self):
        """
        Create the logger and start the memory monitoring if requested.

        :return `Logger`: the logger object to print messages
        """
        self.logger = Logger.get(self.options.jobname, log=True, **self.kwargs)
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
            self.logger.log(f"{self.MEMORY_FMT.format(self.memory.result)}")
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


class Reader:
    """
    A class to read the log file.
    """

    INFO_SEP = ' INFO '
    TOTOAL_TIME = 'Task Total Timing: '
    MEMORY_RE = re.compile(fr'{PEAK_MEMORY_USAGE}: (\d+.\d+) (\w+)')
    TIME_LEN = len(timeutils.ctime())
    START = 'start'
    END = 'end'
    DELTA = 'delta'

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
            if line.endswith(OPTIONS_END):
                self.sidx = idx + 1
                break
            if block is not None:
                block.append(line)
            if line.endswith(OPTIONS_START):
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
        for line in self.lines[self.sidx:]:
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
