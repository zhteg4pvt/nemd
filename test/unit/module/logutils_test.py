import json
import logging
import os
import sys
import types
from unittest import mock

import pytest

from nemd import logutils
from nemd import symbols


class TestFunc:

    @pytest.mark.parametrize('is_err', [False, True])
    def testRedirect(self, is_err, capsys):
        logger = mock.Mock()
        with capsys.disabled():
            with logutils.redirect(logger=logger):
                (sys.stderr if is_err else sys.stdout).write('msg')
        logger.warning.assert_called_with('msg')


class TestHandler:

    @pytest.mark.parametrize('level,key', [(logging.DEBUG, 'DEBUG'),
                                           (logging.INFO, 'INFO'),
                                           (logging.WARNING, 'WARNING'),
                                           (logging.ERROR, 'ERROR')])
    def testHandle(self, level, key):
        hdlr = logutils.Handler()
        logger = logging.Logger('test')
        logger.addHandler(hdlr)
        logger.log(level=level, msg='first')
        logger.log(level=level, msg='second')
        logger.log(level=logging.CRITICAL, msg='third')
        assert 'third' == hdlr.logs['CRITICAL']
        msg = None if level == logging.DEBUG else 'first\nsecond'
        assert msg == hdlr.logs.get(key)


class TestLogger:

    @pytest.fixture
    def logger(self, tmp_dir):
        return logutils.Logger('name')

    @pytest.mark.parametrize('name', ['myname.py', 'myname'])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testSetUp(self, name, debug, tmp_dir):
        logger = logutils.Logger(name, delay=True)
        with mock.patch('nemd.logutils.DEBUG') as mocked:
            mocked.__bool__.return_value = bool(debug)
            logger.setUp()
        has_hdlr = not (name.endswith('.py') and not debug)
        assert has_hdlr == bool(logger.handlers)

    def testInfoJob(self, logger):
        logger.infoJob(types.SimpleNamespace(wa=1, hi=[3, 6]))
        with open(logger.handlers[0].baseFilename) as fh:
            data = fh.read()
        assert '...Options...' in data
        assert 'wa: 1\n' in data
        assert 'hi: 3 6\n' in data
        assert 'JobStart:' in data

    @pytest.mark.parametrize('timestamp', [False, True])
    def testLog(self, timestamp, logger):
        logger.log('hi', timestamp=timestamp)
        with open(logger.handlers[0].baseFilename) as fh:
            data = fh.read()
        assert (timestamp == False) == data.endswith('hi\n')

    def testError(self, logger):
        with mock.patch('nemd.logutils.sys') as mocked:
            logger.error('hi')
            assert mocked.exit.called
        with open(logger.handlers[0].baseFilename) as fh:
            lines = fh.readlines()
        assert 'hi\n' in lines[0]
        assert 'Aborting...\n' in lines[1]

    def testGet(self, tmp_dir):
        logging.Logger.manager.loggerDict.pop('jobname', None)
        created = logutils.Logger.get('jobname')
        assert isinstance(created, logutils.Logger)
        previous = logutils.Logger.get('jobname')
        assert created == previous

    def testOneLine(self, logger):
        with logger.oneLine(logging.DEBUG) as log:
            log('hi')
            log('wa')
            log('last')
        with open(logger.handlers[0].baseFilename) as fh:
            data = fh.read()
        assert data.endswith('hi wa last \n')


class TestScript:

    @pytest.mark.parametrize('ekey,evalue,log,file',
                             [('MEM_INTVL', '', False, True),
                              ('MEM_INTVL', '0.001', True, False)])
    def testEnter(self, env, evalue, log, file, tmp_dir):
        options = types.SimpleNamespace(hi='la', jobname='jobname')
        logging.Logger.manager.loggerDict.pop(options.jobname, None)
        with logutils.Script(options, log=log, file=file):
            pass
        with open(symbols.FN_DOCUMENT) as fh:
            data = json.load(fh)
        assert file == ('outfile' in data)
        assert log == ('logfile' in data)
        with open('jobname.log') as fh:
            data = fh.read()
        assert '..........Options..........\n' in data
        assert 'hi: la\n' in data
        assert bool(evalue) == ('Peak memory usage:' in data)

    @pytest.mark.parametrize('is_raise,raise_type,msg',
                             [(False, None, 'Finished.'),
                              (True, ValueError, "ValueError: ('hi',)"),
                              (True, SystemExit, "E_R_R_O_R\nAborting...")])
    def testExit(self, raise_type, check_raise, msg, tmp_dir):
        options = types.SimpleNamespace(jobname='jobname')
        logging.Logger.manager.loggerDict.pop(options.jobname, None)
        with check_raise():
            with logutils.Script(options) as logger:
                if raise_type:
                    if raise_type == SystemExit:
                        logger.error('E_R_R_O_R')
                    else:
                        raise raise_type('hi')
        with open('jobname.log') as fh:
            data = fh.read()
        assert msg in data