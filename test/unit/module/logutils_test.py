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
    def logger(self, name, debug, env, tmp_dir):
        logging.Logger.manager.loggerDict.pop(os.path.basename(name), None)
        with mock.patch('nemd.logutils.DEBUG') as mocked:
            mocked.__bool__.return_value = bool(debug)
            return logutils.Logger.get(name)

    @pytest.mark.parametrize('name,ekey,evalue', [('name', 'JOBNAME', 'jnm')])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testInfoJob(self, logger):
        logger.infoJob(types.SimpleNamespace(wa=1, hi=[3, 6]))
        with open('name.log') as fh:
            data = fh.read()
        assert '...Options...' in data
        assert 'wa: 1\n' in data
        assert 'hi: 3 6\n' in data
        assert 'JobStart:' in data

    @pytest.mark.parametrize('name,ekey,evalue', [('name', 'JOBNAME', 'jnm')])
    @pytest.mark.parametrize('debug', ['', '1'])
    @pytest.mark.parametrize('timestamp', [False, True])
    def testLog(self, timestamp, logger):
        logger.log('hi', timestamp=timestamp)
        with open('name.log') as fh:
            data = fh.read()
        assert (timestamp == False) == data.endswith('hi\n')

    @pytest.mark.parametrize('name,ekey,evalue', [('name', 'JOBNAME', 'jnm')])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testError(self, logger):
        with mock.patch('nemd.logutils.sys') as mocked:
            logger.error('hi')
            assert mocked.exit.called
        with open('name.log') as fh:
            lines = fh.readlines()
        assert 'hi\n' in lines[0]
        assert 'Aborting...\n' in lines[1]

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'jobname')])
    @pytest.mark.parametrize('name', ['/root/myname.py', 'myname'])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testGet(self, name, debug, logger):
        has_hdlr = not (name.endswith('.py') and not debug)
        assert has_hdlr == bool(logger.handlers)
        assert has_hdlr == os.path.isfile(symbols.FN_DOCUMENT)

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'jnm')])
    @pytest.mark.parametrize('name', ['name', '/root/name.py'])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testOneLine(self, name, debug, logger):
        with logger.oneLine(logging.DEBUG) as log:
            log('hi')
            log('wa')
            log('last')
        isfile = not (name.endswith('.py') and not debug)
        logfile = 'name.log' if name == 'name' else 'name.debug'
        assert isfile == os.path.isfile(logfile)
        if not isfile:
            return
        with open(logfile) as fh:
            data = fh.read()
        assert data.endswith('hi wa last \n') if debug else not data


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
