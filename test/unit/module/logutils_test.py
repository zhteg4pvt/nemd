import json
import logging
import os
import sys
import types
from unittest import mock

import pytest

from nemd import logutils
from nemd import symbols


def get_logger(name, *args, debug='', **kwargs):
    logger_name, _ = os.path.splitext(os.path.basename(name))
    logging.Logger.manager.loggerDict.pop(logger_name, None)
    with mock.patch('nemd.logutils.DEBUG') as mocked:
        mocked.__bool__.return_value = bool(debug)
        return logutils.get_logger(name, *args, **kwargs)


class TestFunc:

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'jobname')])
    @pytest.mark.parametrize('name', ['/root/myname.py', 'myname'])
    @pytest.mark.parametrize('log', [True, False])
    @pytest.mark.parametrize('file', [True, False])
    @pytest.mark.parametrize('debug', ['', '1'])
    def testGetLogger(self, name, log, file, debug, env, tmp_dir):
        logger = get_logger(name, debug=debug, log=log, file=file)
        has_hdlr = not (name.endswith('.py') and not debug)
        assert has_hdlr == bool(logger.handlers)
        assert has_hdlr == os.path.isfile(symbols.FN_DOCUMENT)
        if not has_hdlr:
            return
        with open(symbols.FN_DOCUMENT) as fh:
            data = json.load(fh)
            assert file == ('jobname' in data.get('outfile', {}))
            assert log == ('jobname' in data.get('logfile', {}))

    def testRedirect(self, capsys):
        logger = mock.Mock()
        with capsys.disabled():
            with logutils.redirect(logger=logger):
                sys.stdout.write("warning")
        logger.warning.assert_called_with('warning')
        with capsys.disabled():
            with logutils.redirect(logger=logger):
                sys.stderr.write("error")
        logger.warning.assert_called_with('error')


class TestHandler:

    @pytest.fixture
    def logger(self):
        logger = logging.Logger('test')
        logger.addHandler(logutils.Handler())
        return logger

    @pytest.mark.parametrize('level,key', [(logging.DEBUG, 'DEBUG'),
                                           (logging.INFO, 'INFO'),
                                           (logging.WARNING, 'WARNING'),
                                           (logging.ERROR, 'ERROR')])
    def testHandle(self, level, key, logger):
        logger.log(level=level, msg='first')
        logger.log(level=level, msg='second')
        logger.log(level=logging.CRITICAL, msg='third')
        logs = logger.handlers[0].logs
        assert 'third' == logs['CRITICAL']
        msg = None if level == logging.DEBUG else 'first\nsecond'
        assert msg == logs.get(key)


class TestFileHandler:

    @pytest.fixture
    def logger(self, fmt, tmp_dir):
        hdlr = logutils.FileHandler('name.log')
        hdlr.setFormatter(logging.Formatter(fmt))
        logger = logging.Logger('test')
        logger.addHandler(hdlr)
        return logger

    @pytest.mark.parametrize(
        'fmt', ['%(message)s', f'%(asctime)s %(levelname)s %(message)s'])
    @pytest.mark.parametrize('records,expected',
                             [(['hello', 'hi'], ['hello', 'hi']),
                              (['1[!n]', '2[!n]', '3'], ['123'])])
    def test(self, records, expected, logger):
        for record in records:
            logger.info(record)
        with open('name.log') as fh:
            assert expected == [x.split()[-1] for x in fh.readlines()]


class TestLogger:

    @pytest.fixture
    def logger(self, debug, tmp_dir):
        return get_logger('name', debug=debug)

    @pytest.mark.parametrize('debug', ['', '1'])
    def testInfoJob(self, logger):
        logger.infoJob(types.SimpleNamespace(wa=1, hi=[3, 6]))
        with open('name.log') as fh:
            data = fh.read()
        assert '...Options...' in data
        assert 'wa: 1\n' in data
        assert 'hi: 3 6\n' in data
        assert 'JobStart:' in data

    @pytest.mark.parametrize('debug', ['', '1'])
    @pytest.mark.parametrize('timestamp', [False, True])
    def testLog(self, timestamp, logger):
        logger.log('hi', timestamp=timestamp)
        with open('name.log') as fh:
            data = fh.read()
        assert (timestamp == False) == data.endswith('hi\n')

    @pytest.mark.parametrize('debug', ['', '1'])
    def testError(self, logger):
        with mock.patch('nemd.logutils.sys') as mocked:
            logger.error('hi')
            assert mocked.exit.called
        with open('name.log') as fh:
            lines = fh.readlines()
        assert 'hi\n' in lines[0]
        assert 'Aborting...\n' in lines[1]

    @pytest.mark.parametrize('debug', ['', '1'])
    @pytest.mark.parametrize('newline', [False, True])
    def testDebug(self, debug, logger, newline):
        logger.debug('hi', newline=newline)
        logger.debug('wa', newline=newline)
        logger.debug('last')
        with open('name.log') as fh:
            data = fh.read()
        assert (debug == '1') == bool(data)
        if debug != '1':
            return
        ending_str = 'DEBUG last\n' if newline else 'DEBUG hi, wa, last\n'
        assert data.endswith(ending_str)
