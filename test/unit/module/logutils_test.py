import json
import logging
import os
import sys
from unittest import mock

import pytest

from nemd import logutils
from nemd import symbols


class TestFunc:

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'jobname')])
    @pytest.mark.parametrize('name', ['/root/myname.py', 'myname'])
    @pytest.mark.parametrize('log', [True, False])
    @pytest.mark.parametrize('file', [True, False])
    @pytest.mark.parametrize('DEBUG', ['', '1'])
    def testGetLogger(self, name, log, file, DEBUG, env, tmp_dir):
        logger = logging.getLogger('myname')
        for handler in logger.handlers:
            logger.removeHandler(handler)
        with mock.patch('nemd.logutils.DEBUG') as mocked:
            mocked.__bool__.return_value = bool(DEBUG)
            logger = logutils.get_logger(name, log=log, file=file)
        has_hdlr = not (name.endswith('.py') and not DEBUG)
        assert has_hdlr == bool(logger.handlers)
        assert has_hdlr == os.path.isfile(symbols.FN_DOCUMENT)
        if not has_hdlr:
            return
        with open(symbols.FN_DOCUMENT) as fh:
            data = json.load(fh)
            assert file == ('jobname' in data.get('outfile', {}))
            assert log == ('jobname' in data.get('logfile', {}))

    def testRedirect(self):
        logger = mock.Mock()
        with logutils.redirect(logger=logger):
            sys.stdout.write("warning")
        logger.warning.assert_called_with('warning')
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

    ...
