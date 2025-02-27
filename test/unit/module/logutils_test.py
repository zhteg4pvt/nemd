import json
import logging
import os
import sys
import types
from unittest import mock

import pytest

from nemd import envutils
from nemd import jobutils
from nemd import logutils


class TestFunc:

    @pytest.mark.parametrize('is_err', [False, True])
    @pytest.mark.parametrize('has_logger', [False, True])
    def testRedirect(self, is_err, has_logger, capsys):
        logger = mock.Mock() if has_logger else None
        with capsys.disabled():
            with logutils.redirect(logger=logger) as redirected:
                (sys.stderr if is_err else sys.stdout).write('msg\nmsg2\n')
        assert {'stderr' if is_err else 'stdout': 'msg\nmsg2\n'} == redirected
        if logger is None:
            return
        calls = ['msg\nmsg2\n']
        if is_err == 'stderr':
            calls = ['The following stderr is found:'] + calls
        logger.info.assert_has_calls([mock.call(x) for x in calls])


class TestHandler:

    @pytest.mark.parametrize('level,key', [(logging.DEBUG, 'DEBUG'),
                                           (logging.INFO, 'INFO'),
                                           (logging.WARNING, 'WARNING'),
                                           (logging.ERROR, 'ERROR')])
    def testHandle(self, level, key):
        hdlr = logutils.Handler(level=logging.INFO)
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
        with mock.patch('nemd.logutils.DEBUG', bool(debug)):
            logger.setUp()
        assert (logging.DEBUG if debug else logging.INFO) == logger.level
        has_hdlr = not (name.endswith('.py') and not debug)
        assert has_hdlr == bool(logger.handlers)
        assert has_hdlr == os.path.isfile(jobutils.FN_DOCUMENT)
        if not has_hdlr:
            return
        with open(jobutils.FN_DOCUMENT) as fh:
            data = json.load(fh)
        filename = f"myname.{'debug' if name.endswith('.py')  else 'log'}"
        assert [filename] in data['outfiles'].values()

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

    @pytest.mark.parametrize('timestamp', [False, True])
    def testError(self, timestamp, logger):
        with mock.patch('nemd.logutils.sys') as mocked:
            logger.error('hi', timestamp=timestamp)
            assert mocked.exit.called
        with open(logger.handlers[0].baseFilename) as fh:
            lines = fh.readlines()
        assert 'hi\n' in lines[0]
        assert 'Aborting...\n' in lines[1]
        assert (3 if timestamp else 2) == len(lines)

    def testGet(self, tmp_dir):
        logging.Logger.manager.loggerDict.pop('jobname', None)
        created = logutils.Logger.get('jobname')
        assert isinstance(created, logutils.Logger)
        previous = logutils.Logger.get('jobname')
        assert created == previous

    @pytest.mark.parametrize('logger_lvl, line_lvl',
                             [(logging.WARNING, logging.WARNING),
                              (logging.DEBUG, logging.INFO),
                              (logging.CRITICAL, logging.INFO)])
    def testOneLine(self, logger_lvl, line_lvl, logger):
        logger.setLevel(logger_lvl)
        with logger.oneLine(line_lvl) as log:
            log('hi')
            log('wa')
            log('last')
        with open(logger.handlers[0].baseFilename) as fh:
            data = fh.read()
        expected = 'hi wa last \n' if line_lvl >= logger_lvl else ''
        assert expected == data


class TestScript:

    @pytest.mark.parametrize('ekey,evalue,log,file',
                             [('MEM_INTVL', '', False, True),
                              ('MEM_INTVL', '0.001', True, False)])
    def testEnter(self, evalue, env, log, file, tmp_dir):
        options = types.SimpleNamespace(hi='la', JOBNAME='jobname')
        logging.Logger.manager.loggerDict.pop(options.JOBNAME, None)
        with mock.patch('nemd.logutils.psutils.Memory') as mocked:
            with mock.patch('nemd.logutils.Script.__exit__') as mocked:
                with logutils.Script(options, log=log, file=file):
                    assert bool(evalue) == mocked.called
        with open(jobutils.FN_DOCUMENT) as fh:
            data = json.load(fh)
        assert file == ('outfile' in data)
        assert log == ('logfile' in data)
        with open('jobname.log') as fh:
            data = fh.read()
        assert '..........Options..........\n' in data
        assert 'hi: la\n' in data

    @pytest.mark.parametrize('ekey,evalue', [('MEM_INTVL', '0.001')])
    @pytest.mark.parametrize('is_raise,raise_type,msg',
                             [(False, None, 'Finished.'),
                              (True, ValueError, "ValueError: ('hi',)"),
                              (True, SystemExit, "E_R_R_O_R\nAborting...")])
    def testExit(self, evalue, raise_type, check_raise, msg, env, tmp_dir):
        options = types.SimpleNamespace(JOBNAME='jobname')
        logging.Logger.manager.loggerDict.pop(options.JOBNAME, None)
        with check_raise():
            with logutils.Script(options) as logger:
                if raise_type:
                    if raise_type == SystemExit:
                        logger.error('E_R_R_O_R')
                    else:
                        raise raise_type('hi')
        with open('jobname.log') as fh:
            data = fh.read()
        assert bool(evalue) == ('Peak memory usage:' in data)
        assert msg in data


class TestReader:
    ITEST_DIR = envutils.test_data('itest')
    AMORP_LOG = os.path.join(ITEST_DIR, '5524d62a356ac00d781a9cb1e5a6f03b',
                             'amorp_bldr.log')
    MB_LMP_LOG = os.path.join(ITEST_DIR, '81dc7a1e5728084cb77c2b7d3c8994fc',
                              'mb_lmp_log-driver.log')

    @pytest.fixture
    def raw(self, data):
        reader = logutils.Reader(data, delay=True)
        reader.read()
        return reader

    @pytest.fixture
    def reader(self, data):
        return logutils.Reader(data)

    @pytest.mark.parametrize('data,num', [(AMORP_LOG, 36), (MB_LMP_LOG, 59)])
    def testRead(self, num, raw):
        assert num == len(raw.lines)

    @pytest.mark.parametrize('data,debug', [(AMORP_LOG, 'None'),
                                            (MB_LMP_LOG, 'False')])
    def testSetOptions(self, debug, raw):
        raw.setOptions()
        assert debug == raw.options.debug
        assert 2 == len(raw.options.JobStart)

    @pytest.mark.parametrize('data,onum,num', [(AMORP_LOG, 28, 6),
                                               (MB_LMP_LOG, 34, 23)])
    def testCropOptions(self, onum, num, raw):
        assert onum == len(raw.cropOptions())
        assert num == len(raw.lines)

    @pytest.mark.parametrize('data,task_time', [(AMORP_LOG, '0:00:09'),
                                                (MB_LMP_LOG, '0:00:10')])
    def testTaskTime(self, task_time, reader):
        assert task_time == str(reader.task_time)

    @pytest.mark.parametrize('data', [AMORP_LOG])
    @pytest.mark.parametrize('dtype,expected',
                             [('start', '2025-02-15 19:46:58'),
                              ('end', '2025-02-15 19:47:07'),
                              ('delta', '0:00:09')])
    def testTime(self, dtype, expected, reader):
        assert expected == str(reader.time(dtype=dtype))

    @pytest.mark.parametrize('data,mem', [(AMORP_LOG, 183.6974),
                                          (MB_LMP_LOG, None)])
    def testMemory(self, mem, reader):
        assert mem == reader.memory


class TestBase:

    @pytest.fixture
    def base(self):
        return logutils.Base(logger=mock.Mock())

    @mock.patch('nemd.logutils.print')
    @pytest.mark.parametrize('has_logger', [False, True])
    def testDebug(self, mocked, has_logger):
        base = logutils.Base(logger=mock.Mock() if has_logger else None)
        base.debug('msg')
        (base.logger.debug if has_logger else mocked).assert_called_with('msg')

    @mock.patch('nemd.logutils.print')
    @pytest.mark.parametrize('has_logger', [False, True])
    def testLog(self, mocked, has_logger):
        base = logutils.Base(logger=mock.Mock() if has_logger else None)
        base.log('msg')
        (base.logger.log if has_logger else mocked).assert_called_with('msg')

    def testWarning(self, base):
        base.warning('msg')
        base.logger.log.assert_called_with("WARNING: msg")

    @mock.patch('nemd.logutils.sys')
    def testError(self, mocked, base):
        base.error('msg')
        base.logger.log.assert_called_with("msg\nAborting...", timestamp=True)
        mocked.exit.assert_called_with(1)
