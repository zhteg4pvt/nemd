import functools
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

    @pytest.mark.parametrize('mtype', ['stdout', 'stderr'])
    def testRedirect(self, mtype, capsys):
        logger = mock.Mock()
        with capsys.disabled():
            with logutils.redirect(logger=logger) as redirected:
                getattr(sys, mtype).write('msg\nmsg2\n')
        assert {mtype: 'msg\nmsg2\n'} == redirected
        calls = ['The following stderr is found:'] if mtype == 'stderr' else []
        calls += ['msg\nmsg2\n']
        logger.info.assert_has_calls([mock.call(x) for x in calls])


class TestHandler:

    @pytest.mark.parametrize(
        'hdlr_lvl',
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
    @pytest.mark.parametrize('level,key', [(logging.DEBUG, 'DEBUG'),
                                           (logging.INFO, 'INFO'),
                                           (logging.WARNING, 'WARNING'),
                                           (logging.ERROR, 'ERROR')])
    def testHandle(self, hdlr_lvl, level, key):
        hdlr = logutils.Handler(level=hdlr_lvl)
        logger = logging.Logger('test')
        logger.addHandler(hdlr)
        logger.log(level=level, msg='first')
        logger.log(level=level, msg='second')
        msg = None if level < hdlr_lvl else 'first\nsecond'
        assert msg == hdlr.logs.get(key)
        logger.log(level=logging.CRITICAL, msg='third')
        assert 'third' == hdlr.logs['CRITICAL']


class Logger(logutils.Logger):

    @property
    @functools.cache
    def data(self):
        with open(self.handlers[0].baseFilename) as fh:
            return fh.read()


class TestLogger:

    @pytest.fixture
    def logger(self, tmp_dir):
        return Logger('name')

    @pytest.mark.parametrize('ekey', ['DEBUG'])
    @pytest.mark.parametrize('evalue, name,file',
                             [('', 'myname.py', None),
                              ('1', 'myname.py', 'myname.debug'),
                              ('', 'myname', 'myname.log'),
                              ('1', 'myname', 'myname.log')])
    def testSetUp(self, name, evalue, file, tmp_dir, env):
        logger = logutils.Logger(name, delay=True)
        logger.setUp()
        assert (logging.DEBUG if evalue else logging.INFO) == logger.level
        assert bool(file) == bool(logger.handlers)
        if file:
            assert os.path.isfile(file)

    def testInfoJob(self, logger):
        logger.infoJob(types.SimpleNamespace(wa=1, hi=[3, 6]))
        assert '...Options...' in logger.data
        assert 'wa: 1\n' in logger.data
        assert 'hi: 3 6\n' in logger.data
        assert 'JobStart:' in logger.data

    @pytest.mark.parametrize('timestamp', [False, True])
    def testLog(self, timestamp, logger):
        logger.log('hi', timestamp=timestamp)
        assert (timestamp == False) == logger.data.endswith('hi\n')

    @pytest.mark.parametrize('timestamp', [False, True])
    def testError(self, timestamp, logger):
        with mock.patch('nemd.logutils.sys') as mocked:
            logger.error('hi', timestamp=timestamp)
            assert mocked.exit.called
        lines = logger.data.split('\n')
        assert 'hi' == lines[0]
        assert 'Aborting...' == lines[1]
        assert timestamp == bool(lines[2])

    def testGet(self, tmp_dir):
        logging.Logger.manager.loggerDict.pop('jobname', None)
        created = logutils.Logger.get('jobname')
        assert isinstance(created, logutils.Logger)
        previous = logutils.Logger.get('jobname')
        assert created is previous

    @pytest.mark.parametrize('logger_lvl, lvl',
                             [(logging.WARNING, logging.WARNING),
                              (logging.DEBUG, logging.INFO),
                              (logging.CRITICAL, logging.INFO)])
    def testOneLine(self, logger_lvl, lvl, logger):
        logger.setLevel(logger_lvl)
        with logger.oneLine(lvl) as log:
            log('hi')
            log('wa')
            log('last')
        assert ('' if lvl < logger_lvl else 'hi wa last \n') == logger.data


@mock.patch('nemd.logutils.Logger', Logger)
@pytest.mark.parametrize('options',
                         [types.SimpleNamespace(hi='la', JOBNAME='hi')])
class TestScript:

    @pytest.mark.parametrize('ekey', ['MEM_INTVL'])
    @pytest.mark.parametrize('evalue', ['', '0.001'])
    @pytest.mark.parametrize('log,file', [(False, True), (True, False)])
    def testEnter(self, evalue, options, log, file, env, tmp_dir):
        logging.Logger.manager.loggerDict.pop(options.JOBNAME, None)
        with mock.patch('nemd.logutils.psutils.Memory') as mocked:
            with mock.patch('nemd.logutils.Script.__exit__'):
                with logutils.Script(options, log=log, file=file) as logger:
                    assert bool(evalue) == mocked.called
        job = jobutils.Job()
        assert ['hi.log'] == job._outfiles
        assert file == ('hi.log' == job._outfile)
        rdr = logutils.Reader(job._outfiles[0])
        rdr.read()
        assert 5 == len(rdr.lines)

    @pytest.mark.parametrize('ekey', ['MEM_INTVL'])
    @pytest.mark.parametrize('evalue', ['', '0.001'])
    @pytest.mark.parametrize('expected,msg',
                             [(None, 'Finished.'),
                              (ValueError, "ValueError: hi"),
                              (SystemExit, "msg\nAborting...")])
    def testExit(self, evalue, options, msg, expected, raises, env, tmp_dir):
        logging.Logger.manager.loggerDict.pop(options.JOBNAME, None)
        with raises:
            with logutils.Script(options) as logger:
                if expected is None:
                    logger.log('')
                elif expected is ValueError:
                    raise expected('hi')
                elif expected is SystemExit:
                    logger.error('msg')
        assert bool(evalue) == ('Peak memory usage:' in logger.data)
        assert msg in logger.data


@pytest.mark.skipif(envutils.get_src() is None,
                    reason="cannot locate test dir")
class TestReader:
    AMORP_LOG = envutils.test_data('0001_test', 'workspace',
                                   'cab913cd09a661ad4e2585885ebed4a9',
                                   'amorp_bldr.log')
    MB_LMP_LOG = envutils.test_data('0046_test', 'mb_lmp_log.log')

    @pytest.fixture
    def raw(self, data):
        reader = logutils.Reader(data, delay=True)
        reader.read()
        return reader

    @pytest.fixture
    def reader(self, data):
        return logutils.Reader(data)

    @pytest.mark.parametrize('data,num', [(AMORP_LOG, 34), (MB_LMP_LOG, 59)])
    def testRead(self, num, raw):
        assert num == len(raw.lines)

    @pytest.mark.parametrize('data,debug', [(AMORP_LOG, 'None'),
                                            (MB_LMP_LOG, 'False')])
    def testSetOptions(self, debug, raw):
        raw.setOptions()
        assert debug == raw.options.DEBUG
        assert 2 == len(raw.options.JobStart)

    @pytest.mark.parametrize('data,onum,num', [(AMORP_LOG, 26, 6),
                                               (MB_LMP_LOG, 35, 22)])
    def testCropOptions(self, onum, num, raw):
        assert onum == len(raw.cropOptions())
        assert num == len(raw.lines)

    @pytest.mark.parametrize('data,task_time', [(AMORP_LOG, '0:00:01'),
                                                (MB_LMP_LOG, '0:00:07')])
    def testTaskTime(self, task_time, reader):
        assert task_time == str(reader.task_time)

    @pytest.mark.parametrize('data', [AMORP_LOG])
    @pytest.mark.parametrize('dtype,expected',
                             [('start', '2025-06-09 11:02:56'),
                              ('end', '2025-06-09 11:02:57'),
                              ('delta', '0:00:01')])
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
