import datetime
import functools
import logging
import os
import sys
import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from nemd import envutils
from nemd import jobutils
from nemd import logutils
from nemd import parserutils


class TestFunc:

    @pytest.mark.parametrize('mtype', ['stdout', 'stderr'])
    @pytest.mark.parametrize('logger', [mock.Mock(), None])
    def testRedirect(self, mtype, logger, capsys):
        with capsys.disabled():
            with logutils.redirect(logger=logger) as redirected:
                getattr(sys, mtype).write('msg\nmsg2\n')
        assert {mtype: 'msg\nmsg2\n'} == redirected
        calls = ['The following stderr is found:'] if mtype == 'stderr' else []
        calls += ['msg\nmsg2\n']
        if logger is None:
            return
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

    @functools.cached_property
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

    @pytest.mark.parametrize(
        'orig,lvl,num,expected',
        [(logging.WARNING, logging.WARNING, 3, '33% 66% 100%\n'),
         (logging.DEBUG, logging.INFO, 3, '33% 66% 100%\n'),
         (logging.CRITICAL, logging.INFO, 3, ''),
         (logging.INFO, logging.INFO, 12,
          '8% 16% 25% 33% 41% 50% 66% 75% 83% 91% 100%\n')])
    def testProgress(self, orig, lvl, num, expected, logger):
        logger.setLevel(orig)
        with logger.progress(num, level=lvl) as prog:
            for idx in range(num):
                prog(idx + 1)
        assert expected == logger.data


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

    @pytest.mark.parametrize(
        'parser',
        [parserutils.Driver(), parserutils.Workflow()])
    def testRun(self, parser, options, tmp_dir):
        mocked = mock.Mock()
        Main = type('Main', (object, ),
                    dict(__init__=lambda self, x, **y: None, run=mocked))
        with mock.patch('sys.argv', []):
            logutils.Script.run(Main, parser)
        assert mocked.called


@pytest.mark.skipif(envutils.get_src() is None,
                    reason="cannot locate test dir")
class TestReader:
    AMORP_LOG = envutils.test_data('0001_test', 'workspace',
                                   '0aee44e791ffa72655abcc90e25355d8',
                                   'amorp_bldr.log')
    MB_LMP_LOG = envutils.test_data('0046_test', 'mb_lmp_log.log')
    TEST001_LOG = envutils.test_data('0001_fail', 'test.log')
    EMPTY_LOG = envutils.test_data('ar', 'empty.log')
    TEST0049 = os.path.join('0049_test', 'workspace',
                            '3ec5394f589c9363bd15af35d45a7c44')
    TEST0035 = os.path.join('0035_test', 'workspace',
                            'a080ae2cf5e6ba8233bd80fc4cfde5d3')

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
        assert isinstance(raw.options.NAME, str)

    @pytest.mark.parametrize('data,onum,num', [(AMORP_LOG, 26, 6),
                                               (MB_LMP_LOG, 35, 22),
                                               (EMPTY_LOG, 0, 0)])
    def testCropOptions(self, onum, num, raw):
        assert onum == len(raw.cropOptions())
        assert num == len(raw.lines)

    @pytest.mark.parametrize('data,task_time', [(AMORP_LOG, '0:00:01'),
                                                (MB_LMP_LOG, '0:00:07')])
    def testTaskTime(self, task_time, reader):
        assert task_time == str(reader.task_time)

    @pytest.mark.parametrize('data', [AMORP_LOG])
    @pytest.mark.parametrize('dtype,expected',
                             [('start', '2025-07-06 09:26:19'),
                              ('end', '2025-07-06 09:26:20'),
                              ('delta', '0:00:01')])
    def testTime(self, dtype, expected, reader):
        assert expected == str(reader.time(dtype=dtype))

    @pytest.mark.parametrize('data,mem', [(AMORP_LOG, 183.6974),
                                          (MB_LMP_LOG, None)])
    def testMemory(self, mem, reader):
        assert mem == reader.memory

    @pytest.mark.parametrize(
        'data,mem', [(AMORP_LOG, datetime.datetime(2025, 7, 6, 9, 26, 20)),
                     (MB_LMP_LOG, datetime.datetime(2025, 4, 27, 18, 41, 37)),
                     (TEST001_LOG, None)])
    def testFinished(self, mem, reader):
        assert mem == reader.finished

    @pytest.mark.parametrize('dirname,columns,expected',
                             [('0049', ['task_time'], (0, 1, np.int64)),
                              ('0058_test', ['task_time'], (2, 1, np.float64)),
                              (TEST0049, ['task_time'], (2, 1, np.int64)),
                              (TEST0049, ['finished'], (2, 1, np.int64)),
                              ('0001_fail', ['task_time'], (1, 1, np.object_)),
                              ('0001_fail', ['finished'], (1, 0, np.object_)),
                              (TEST0049, ['task_time', 'memory'],
                               (2, 1, np.int64)),
                              ('0049_ubuntu', ['task_time', 'memory'],
                               (2, 2, np.int64)),
                              (TEST0035, ['task_time'], (2, 1, np.object_)),
                              ('0049_ubuntu', ['memory'], (2, 1, np.int64))])
    def testCollect(self, columns, expected, copied):
        collected = logutils.Reader.collect(*columns)
        assert expected == (*collected.shape, collected.index.dtype)

    @pytest.mark.parametrize('data,expected',
                             [([[0, 1], [1, 2]], np.int64),
                              ([['1', 1], ['0', 2]], np.int64),
                              ([['1.1', 1], ['0', 2]], np.float64)])
    def testSort(self, data, expected):
        data = pd.DataFrame(data).set_index(0)
        logutils.Reader.sort(data)
        assert expected == data.index.dtype
        assert (data.index == sorted(data.index)).all()

    @pytest.mark.parametrize('data', [AMORP_LOG])
    @pytest.mark.parametrize('attr,expected',
                             [('task_time', datetime.timedelta(seconds=1)),
                              ('cru_num', '1'), ('not_exits', None)])
    def testGet(self, reader, attr, expected):
        assert expected == reader.get(attr)


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

    def testWarning(self, base):
        base.warning('msg')
        base.logger.log.assert_called_with("WARNING: msg")

    @mock.patch('nemd.logutils.print')
    @pytest.mark.parametrize('has_logger', [False, True])
    def testLog(self, mocked, has_logger):
        base = logutils.Base(logger=mock.Mock() if has_logger else None)
        base.log('msg')
        (base.logger.log if has_logger else mocked).assert_called_with('msg')

    @mock.patch('nemd.logutils.sys')
    @pytest.mark.parametrize(
        'has_logger,expected',
        [(False, ['msg', {}]),
         (True, ["Aborting...", dict(timestamp=True)])])
    def testError(self, mocked, base, has_logger, expected):
        if not has_logger:
            base.logger = None
        base.log = mock.Mock()
        base.error('msg')
        mocked.exit.assert_called_with(1)
        base.log.assert_called_with(expected[0], **expected[1])
