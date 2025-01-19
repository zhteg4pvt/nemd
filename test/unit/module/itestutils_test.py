import contextlib
import os
import types

import pytest

from nemd import envutils
from nemd import itest
from nemd import jobutils

BASE_DIR = envutils.get_nemd_src('test', 'data', 'itest')

TID = 1
JOB_DIR = os.path.join(BASE_DIR, 'ea8c25e09124635e93178c1725ae8ee7')


class Job(jobutils.Job):

    def __init__(self, tid=TID, flag_dir=None, job_dir=JOB_DIR):
        super().__init__(job_dir)
        self.tid = tid
        self.flag_dir = flag_dir
        if self.flag_dir is None:
            self.flag_dir = os.path.join(BASE_DIR, f"{self.tid:0>4}")
        self.statepoint[itest.FLAG_DIR] = self.flag_dir


class TestCmd:

    @pytest.fixture
    def cmd(self):
        return itest.Cmd(os.path.join(BASE_DIR, '0001'), delay=True)

    def testRead(self, cmd):
        cmd.read()
        assert len(cmd.args) == 2

    def testSetComment(self, cmd):
        cmd.read()
        cmd.setComment()
        assert cmd.comment == 'Amorphous builder on C'


class TestCmdJob:

    @pytest.fixture
    def job(self, tmp_dir):
        return itest.CmdJob(Job(job_dir=os.curdir), delay=True)

    def testParse(self, job):
        job.parse()
        assert job.comment == 'Amorphous builder on C'

    def testSetName(self, job):
        job.parse()
        job.setName()
        assert job.args[0].endswith('amorp_bldr')

    def testAddQuote(self, job):
        job.args = ['run_nemd amorp_bldr_driver.py C(C)']
        job.addQuote()
        assert job.args[0] == 'run_nemd amorp_bldr_driver.py "C(C)"'

    def testGetCmd(self, job, tmp_dir):
        job.run()
        cmd = job.getCmd()
        assert cmd

    def testPost(self, job):
        assert not job.post()
        job.doc['outfile'] = {'amorp_bldr': 'amorp_bldr.data'}
        assert job.post()


class TestExist:

    @pytest.fixture
    def exist(self, fn):
        return itest.Exist(fn, job=Job())

    @pytest.mark.parametrize("fn,is_raise,raise_type",
                             [('amorp_bldr.data', False, FileNotFoundError),
                              ('hi.data', True, FileNotFoundError)])
    def testRun(self, exist, fn, check_raise):
        with check_raise():
            exist.run()


class TestNotExist:

    @pytest.fixture
    def not_exist(self, fn):
        return itest.NotExist(fn, job=Job())

    @pytest.mark.parametrize("fn,is_raise,raise_type",
                             [('amorp_bldr.data', True, FileNotFoundError),
                              ('hi.data', False, FileNotFoundError)])
    def testRun(self, not_exist, fn, check_raise):
        with check_raise():
            not_exist.run()


class TestIn:

    @pytest.fixture
    def in_obj(self, containing):
        return itest.In(containing, 'amorp_bldr-driver.log', job=Job())

    @pytest.mark.parametrize("containing,is_raise,raise_type",
                             [('Finished.', False, ValueError),
                              ('Aborted..', True, ValueError)])
    def testRun(self, in_obj, containing, check_raise):
        with check_raise():
            in_obj.run()


class TestCmp:

    JOB42_DIR = os.path.join(BASE_DIR, '1c57f0964168565049315565b1388af9')
    JOB46_DIR = os.path.join(BASE_DIR, '81dc7a1e5728084cb77c2b7d3c8994fc')

    @pytest.fixture
    def cmp(self, tfn, jfn, atol, rtol, equal_nan, tid, job_dir):
        return itest.Cmp(tfn,
                         jfn,
                         atol=atol,
                         rtol=rtol,
                         equal_nan=equal_nan,
                         job=Job(tid=tid, job_dir=job_dir))

    @pytest.mark.parametrize(
        'tfn,jfn,atol,rtol,equal_nan,tid,job_dir,is_raise,raise_type',
        [('cmd', 'cmd', None, None, None, 1, JOB_DIR, True, FileNotFoundError),
         ('cmd', '0001_cmd', None, None, None, 1, JOB_DIR, False, ValueError),
         ('cmd', 'amorp_bldr-driver.log', None, None, None, 1, JOB_DIR, True,
          ValueError)])
    def testCmpFile(self, cmp, tfn, jfn, check_raise):
        with check_raise():
            cmp.run()

    @pytest.mark.parametrize(
        'tfn,jfn,atol,rtol,equal_nan,tid,job_dir,expected',
        [('', '', None, None, None, 1, '', {
            'atol': 1e-08,
            'rtol': 1e-05,
            'equal_nan': True
        }),
         ('', '', '2', '1e-05', 'False', 1, '', {
             'atol': 2.0,
             'rtol': 1e-05,
             'equal_nan': False
         })])
    def testKwargs(self, cmp, atol, rtol, equal_nan, expected):
        assert expected == cmp.kwargs

    @pytest.mark.parametrize(
        'tfn,jfn,atol,rtol,equal_nan,tid,job_dir,is_raise,raise_type',
        [('lmp_log_thermo.csv', 'lmp_log_toteng.csv', '1e-08', '1e-05', None,
          42, JOB42_DIR, True, ValueError),
         ('lmp_log_thermo.csv', 'lmp_log_toteng.csv', '2', '1e-05', None, 42,
          JOB42_DIR, False, ValueError),
         ('mb_lmp_log_toteng.csv', 'mb_lmp_log_toteng.csv', None, None, None,
          46, JOB46_DIR, False, ValueError),
         ('mb_lmp_log_toteng.csv', 'mb_lmp_log_toteng_diff.csv', None, None,
          None, 46, JOB46_DIR, True, ValueError),
         ('mb_lmp_log_toteng_diff.csv', 'mb_lmp_log_toteng_diff.csv', None,
          None, None, 46, JOB46_DIR, False, ValueError)])
    def testCmpCsv(self, cmp, atol, rtol, check_raise):
        with check_raise():
            cmp.cmpCsv()

    @pytest.mark.parametrize(
        'tfn,jfn,atol,rtol,equal_nan,tid,job_dir,is_raise,raise_type',
        [('polymer_builder.data', 'amorp_bldr.data', '1e-08', '1e-05', None, 1,
          JOB_DIR, True, ValueError),
         ('polymer_builder.data', 'amorp_bldr.data', '0.001', '1e-05', None, 1,
          JOB_DIR, False, ValueError)])
    def testCmpData(self, cmp, atol, rtol, check_raise):
        with check_raise():
            cmp.cmpData()


class TestCheck:

    @pytest.fixture
    def check(self, tid):
        return itest.Check(job=Job(tid=tid), delay=True)

    @pytest.mark.parametrize("tid", [(1)])
    def testSetOperators(self, check, tid):
        check.parse()
        check.setOperators()
        assert len(check.operators) == 1

    @pytest.mark.parametrize("tid,is_raise,raise_type",
                             [(1, False, FileNotFoundError),
                              (42, True, FileNotFoundError)])
    def testCheck(self, check, tid, check_raise):
        check.parse()
        check.setOperators()
        with check_raise(), contextlib.redirect_stdout(None):
            check.check()

    @pytest.mark.parametrize('tid,is_raise,raise_type', [(1, False, KeyError),
                                                         (0, True, KeyError)])
    def testGetClass(self, check, check_raise):
        check.parse()
        check.setOperators()
        with check_raise():
            check.getClass(check.operators[0])

    @pytest.mark.parametrize('tid', [(42)])
    def testGetArg(self, check):
        check.parse()
        check.setOperators()
        args, kwargs = check.getArg(check.operators[0])
        assert ['lmp_log_thermo.csv', 'lmp_log_toteng.csv'] == args
        assert {'rtol': '1e-6'} == kwargs


class TestCheckJob:

    @pytest.fixture
    def job(self):
        return itest.CheckJob(Job())

    def testRun(self, job):
        with contextlib.redirect_stdout(None):
            job.run()
        assert job.message is False

    def testPost(self, job):
        assert job.post() is False
        with contextlib.redirect_stdout(None):
            job.run()
        assert job.post() is True


class TestTag:

    @pytest.fixture
    def tag(self):
        return itest.Tag(job=Job())

    def testSetLogs(self, tag):
        tag.setLogs()
        assert len(tag.logs) == 1

    def testSetSlow(self, tag):
        tag.operators = []
        tag.setLogs()
        tag.setSlow()
        assert tag.operators[0][0] == 'slow'

    def testSet(self, tag):
        tag.set('wa', 1)
        assert ['wa', 1] == tag.operators[-1]

    def testSetLabel(self, tag):
        tag.operators = []
        tag.setLogs()
        tag.setLabel()
        assert tag.operators[0][0] == 'label'

    def testGet(self, tag):
        assert ['00:00:03'] == tag.get('slow')
        assert not tag.get('slow2')

    def testWrite(self, tag, tmp_dir):
        tag.pathname = os.path.basename(tag.pathname)
        with contextlib.redirect_stdout(None):
            tag.write()
        assert os.path.exists('tag')

    @pytest.mark.parametrize("slow,label,selected", [(0, 'amorp_bldr', False),
                                                     (5, 'wa', False),
                                                     (5, 'amorp_bldr', True)])
    def testSelected(self, tag, slow, label, selected):
        tag.options = types.SimpleNamespace(slow=slow, label=[label])
        assert selected == tag.selected()

    def testSlow(self, tag):
        tag.options = types.SimpleNamespace(slow=None)
        assert tag.slow() is False
        tag.options = types.SimpleNamespace(slow=2.)
        assert tag.slow() is True

    def testLabeled(self, tag):
        tag.options = types.SimpleNamespace(label=None)
        assert tag.labeled() is True
        tag.options = types.SimpleNamespace(label=['wa'])
        assert tag.labeled() is False
        tag.options = types.SimpleNamespace(label=['wa', 'amorp_bldr'])
        assert tag.labeled() is True


class TestTagJob:

    @pytest.fixture
    def tag(self):
        return itest.TagJob(job=Job(flag_dir=os.curdir))

    def testRun(self, tag, tmp_dir):
        with contextlib.redirect_stdout(None):
            tag.run()
        assert tag.message is False
        assert os.path.exists('tag')
