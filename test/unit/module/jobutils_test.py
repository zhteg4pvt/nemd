import json
import os
import shutil

import flow
import pytest

from nemd import envutils
from nemd import jobutils


class TestFunc:

    @pytest.mark.parametrize(
        'cmd,first,all_args',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], '1', ['1', '2']),
         ([], None, None)])
    def testGetArg(self, cmd, first, all_args):
        assert first == jobutils.get_arg(cmd, '-cru_num')
        val = first if first else 'val'
        assert val == jobutils.get_arg(cmd, '-cru_num', default='val')
        assert all_args == jobutils.get_arg(cmd, '-cru_num', first=False)
        vals = all_args if all_args else ['val']
        assert vals == jobutils.get_arg(cmd,
                                        '-cru_num',
                                        default=['val'],
                                        first=False)

    @pytest.mark.parametrize(
        'cmd,popped,num',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], ['1', '2'], 2),
         ([], None, 0), (['[Ar]', '-cru_num', '2', '-DEBUG'], ['2'], 2)])
    def testPopArg(self, cmd, popped, num):
        assert popped == jobutils.pop_arg(cmd, '-cru_num')
        assert num == len(cmd)

    @pytest.mark.parametrize('cmd,expected',
                             [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '2', '-DEBUG']),
                              ([], ['-cru_num', '5']),
                              (['[Ar]', '-cru_num', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '-DEBUG'])])
    def testSetArg(self, cmd, expected):
        assert expected == jobutils.set_arg(cmd, '-cru_num', '5')


class TestJob:

    NEMD_SRC = envutils.get_src()

    @pytest.fixture
    def raw(self, tmp_dir):
        return jobutils.Job()

    @pytest.fixture
    def job(self, jobname, dirname):
        if dirname:
            dirname = envutils.test_data('itest', dirname)
        return jobutils.Job(jobname, dirname=dirname if dirname else None)

    @pytest.fixture
    def copied(self, job, tmp_dir):
        shutil.copytree(job.dirname, os.curdir, dirs_exist_ok=True)
        job.dirname = os.curdir
        return job

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("jobname,evalue,expected",
                             [('myname', 'envname', 'myname'),
                              (None, 'envname', 'envname'),
                              (None, None, 'job')])
    def testInit(self, jobname, expected, env):
        assert expected == jobutils.Job(jobname).jobname

    @pytest.mark.parametrize('jobname', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname,expected', [('0046_test', 4)])
    def testData(self, job, expected):
        assert 3 == len(job.data)

    @pytest.mark.parametrize('dirname', ['0046_test'])
    @pytest.mark.parametrize('jobname,expected', [('mb_lmp_log', True),
                                                  ('mb_lmp_log2', False)])
    def testFile(self, job, expected):
        assert expected == os.path.isfile(job.file)

    @pytest.mark.parametrize('values', [(['first', 'second'])])
    @pytest.mark.parametrize("key", [('outfiles'), ('myfiles')])
    def testAppend(self, raw, key, values):
        for file in values:
            raw.append(file, key=key)
        assert raw.data[key] == values

    @pytest.mark.parametrize('value', [('file')])
    @pytest.mark.parametrize("key", [('outfile'), ('logfile')])
    def testSet(self, raw, key, value):
        raw.set(value, key=key)
        assert raw.data[key] == value

    def testWrite(self, raw):
        raw.set('my.log', key='logfile')
        raw.write()
        with open(raw.file) as fh:
            assert raw.data == json.load(fh)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('jobname', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname, key, expected',
                             [('0000', 'outfile', 'mb_lmp_log.log'),
                              ('0000', 'outfile2', '/mb_lmp_log2.log'),
                              (None, 'outfile', None)])
    def testGetFile(self, key, expected, dirname, job):
        file = job.getFile(key=key)
        assert file is None if expected is None else file.endswith(expected)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('jobname,dirname,expected',
                             [('mb_lmp_log', '0000', 'mb_lmp_log.log')])
    def testLogFile(self, expected, dirname, job):
        assert job.logfile.endswith(expected)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('jobname,dirname,expected',
                             [(None, '0000', 'mb_lmp_log.log')])
    def testGetJobs(self, expected, job):
        assert 4 == len([x.file for x in job.getJobs()])

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('jobname,dirname,expected',
                             [('mb_lmp_log', '0000', 'mb_lmp_log.log')])
    def testClean(self, expected, copied):
        assert os.path.exists(copied.file)
        copied.clean()
        assert not os.path.exists(copied.file)

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'name')])
    @pytest.mark.parametrize('jobname', [None, 'jobname'])
    @pytest.mark.parametrize('file', [False, True])
    @pytest.mark.parametrize('log', [False, True])
    def testReg(self, jobname, file, log, tmp_dir, env):
        jobutils.Job.reg('file', jobname=jobname, file=file, log=log)
        json_file = f".{jobname if jobname else 'name'}_document.json"
        with open(json_file) as fh:
            data = json.load(fh)
            assert ['file'] == data['outfiles']
            assert ('file' if file else None) == data.get('outfile')
            assert ('file' if log else None) == data.get('logfile')
