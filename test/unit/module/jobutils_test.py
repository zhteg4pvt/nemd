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

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'name')])
    @pytest.mark.parametrize('jobname', [None, 'jobname'])
    @pytest.mark.parametrize('file', [False, True])
    @pytest.mark.parametrize('log', [False, True])
    def testAddOutfile(self, jobname, file, log, tmp_dir, env):
        jobutils.add_outfile('file', jobname=jobname, file=file, log=log)
        json_file = f".{jobname if jobname else 'name'}_document.json"
        with open(json_file) as fh:
            data = json.load(fh)
            assert ['file'] == data['outfiles']
            assert ('file' if file else None) == data.get('outfile')
            assert ('file' if log else None) == data.get('logfile')


class TestJob:

    MB_LMP_LOG = 'test_mb_lmp_log'
    NEMD_SRC = envutils.get_src()

    @pytest.fixture
    def raw(self, tmp_dir):
        return jobutils.Job()

    @pytest.fixture
    def job(self, name, dirname):
        if dirname is None:
            return jobutils.Job(name=name)
        test_dir = envutils.test_data('itest', dirname)
        jobs = flow.project.FlowProject.get_project(test_dir).find_jobs()
        return jobutils.Job(name=name, job=list(jobs)[0])

    @pytest.fixture
    def copied(self, name, dirname, tmp_dir):
        test_dir = envutils.test_data('itest', dirname)
        shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
        jobs = flow.project.FlowProject.get_project(os.curdir).find_jobs()
        return jobutils.Job(name=name, job=list(jobs)[0])

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("evalue,expected", [('myname', 'myname'),
                                                 (None, 'job')])
    def testDefault(self, expected, env):
        assert expected == jobutils.Job.default

    @pytest.mark.parametrize('name', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname,expected', [(MB_LMP_LOG, 4)])
    def testData(self, job, expected):
        assert 4 == len(job.data)

    def testGetData(self, raw):
        assert not raw.getData()
        raw.data['outfile'] = 'mol_bldr.log'
        assert not raw.getData()
        raw.write()
        assert 'mol_bldr.log' == raw.getData()['outfile']

    @pytest.mark.parametrize('files', [(['first', 'second'])])
    @pytest.mark.parametrize("ftype", [('outfiles'), ('myfiles')])
    def testAdd(self, raw, ftype, files):
        for file in files:
            raw.add(file, ftype=ftype)
        assert raw.data[ftype] == files

    @pytest.mark.parametrize('file', [('file')])
    @pytest.mark.parametrize("ftype", [('outfile'), ('logfile')])
    def testSet(self, raw, ftype, file):
        raw.set(file, ftype=ftype)
        assert raw.data[ftype] == file

    def testWrite(self, raw):
        raw.set('my.log', ftype='logfile')
        raw.write()
        with open(raw.file) as fh:
            assert raw.data == json.load(fh)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('name', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname, ftype, expected',
                             [(MB_LMP_LOG, 'outfile', 'mb_lmp_log.log'),
                              (MB_LMP_LOG, 'outfile2', None),
                              (None, 'outfile', None)])
    def testGetFile(self, ftype, expected, dirname, job):
        file = job.getFile(ftype=ftype)
        assert file is None if expected is None else file.endswith(expected)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('name', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname, expected',
                             [(MB_LMP_LOG, 'mb_lmp_log.log')])
    def testLogFile(self, expected, dirname, job):
        assert job.logfile.endswith(expected)

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('name', [(None)])
    @pytest.mark.parametrize('dirname, expected',
                             [(MB_LMP_LOG, 'mb_lmp_log.log')])
    def testGetJobs(self, expected, job):
        assert 4 == len([x.file for x in job.getJobs()])

    @pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
    @pytest.mark.parametrize('name', [('mb_lmp_log')])
    @pytest.mark.parametrize('dirname, expected',
                             [(MB_LMP_LOG, 'mb_lmp_log.log')])
    def testClean(self, expected, copied):
        assert os.path.exists(copied.file)
        copied.clean()
        assert not os.path.exists(copied.file)
