import json
import os

import pytest

from nemd import envutils
from nemd import jobutils

NEMD_SRC = envutils.get_src()


class TestArgs:

    @pytest.fixture
    def args(self, cmd):
        return jobutils.Args(cmd)

    @pytest.mark.parametrize(
        'cmd,first,all_args',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], '1', ['1', '2']),
         ([], None, None)])
    def testGet(self, args, first, all_args):
        assert first == args.get('-cru_num')
        val = first if first else 'val'
        assert val == args.get('-cru_num', default='val')
        assert all_args == args.get('-cru_num', first=False)
        vals = all_args if all_args else ['val']
        assert vals == args.get('-cru_num', default=['val'], first=False)

    @pytest.mark.parametrize(
        'cmd,popped,num',
        [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'], ['1', '2'], 2),
         ([], None, 0), (['[Ar]', '-cru_num', '2', '-DEBUG'], ['2'], 2)])
    def testRm(self, args, popped, num):
        assert popped == args.rm('-cru_num')
        assert num == len(args)

    @pytest.mark.parametrize('cmd,expected',
                             [(['[Ar]', '-cru_num', '1', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '2', '-DEBUG']),
                              ([], ['-cru_num', '5']),
                              (['[Ar]', '-cru_num', '2', '-DEBUG'
                                ], ['[Ar]', '-cru_num', '5', '-DEBUG'])])
    def testSet(self, args, expected):
        assert expected == args.set('-cru_num', '5')


@pytest.mark.skipif(NEMD_SRC is None, reason="cannot locate test dir")
class TestJob:

    @pytest.fixture
    def job(self, jobname, dirname):
        if dirname:
            dirname = envutils.test_data(dirname)
        return jobutils.Job(jobname=jobname, dirname=dirname)

    @pytest.mark.parametrize("ekey", ['JOBNAME'])
    @pytest.mark.parametrize("jobname,dirname,evalue,expected",
                             [('myname', None, 'envname', 'myname'),
                              (None, None, 'envname', 'envname'),
                              (None, None, None, 'job'),
                              ('mb_lmp_log', '0046_test', None, 'mb_lmp_log')])
    def testInit(self, jobname, dirname, expected, env):
        job = jobutils.Job(jobname=jobname, dirname=dirname)
        assert [expected, 3] == [job.jobname, len(job)]

    @pytest.mark.parametrize('dirname', ['0046_test'])
    @pytest.mark.parametrize('jobname,expected', [('mb_lmp_log', True),
                                                  ('mb_lmp_log2', False)])
    def testFile(self, job, expected):
        assert expected == os.path.isfile(job.file)

    @pytest.mark.parametrize('jobname', ['jobname'])
    @pytest.mark.parametrize(
        'dirname,filename,expected',
        [(NEMD_SRC, 'myfile', 'myfile'), (None, 'filename', 'filename'),
         (None, os.path.join(os.getcwd(), 'name'), 'name')])
    def testFn(self, job, filename, expected):
        dirname, filename = os.path.split(job.fn(filename))
        assert expected == filename
        assert os.path.isdir(dirname)

    @pytest.mark.parametrize('dirname', ['0046_test'])
    @pytest.mark.parametrize('jobname,expected',
                             [('mb_lmp_log', 'mb_lmp_log.log'),
                              ('mb_lmp_log2', None)])
    def testLogFile(self, job, expected):
        if not expected:
            assert job.outfile is None
            return
        assert expected == os.path.basename(job.outfile)
        assert os.path.isfile(job.outfile)

    @pytest.mark.parametrize('dirname', [
        os.path.join('0049_test', 'workspace',
                     '3ec5394f589c9363bd15af35d45a7c44')
    ])
    @pytest.mark.parametrize(
        'jobname,expected',
        [('number_of_molecules_100', 'number_of_molecules_100.in'),
         ('mb_lmp_log', None)])
    def testOutFile(self, job, expected):
        if not expected:
            assert job.outfile is None
            return
        assert expected == os.path.basename(job.outfile)
        assert os.path.isfile(job.outfile)

    def testWrite(self, tmp_dir):
        job = jobutils.Job(jobname=('jobname'))
        job._logfile = 'mylog'
        job._outfile = 'myout'
        job._outfiles.append('outfile1')
        job._outfiles.append('outfile2')
        job.write()
        with open(job.file) as fh:
            assert job == json.load(fh)

    def testClean(self, tmp_dir):
        job = jobutils.Job(jobname='jobname')
        job.write()
        assert os.path.exists(job.file)
        job.clean()
        assert not os.path.exists(job.file)

    @pytest.mark.parametrize('jobname,dirname,expected', [(None, '0000', 4)])
    def testSearch(self, expected, job):
        assert expected == len(jobutils.Job.search(job.dirname))

    @pytest.mark.parametrize('jobname,dirname,expected',
                             [('mb_lmp_log', '0000', 4)])
    def testFromFile(self, jobname, dirname, expected, copied):
        assert expected == len(jobutils.Job.search(copied))
        assert expected == len(jobutils.Job.search())

    @pytest.mark.parametrize('ekey,evalue', [('JOBNAME', 'name')])
    @pytest.mark.parametrize('jobname', [None, 'jobname'])
    @pytest.mark.parametrize('file', [False, True])
    @pytest.mark.parametrize('log', [False, True])
    def testReg(self, jobname, file, log, tmp_dir, env):
        jobutils.Job.reg('file', jobname=jobname, file=file, log=log)
        json_file = f".{jobname if jobname else 'name'}_document.json"
        with open(json_file) as fh:
            data = json.load(fh)
            assert ['file'] == data['_outfiles']
            assert ('file' if file else None) == data.get('_outfile')
            assert ('file' if log else None) == data.get('_logfile')
