import os
import types
from unittest import mock

import numpy as np
import pytest

from nemd import jobcontrol
from nemd import jobutils
from nemd import parserutils
from nemd import taskbase


class TestRunner:

    @pytest.fixture
    def runner(self, original, logger, flow_opr, tmp_dir):
        options = parserutils.Workflow().parse_args(original)
        return jobcontrol.Runner(options=options,
                                 original=original,
                                 logger=logger)

    @pytest.fixture
    def ran(self, runner, Cmd, Job, pre):
        runner.add(Cmd)
        runner.add(Job, pre=pre)
        runner.setProj()
        runner.openJobs()
        runner.runJobs()
        return runner

    @pytest.fixture
    def agg(self, ran):
        ran.add(taskbase.Agg)
        ran.setAggProj()
        ran.runProj(agg=True)
        return ran

    @pytest.fixture
    def Cmd(self, file):
        """
        Return a simple Cmd class that runs under jobcontrol.

        :return `taskbase.Cmd`: the Cmd class
        """

        class Cmd(taskbase.Cmd):
            FILE = (f"-c 'from nemd import jobutils;"
                    f"jobutils.Job.reg(jobutils.JOB, file={file})'")

        return Cmd

    @pytest.fixture
    def Job(self, status):
        """
        Return a simple non-cmd class that runs under jobcontrol.

        :return `taskbase.Job`: the Cmd class
        """

        class Job(taskbase.Job):

            def run(self, *args, **kwargs):
                self.out = status

        return Job

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize(
        'agg,pre,expected',
        [(False, None, [1, 0, ['job'], 'job', 'job2', 1, 2]),
         (False, False, [1, 0, None, 'job', 'job2', 0, 2]),
         (True, None, [1, 0, None, 'agg', 'job2', 0, 2])])
    def testAdd(self, agg, pre, runner, expected, check_flow):
        job = taskbase.Agg if agg else taskbase.Job
        runner.add(job)
        assert expected[:2] == [len(runner.oprs), len(runner.prereq)]
        runner.add(job, jobname='job2', pre=pre)
        assert expected[2] == runner.prereq.get('job2')

    @pytest.mark.parametrize('original,expected',
                             [(['-JOBNAME', 'myname', '-DEBUG', 'off'], False),
                              (['-JOBNAME', 'myname', '-DEBUG', 'on'], True)])
    def testSetProj(self, runner, expected):
        runner.add(taskbase.Job)
        runner.add(taskbase.Job, jobname='job2')
        runner.setProj()
        assert runner.proj is not None
        assert expected == os.path.isfile('myname_nx.png')

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize(
        'state', [dict(seed=['1', '2'], scale_factor=['0.95', '1'])])
    def testOpenJobs(self, state, runner):
        runner.state = state
        runner.setProj()
        runner.openJobs()
        assert 4 == len(runner.jobs)

    @pytest.mark.parametrize('kwargs,expected',
                             [({}, None), (dict(state_num=1), [1]),
                              (dict(state_num=3), [1, 2, 3])])
    def testState(self, kwargs, expected):
        runner = jobcontrol.Runner(types.SimpleNamespace(**kwargs), [])
        np.testing.assert_equal(runner.state.get('-seed'), expected)

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-clean', '-DEBUG'], True, True, None)])
    def testRunJobs(self, ran):
        assert 2 == len(ran.options.CPU)
        assert 'args' in ran.proj.document
        assert 'prereq' in ran.proj.document
        assert 2 == len([y for x in ran.status.values() for y in x.values()])

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-clean', '-DEBUG'], True, True, None)])
    def testClean(self, ran):
        dirname = ran.jobs[0].fn('')
        files = [jobutils.Job(x, dirname=dirname).file for x in ['cmd', 'job']]
        assert all([os.path.exists(x) for x in files])
        ran.clean()
        assert not any([os.path.exists(x) for x in files])

    @pytest.mark.parametrize('original', [(['-DEBUG'])])
    @pytest.mark.parametrize('file', [True, False])
    @pytest.mark.parametrize('status', [True, False])
    @pytest.mark.parametrize('pre', [(None), (False)])
    def testRunProj(self, pre, file, status, ran):
        dirname = ran.jobs[0].fn('')
        outfile = jobutils.Job('cmd', dirname=dirname).outfile
        assert outfile.endswith('job') if outfile else (outfile is None)
        stat = jobutils.Job('job', dirname=dirname).get('status')
        assert (status if pre is False or file else None) == stat

    @pytest.mark.parametrize('original', [(['-JOBNAME', 'name', '-DEBUG'])])
    @pytest.mark.parametrize('file', [True, False])
    @pytest.mark.parametrize('status', [True, False])
    @pytest.mark.parametrize('pre', [False])
    def testLogStatus(self, file, status, ran):
        ran.logStatus()
        num = int(file and status)
        calls = ran.logger.log.call_args_list
        assert mock.call(f'{num} / 1 completed jobs.') == calls[0]
        if num:
            assert 1 == len(calls)
            return
        oprs = calls[1][0][0].split('|')[-2]
        assert not file == ('cmd' in oprs)
        assert not status == ('job' in oprs)

    @pytest.mark.parametrize('original,expected', [([], ['time_agg', 0, 1])])
    def testSetAgg(self, runner, check_flow):
        runner.setAggs()

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-DEBUG'], True, True, None)])
    def testSetAggProj(self, ran):
        ran.setAggProj()
        assert ran.proj

    @pytest.mark.parametrize('original', [(['-DEBUG'])])
    def testSetAggProjClean(self, runner):
        with pytest.raises(SystemExit):
            runner.setAggProj()

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-DEBUG'], True, True, None)])
    @pytest.mark.parametrize('clear,expected', [
        (False, 1),
        (True, 1),
    ])
    def testFindJobs(self, ran, clear, expected, raises):
        if clear:
            ran.jobs = []
        ran.setAggProj()
        ran.findJobs()
        assert expected == len(ran.jobs)

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-clean', '-DEBUG'], True, True, None)])
    def testCleanAgg(self, agg):
        file = jobutils.Job('agg', dirname=agg.proj.fn('')).file
        assert os.path.exists(file)
        agg.clean(agg=True)
        assert not os.path.exists(file)

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-DEBUG'], True, True, None)])
    def testRunProj(self, agg):
        assert jobutils.Job('agg', dirname=agg.proj.fn(''))['status']
