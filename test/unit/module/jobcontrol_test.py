import os
import shutil
import types
from unittest import mock

import pytest

from nemd import jobcontrol
from nemd import jobutils
from nemd import parserutils
from nemd import taskbase


class TestRunner:

    @pytest.fixture
    def runner(self, original, flow_opr, tmp_dir):
        options = parserutils.Workflow().parse_args(original)
        return jobcontrol.Runner(options=options,
                                 original=original,
                                 logger=mock.Mock())

    @pytest.fixture
    def ran(self, runner, Cmd, Job, pre):
        runner.setMaxCpu()
        runner.add(Cmd)
        runner.add(Job, pre=pre)
        runner.setProj()
        runner.openJobs()
        runner.setCpu()
        runner.runProj()
        return runner

    @pytest.mark.parametrize('cpu_count', [8])
    @pytest.mark.parametrize('original,expected',
                             [(['-DEBUG', 'off'], 6), (['-DEBUG', 'on'], 1),
                              (['-DEBUG', 'off', '-CPU', '2'], 2),
                              (['-DEBUG', 'on', '-CPU', '2'], 2)])
    def testSetMaxCpu(self, original, cpu_count, runner, expected):
        with mock.patch('os.cpu_count', return_value=cpu_count):
            runner.setMaxCpu()
        assert expected == runner.max_cpu

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize('agg,pre,expected,num',
                             [(False, None, ['job'], 1),
                              (False, False, None, 0), (True, None, None, 0)])
    def testAdd(self, agg, pre, runner, expected, num, flow_opr):
        job = taskbase.Agg if agg else taskbase.Job
        runner.add(job)
        assert 1 == len(runner.added)
        assert 0 == len(runner.prereq)
        runner.add(job, jobname='job2', pre=pre)
        assert 2 == len(runner.added)
        assert expected == runner.prereq.get('job2')
        assert num == len(flow_opr._OPERATION_PRECONDITIONS)

    @pytest.mark.parametrize('original', [[]])
    def testSetProj(self, original, runner):
        runner.setProj()
        assert runner.proj is not None
        assert 'args' in runner.proj.document
        assert 'prereq' in runner.proj.document

    @pytest.mark.parametrize('original,expected',
                             [(['-JOBNAME', 'myname', '-DEBUG', 'off'], False),
                              (['-JOBNAME', 'myname', '-DEBUG', 'on'], True)])
    def testPlotJobs(self, runner, expected):
        runner.add(taskbase.Job)
        runner.add(taskbase.Job, jobname='job2')
        runner.setProj()
        runner.plotJobs()
        assert expected == os.path.isfile('myname_nx.png')

    @pytest.mark.parametrize('kwargs,expected',
                             [({}, None), (dict(state_num=1), ['1']),
                              (dict(state_num=3), ['1', '2', '3'])])
    def testSetState(self, kwargs, expected, tmp_dir):
        runner = jobcontrol.Runner(types.SimpleNamespace(**kwargs), [])
        runner.setProj()
        runner.setState()
        assert expected == runner.state.get('-seed')

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize(
        'state', [dict(seed=['1', '2'], scale_factor=['0.95', '1'])])
    def testOpenJobs(self, state, runner):
        runner.state = state
        runner.setProj()
        runner.openJobs()
        assert 4 == len(runner.jobs)

    @pytest.mark.parametrize('max_cpu,jobs', [(12, [None] * 4)])
    @pytest.mark.parametrize('original,expected',
                             [([], [12, 1]), (['-CPU', '3'], [3, 1]),
                              (['-CPU', '12'], [4, 3]),
                              (['-CPU', '7', '3'], [2, 3])])
    def testSetCpu(self, max_cpu, jobs, runner, expected, tmp_dir):
        runner.max_cpu = max_cpu
        runner.jobs = jobs
        runner.setProj()
        runner.setCpu()
        assert expected == runner.cpu

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
        assert os.path.isfile('name_status.log')
        num = int(file and status)
        calls = ran.logger.log.call_args_list
        assert mock.call(f'{num} / 1 completed jobs.') == calls[0]
        if num:
            assert 1 == len(calls)
            return
        oprs = calls[1][0][0].split('|')[-2]
        assert not file == ('cmd' in oprs)
        assert not status == ('job' in oprs)

    @pytest.mark.parametrize('original', [[]])
    def testSetAgg(self, runner, flow_opr):
        runner.setAggs()
        assert 'time_agg' == flow_opr._OPERATION_FUNCTIONS[0][0]

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-DEBUG'], True, True, None)])
    def testSetAggProj(self, ran):
        ran.setAggProj()
        assert 1 == len(ran.jobs)
        ran.jobs = []
        ran.setAggProj()
        assert 1 == len(ran.jobs)
        shutil.rmtree(ran.proj.workspace)
        with pytest.raises(SystemExit):
            ran.setAggProj()

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-clean', '-DEBUG'], True, True, None)])
    def testCleanAgg(self, ran):
        ran.add(taskbase.Agg)
        ran.setAggProj()
        ran.runProj(agg=True)
        file = jobutils.Job('agg', dirname=ran.jobs[0].project.fn('')).file
        assert os.path.exists(file)
        ran.clean(agg=True)
        assert not os.path.exists(file)

    @pytest.mark.parametrize('original,file,status,pre',
                             [(['-DEBUG'], True, True, None)])
    def testRunProjAgg(self, ran):
        ran.add(taskbase.Agg)
        ran.setAggProj()
        ran.runProj(agg=True)
        job = jobutils.Job('agg', dirname=ran.jobs[0].project.fn(''))
        assert job['status']
