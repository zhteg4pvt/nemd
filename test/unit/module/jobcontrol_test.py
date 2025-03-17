import glob
import os
import types
from unittest import mock

import pytest

from nemd import jobcontrol
from nemd import parserutils
from nemd import taskbase


class TestRunner:

    @pytest.fixture
    def runner(self, original, tmp_dir):
        options = parserutils.Workflow().parse_args(original)
        return jobcontrol.Runner(options=options,
                                 original=original,
                                 logger=mock.Mock())

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
    @pytest.mark.parametrize('agg,pre,expected', [(False, None, ['job']),
                                                  (False, False, None),
                                                  (True, None, None)])
    def testAdd(self, agg, pre, runner, expected, flow_opr):
        job = taskbase.Agg if agg else taskbase.Job
        runner.add(job)
        assert 1 == len(runner.added)
        assert 0 == len(runner.prereq)
        runner.add(job, jobname='job2', pre=pre)
        assert 2 == len(runner.added)
        assert expected == runner.prereq.get('job2')

    @pytest.mark.parametrize('original', [[]])
    def testSetProj(self, original, runner):
        runner.setProj()
        assert runner.proj is not None
        assert 'args' in runner.proj.document
        assert 'prereq' in runner.proj.document

    @pytest.mark.parametrize('original,expected',
                             [(['-JOBNAME', 'myname', '-DEBUG', 'off'], False),
                              (['-JOBNAME', 'myname', '-DEBUG', 'on'], True)])
    def testPlotJobs(self, runner, expected, flow_opr):
        runner.add(taskbase.Job)
        runner.add(taskbase.Job, jobname='job2')
        runner.setProj()
        runner.plotJobs()
        assert expected == os.path.isfile('myname_nx.png')

    @pytest.mark.parametrize('kwargs,expected',
                             [({}, None), (dict(state_num=1), ['1']),
                              (dict(state_num=3), ['1', '2', '3'])])
    def testSetState(self, kwargs, expected):
        runner = jobcontrol.Runner(types.SimpleNamespace(**kwargs), [])
        runner.setState()
        assert expected == runner.state.get('-seed')

    @pytest.mark.parametrize('original', [[]])
    @pytest.mark.parametrize(
        'state', [dict(seed=['1', '2'], scale_factor=['0.95', '1'])])
    def testOpenJobs(self, state, runner, tmp_dir):
        runner.state = state
        runner.setProj()
        runner.openJobs()
        assert 4 == len(runner.jobs)

    @pytest.mark.parametrize('max_cpu,jobs', [(12, [None] * 4)])
    @pytest.mark.parametrize('original,expected',
                             [([], [12, 1]), (['-CPU', '3'], [3, 1]),
                              (['-CPU', '12'], [4, 3]),
                              (['-CPU', '7', '3'], [2, 3])])
    def testSetCpu(self, max_cpu, jobs, runner, expected):
        runner.max_cpu = max_cpu
        runner.jobs = jobs
        runner.setCpu()
        assert expected == runner.cpu

    @pytest.mark.parametrize('ekey', ['DEBUG'])
    @pytest.mark.parametrize('original,evalue', [(['-DEBUG'], '')])
    def testRunProj(self, runner, env, flow_opr):
        cmd = "'from nemd import jobutils; jobutils.add_outfile(1, file=True)'"
        runner.args = ['-c', cmd]
        runner.setMaxCpu()
        runner.add(taskbase.Cmd)
        runner.add(taskbase.Job)
        runner.setProj()
        runner.openJobs()
        runner.setCpu()
        runner.runProj()
        assert 1 == len(glob.glob('workspace/*/.cmd_document.json'))
        assert 1 == len(glob.glob('workspace/*/.job_document.json'))