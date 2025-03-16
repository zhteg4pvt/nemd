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

    @pytest.mark.parametrize('ekey,cpu_count', [('DEBUG', 8)])
    @pytest.mark.parametrize('original,evalue,expected',
                             [([], '', 6), ([], '1', 1),
                              (['-CPU', '2'], '', 2), (['-CPU', '2'], '1', 2)])
    def testInit(self, original, cpu_count, expected, env):
        options = parserutils.Workflow().parse_args(original)
        with mock.patch('os.cpu_count', return_value=cpu_count):
            runner = jobcontrol.Runner(options=options, original=original)
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
        assert original == runner.original
        assert {} == runner.proj.document['prereq']
