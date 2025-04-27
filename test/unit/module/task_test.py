from unittest import mock

import pytest

from nemd import task


class TestLmpLog:

    @pytest.fixture
    def lmp_log(self, jobs):
        return task.LmpLog(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('dirname,expected',
                             [('0046_test', 'mol_bldr.data')])
    def testAddfiles(self, lmp_log, expected):
        lmp_log.addfiles()
        assert expected == lmp_log.args[-1]


class TestLmpTraj:

    @pytest.fixture
    def lmp_traj(self, jobs):
        return task.LmpTraj(*jobs, status={}, logger=mock.Mock())

    @pytest.mark.parametrize('dirname,expected',
                             [('0045_test', 'amorp_bldr.custom.gz')])
    def testAddfiles(self, lmp_traj, expected):
        lmp_traj.addfiles()
        assert expected == lmp_traj.args[0]
