# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Test configuration, such as global testing fixtures...
"""
import collections
import os
import shutil
from unittest import mock

import flow
import pytest

from nemd import envutils
from nemd import frame
from nemd import jobutils
from nemd import osutils
from nemd import pytestutils
from nemd import taskbase


@pytest.fixture
def tmp_dir(request, tmpdir):
    """
    Create a temporary directory and change to it for the duration of the test.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param tmpdir '_pytest._py.path.LocalPath': The temporary directory factory.
    :return tmpdir '_pytest._py.path.LocalPath': The temporary directory.
    """
    with osutils.chdir(tmpdir, rmtree=True):
        yield tmpdir


@pytest.fixture
def env(ekey, evalue):
    """
    Temporarily set environment.

    :param ekey str: The environmental keyword.
    :param evalue str: the environmental value.
    :return environ dict: the environment.
    """
    environ = {} if evalue is None else {ekey: evalue}
    with mock.patch.dict('os.environ', environ, clear=True):
        yield environ


@pytest.fixture
def raises(request, expected):
    """
    Return function to open context management for the exception assertion.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param expected 'type' or any: The raised exception class (e.g. ValueError),
        or the expected return instance
    :return 'ContextManager': context manager to assert the raise
    """
    # FIXME: leaks exceptions when fixture catches error via contextmanager
    # with pytest.raises(expected):
    #     yield
    return pytestutils.Raises.ctxmgr(expected)


@pytest.fixture
def flow_opr():
    """
    Yield FlowProject with restorable _OPERATION_*.

    :return `flow.FlowProject`: the patched flow.FlowProject
    """
    project = flow.FlowProject
    functions = project._OPERATION_FUNCTIONS
    preconditions = project._OPERATION_PRECONDITIONS
    postconditions = project._OPERATION_POSTCONDITIONS
    project._OPERATION_FUNCTIONS = []
    project._OPERATION_PRECONDITIONS = collections.defaultdict(list)
    project._OPERATION_POSTCONDITIONS = collections.defaultdict(list)
    yield project
    project._OPERATION_FUNCTIONS = functions
    project._OPERATION_PRECONDITIONS = preconditions
    project._OPERATION_POSTCONDITIONS = postconditions


@pytest.fixture
def copied(dirname, tmp_dir):
    """
    Locate and copy an itest.

    :return `str`: the dirname to the original itest
    """
    if dirname is None:
        return
    test_dir = envutils.test_data(dirname)
    shutil.copytree(test_dir, os.curdir, dirs_exist_ok=True)
    return test_dir


@pytest.fixture
def jobs(dirname, copied):
    """
    Return the signac jobs of a copied itest.

    :return `list`: signac jobs
    """
    try:
        project = flow.project.FlowProject.get_project(os.curdir)
    except LookupError:
        return []
    else:
        return list(project.find_jobs())


@pytest.fixture
def job(jobname, copied):
    """
    Return the job of a copied itest.

    :return `jobutils.Job`: Job loaded from a job json file
    """
    return jobutils.Job(jobname)


@pytest.fixture
def Cmd(file):
    """
    Return a simple Cmd class that runs under jobcontrol.

    :return `taskbase.Cmd`: the Cmd class
    """

    class Cmd(taskbase.Cmd):
        FILE = (f"-c 'from nemd import jobutils;"
                f"jobutils.Job.reg(jobutils.JOB, file={file})'")

    return Cmd


@pytest.fixture
def Job(status):
    """
    Return a simple non-cmd class that runs under jobcontrol.

    :return `taskbase.Job`: the Cmd class
    """

    class Job(taskbase.Job):

        def run(self, *args, **kwargs):
            self.out = status

    return Job


@pytest.fixture
def frm(file):
    """
    Return a trajectory frame.

    :param file str: the trajectory file
    :return `frame.Frame`: loaded frame
    """
    with open(file, 'r') as fh:
        return frame.Frame.read(fh)
