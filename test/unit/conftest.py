# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Test configuration, such as global testing fixtures...
"""
import contextlib
import functools
import inspect
from unittest import mock

import pytest

from nemd import osutils


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
    Return function to open context management to assert the exception.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param expected 'type' or any: The raised exception class (e.g. ValueError),
        or the expected return instance
    :return 'ContextManager': context manager to assert the raise
    """
    # FIXME: leaks exceptions when fixture catches error via contextmanager
    # with pytest.raises(expected):
    #     yield
    if inspect.isclass(expected) and issubclass(expected, BaseException):
        return pytest.raises(expected)
    return contextlib.nullcontext()
