# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Test configuration, such as global testing fixtures...
"""
import contextlib
import functools
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


@contextlib.contextmanager
def assert_raises(is_raise, raise_type):
    """
    Assert that the exception is raised or not raised according to the input.

    :param is_raise: when True, the exception is expected to be raised.
    :type is_raise: bool
    :param raise_type: The exception type.
    :type raise_type: 'type' (e.g. ValueError, TypeError, etc.)
    """
    if is_raise:
        with pytest.raises(raise_type):
            yield
        return

    try:
        yield
    except raise_type as err:
        assert False, f"{err} should not be raised."


@pytest.fixture
def check_raise(request, is_raise, raise_type):
    """
    Assert that the exception is raised or not raised according to the input.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param is_raise: when True, the exception is expected to be raised.
    :type is_raise: bool
    :param raise_type: The exception type.
    :type raise_type: 'type' (e.g. ValueError, TypeError, etc.)
    :return: function to check if the expected exception is raised.
    :rtype: 'function'
    """
    if not raise_type:
        raise_type = Exception
    return functools.partial(assert_raises, is_raise, raise_type)
