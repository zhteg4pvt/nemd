import contextlib
import functools
import inspect

import pytest


def get_raises(expected):
    """
    Return function to open context management to assert the exception.

    :param request '_pytest.fixtures.SubRequest': The requested information.
    :param expected 'type' or any: The raised exception class (e.g. ValueError),
        or the expected return instance
    :return 'ContextManager': context manager to assert the raise
    """
    if inspect.isclass(expected) and issubclass(expected, BaseException):
        return pytest.raises(expected)
    return contextlib.nullcontext()


def raises(func):
    """
    Decorate a function so that the expected exception can be catched.

    :return 'function': decorated function or the decorator function.
    """

    @functools.wraps(func)
    def wrapped(*args, expected=None, **kwargs):
        """
        Run function with the expected exception catched.

        :param expected 'type' or any: The raised exception class (e.g. ValueError),
            or the expected return instance
        :return any: the return of the function
        """
        with get_raises(expected):
            return func(*args, expected=expected, **kwargs)

    return wrapped
