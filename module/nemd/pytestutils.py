# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Pytest utilities.
"""
import contextlib
import functools
import inspect

import pytest


class Raises:
    """
    The class to handle raises.
    """

    def __new__(cls, obj):
        """
        Class and method decorator.

        :param obj `class`, `method` or `func`: the object to decorate
        :return obj: the decorated object.
        """
        if not inspect.isclass(obj):
            return cls.decorate(obj)
        for name, method in obj.__dict__.items():
            if name.startswith("test") and callable(method):
                setattr(obj, name, cls.decorate(method))
        return obj

    @classmethod
    def decorate(cls, func):
        """
        Decorate a function so that the expected exception can be asserted.

        :return 'function': decorated function or the decorator function.
        """

        @functools.wraps(func)
        def wrapped(*args, expected=None, **kwargs):
            """
            Run function with the expected exception catched.

            :param expected 'type' or any: The raised exception class
                (e.g. ValueError), or the expected return instance
            :return any: the function return
            """
            with cls.ctxmgr(expected):
                return func(*args, expected=expected, **kwargs)

        return wrapped

    @staticmethod
    def ctxmgr(expected):
        """
        Return function to open context management for the exception assertion.

        :param request '_pytest.fixtures.SubRequest': The requested information.
        :param expected 'type' or any: The raised exception class (e.g. ValueError),
            or the expected return instance
        :return 'ContextManager': context manager to assert the raise
        """
        if inspect.isclass(expected) and issubclass(expected, BaseException):
            return pytest.raises(expected)
        return contextlib.nullcontext()
