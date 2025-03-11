import contextlib
import functools
import inspect
from inspect import isclass

import pytest


class Raises:
    """
    The class to handle raises.
    """

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

            :param expected 'type' or any: The raised exception class (e.g. ValueError),
                or the expected return instance
            :return any: the function return
            """
            with cls.ctxmgr(expected):
                return func(*args, expected=expected, **kwargs)

        return wrapped

    @classmethod
    def raises(cls, obj):
        """
        Class and method decorator.

        :param obj `class`, `method` or `func`: the object to decorate
        :return obj: the decorated object.
        """
        if not isclass(obj):
            return cls.decorate(obj)
        for name, method in obj.__dict__.items():
            if name.startswith("test") and callable(method):
                setattr(obj, name, cls.decorate(method))
        return obj
