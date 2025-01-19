# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
numba utilities.
"""
import numba
import numpy as np

from nemd import envutils


def jit(*args, **kwargs):
    """
    Decorate a function using numba.jit.

    See 'Writing a decorator that works with and without parameter' in
    https://stackoverflow.com/questions/5929107/decorators-with-parameters

    :return 'function': decorated function or the decorator function.
    """
    # decorator directly applies instead of being called with parameters as:
    # @jit
    # def foo():
    direct = bool(args and callable(args[0]))

    def _decorator(func):
        """
        Decorate a function using numba.jit.

        :param func 'function': the function to be decorated.
        :return 'function': the (decorated) function.
        """
        jwargs = envutils.get_jit_kwargs(**kwargs)
        return numba.jit(func, **jwargs) if jwargs[envutils.NOPYTHON] else func

    return _decorator(args[0]) if direct else _decorator


@jit
def remainder(dists, span):
    """
    Calculate IEEE 754 remainder.

    https://stackoverflow.com/questions/26671975/why-do-we-need-ieee-754-remainder

    :param dists numpy.ndarray: distances
    :param span numpy.ndarray: box span
    :return list of floats: distances within half box span
    """
    dists -= np.round(np.divide(dists, span)) * span
    return [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in dists]
