# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
numba utilities.
"""
import numba
import numpy as np

from nemd import envutils

NOPYTHON = envutils.NOPYTHON
JIT = envutils.get_jit_kwargs()
NOPYTHON = envutils.is_nopython()


def jit(*args, **kwargs):
    """
    Decorate a function using numba.jit.

    See 'Writing a decorator that works with and without parameter' in
    https://stackoverflow.com/questions/5929107/decorators-with-parameters

    :return 'function': decorated function or the decorator function.
    """
    # direct=True when directly applied as:
    # @jit
    # def foo():
    direct = bool(args and callable(args[0]))

    def _decorator(func):
        """
        Decorate a function using numba.jit.

        :param func 'function': the function to be decorated.
        :return 'function': the (decorated) function.
        """
        return numba.jit(func, **JIT, **kwargs) if JIT[NOPYTHON] else func

    return _decorator(args[0]) if direct else _decorator


@jit
def norm(vecs, span):
    """
    Calculate IEEE 754 remainder (pbc distance) and norm (vector length).

    https://stackoverflow.com/questions/26671975/why-do-we-need-ieee-754-remainder

    :param vecs vectors nx3 numpy.ndarray: each sublist is a vector
    :param span numpy.ndarray: box span lengths
    :return list of floats: distances within half box span
    """
    vecs -= np.round(np.divide(vecs, span)) * span
    return [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in vecs]


@jit
def getIds(xyzs, grids, dims, nopython=NOPYTHON):
    """
    Get the cell id for xyz.

    :param grids 1x3 'numpy.ndarray': the grid lengths
    :param dims list of 'numba.int32': the grid numbers
    :param nopython bool: whether numba nopython mode is on
    :return list of ints: the cell id
    """
    int32 = numba.int32 if nopython else np.int32
    return np.round(xyzs / grids).astype(int32) % dims
