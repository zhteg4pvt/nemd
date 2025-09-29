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
    # direct=True when directly applied as:
    # @jit
    # def foo():
    direct = bool(args and callable(args[0]))

    def _decorator(func, mode=envutils.Mode()):
        """
        Decorate a function using numba.jit.

        :param func 'function': the function to be decorated.
        :param python 'envutils.Mode': the python mode.
        :return 'function': the (decorated) function.
        """
        return numba.jit(func, **mode.kwargs, **kwargs) if mode.no else func

    return _decorator(args[0]) if direct else _decorator


@jit
def norms(vecs, span):
    """
    Calculate IEEE 754 remainder (pbc distance) and norm (vector length).

    https://stackoverflow.com/questions/26671975/why-do-we-need-ieee-754-remainder

    :param vecs vectors nx3 numpy.ndarray: each sublist is a vector
    :param span numpy.ndarray: box span lengths
    :return np.ndarray: distances within half box span
    """
    shift = np.round(np.divide(vecs, span)) * span
    return np.array([np.linalg.norm(x) for x in vecs - shift])


@jit
def msd(trj, gids, wts):
    """
    Get the iterator of mean squared displacement.

    :param trj np.ndarray: the trajectory.
    :param gids np.ndarray: the selected global atom ids.
    :param wts np.ndarray: the weight of each atom.
    :return np.ndarray: mean squared displacement of each tau.
    """
    wts = wts.astype(trj.dtype)
    wts /= wts.sum()
    for frm in trj:
        frm -= np.dot(wts, np.ascontiguousarray(frm))
    wts = wts[gids]
    wts /= wts.sum()
    num = len(trj)
    data = np.zeros(num, dtype=trj.dtype)
    for idx in range(num):
        sq = np.square(trj[idx:, gids, :] - trj[:num - idx, gids, :])
        data[idx] = np.dot(wts, sq.sum(axis=2).sum(axis=0) / (num - idx))
    return data
