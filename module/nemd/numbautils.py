# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
numba utilities.
"""
import numba
import numpy as np

from nemd import envutils

IS_NOPYTHON = envutils.is_nopython()
JIT_KWARGS = envutils.get_jit_kwargs()


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
        return numba.jit(func, **JIT_KWARGS, **kwargs) if IS_NOPYTHON else func

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
    shift = np.round(np.divide(vecs, span)) * span
    return [np.linalg.norm(x) for x in vecs - shift]


@jit
def get_ids(grids, dims, xyzs, gid):
    """
    Get the cell ids from xyz coordinates.

    :param grids 1x3 'numpy.ndarray': the grid sizes
    :param dims 1x3 'numpy.ndarray': the grid numbers
    :param xyzs nx3 'numpy.ndarray': the coordinates
    :param gid int: the global atom id
    :return list of ints: the cell id
    """
    return np.round(xyzs[gid] / grids).astype(np.int32) % dims


@jit
def set(cell, grids, dims, xyzs, gids, state=True):
    """
    Get the cell ids from xyz coordinates.

    :param cell nxnxnxm 'numpy.ndarray': the distance cell
    :param grids 1x3 'numpy.ndarray': the grid sizes
    :param dims 1x3 'numpy.ndarray': the grid numbers
    :param xyzs nx3 'numpy.ndarray': the coordinates
    :param gids 'set': the corresponding global atom ids
    :param state 'bool': the state to set
    :return list of ints: the cell id
    """
    for gid in gids:
        ids = get_ids(grids, dims, xyzs, gid)
        cell[ids[0], ids[1], ids[2], gid] = state


@jit
def get(cell, grids, dims, xyzs, nbrs, gid, gt=False):
    """
    Get the neighbor atom ids from the neighbor cells (including the current
    cell itself) via Numba.

    :param cell nxnxnxm 'numpy.ndarray': the distance cell
    :param grids 1x3 'numpy.ndarray': the grid sizes
    :param dims 1x3 'numpy.ndarray': the grid numbers
    :param xyzs nx3 'numpy.ndarray': the coordinates
    :param nbr ixjxkxnx3 'numpy.ndarray': map from cell id to neighbor ids
    :param gid int: the global atom id
    :param gt bool: only include the global atom ids greater than the gid
    :return list of int: the atom ids of the neighbor atoms
    """
    cid = get_ids(grids, dims, xyzs, gid)
    cids = nbrs[cid[0], cid[1], cid[2], :]
    # The atom ids from all neighbor cells
    gids = [cell[x[0], x[1], x[2], :].nonzero()[0] for x in cids]
    if gt:
        gids = [x[x > gid] for x in gids]
    return [y for x in gids for y in x]


@jit
def get_nbrs(dims, orig):
    """
    Get map from any cell id to neighbor ids.

    :param dims numpy.ndarray: the number of cells in three dimensions
    :param orig numpy.ndarray: Neighbors cells of the (0,0,0) cell
    :return numpy.ndarray: map from any cell id to neighbor ids
    """
    nbrs = np.zeros((dims[0], dims[1], dims[2], *orig.shape), dtype=np.int32)
    for ix in numba.prange(dims[0]):
        for iy in numba.prange(dims[1]):
            for iz in numba.prange(dims[2]):
                nbrs[ix, iy, iz, :, :] = (orig + np.array([ix, iy, iz])) % dims
    return nbrs
