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
def get_ids(grids, dims, xyzs):
    """
    Get the cell ids from xyz coordinates.

    :param grids 1x3 'numpy.ndarray': the grid sizes
    :param dims 1x3 'numpy.ndarray': the grid numbers
    :param xyzs nx3 'numpy.ndarray': the coordinates
    :return list of ints: the cell id
    """
    return np.round(xyzs / grids).astype(np.int32) % dims


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
    for ids, aid in zip(get_ids(grids, dims, xyzs), gids):
        cell[ids[0], ids[1], ids[2], aid] = state


@jit
def get_atoms(cid, nbr, cell):
    """
    Get the neighbor atom ids from the neighbor cells (including the current
    cell itself) via Numba.

    :param idx 1x3 'numpy.ndarray': the distance cell id
    :param nbr ixjxkxnx3 'numpy.ndarray': distance cell id to neighbor cell ids
    :param cell nxnxnxm 'numpy.ndarray': the distance cell
    :return list of int: the atom ids of the neighbor atoms
    """
    ids = nbr[cid[0], cid[1], cid[2], :]
    # The atom ids from all neighbor cells
    return [y for x in ids for y in cell[x[0], x[1], x[2], :].nonzero()[0]]


@jit
def get_nbr(nbr_inc, dims):
    """
    Get map between node id to neighbor node ids.

    :param nbr_inc numpy.ndarray: Neighbors cells sitting on the (0,0,0) cell
    :param dims numpy.ndarray: the number of cells in three dimensions
    :return numpy.ndarray: map between node id to neighbor node ids
    """
    # Unique neighbor cell ids
    min_id = np.min(nbr_inc)
    shifted_nbr_ids = nbr_inc - min_id
    wrapped_nbr_ids = shifted_nbr_ids % dims
    ushape = np.max(wrapped_nbr_ids) + 1
    uids = np.zeros((ushape, ushape, ushape), dtype=np.bool_)
    for wrapped_ids in wrapped_nbr_ids:
        uids[wrapped_ids[0], wrapped_ids[1], wrapped_ids[2]] = True
    uq_ids = np.array(list([list(x) for x in uids.nonzero()])).T + min_id
    # Build neighbor map based on unique neighbor ids
    shape = (dims[0], dims[1], dims[2], len(uq_ids), 3)
    neigh_mp = np.zeros(shape, dtype=np.int32)
    for xid in numba.prange(dims[0]):
        for yid in numba.prange(dims[1]):
            for zid in numba.prange(dims[2]):
                idx = np.array([xid, yid, zid])
                neigh_mp[xid, yid, zid, :, :] = (uq_ids + idx) % dims
    return neigh_mp