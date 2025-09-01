# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
numpy utilities.
"""
import numpy as np

from nemd import numbautils


class IntArray(np.ndarray):
    """
    Integers represented by the on values of a bool numpy.ndarray.
    """

    def __new__(cls, shape=None, on=None):
        """
        :param shape tuple or int: shape of the bit array.
        :param on np.ndarray: the int array values to set on.
        """
        if on is None:
            on = []
        if shape is None:
            shape = max(on) + 1 if len(on) else 1
        array = np.zeros(shape, dtype=bool)
        array[on] = True
        obj = np.asarray(array).view(cls)
        obj.zeros = None
        return obj

    @property
    def on(self):
        """
        Return the indexes of on values.
        """
        return self.nonzero()[0]

    def index(self, on):
        """
        Return the indexes in on values.

        :param on: the on values to retrieve the indexes.
        :return np.ndarray: the on indexes.
        :raise KeyError: if some values are not on.
        """
        if not self[on].all():
            raise KeyError('Not all values are on.')
        return np.cumsum(self)[on] - 1

    def diff(self, off, on=None):
        """
        Get the values that are on and not off.

        :param off np.ndarray: another other array
        :param on np.ndarray: the values to compute difference with
        :return list of int: the difference
        """
        if self.zeros is None:
            self.zeros = np.zeros(shape=self.shape, dtype=np.bool_)
        return self._diff(self.zeros, self.on if on is None else on, off)

    @staticmethod
    @numbautils.jit
    def _diff(zeros, on, off):
        """
        Get the values that are on and not off.

        :param zeros np.ndarray: array stays as zeros before and after.
        :param on np.ndarray: compute the difference from this to the off.
        :param off np.ndarray: compute the difference from the on to this one.
        :return list of int: the difference
        """
        zeros[on] = True
        zeros[off] = False
        left = zeros.nonzero()[0]
        zeros[left] = False
        return left

    def less(self, value):
        """
        Get the values that are less than the input.

        :param value np.ndarray: values less than this one are returned
        :return list of int: the less than values
        """
        copied = self.copy()
        copied[value:] = False
        return copied.on


def assert_almost_equal(actual, desired, **kwargs):
    """
    See np.testing.assert_almost_equal.
    """
    if desired is None:
        assert actual is None
        return
    np.testing.assert_almost_equal(actual, desired, **kwargs)
