# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
numpy utilities.
"""
import numpy as np


class IntArray(np.ndarray):
    """
    Integers represented by the on values of a bool numpy.ndarray.
    """

    def __new__(cls, shape=None, on=None):
        """
        :param on list: the int array values to set on.
        :param shape tuple or int: shape of the bit array.
        """
        if shape is None:
            shape = max(on) + 1 if len(on) else 0
        array = np.zeros(shape, dtype=bool)
        if on is not None:
            array[on] = True
        return np.asarray(array).view(cls)

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
        value_to_index = {x: i for i, x in enumerate(self.on)}
        try:
            return np.array([value_to_index[x] for x in on])
        except KeyError:
            raise KeyError(f"{on} not in {self.on}")

    def difference(self, other, on=None):
        """
        Get the values that are on but not in the other array.

        :param other np.ndarray: another other array
        :param on np.ndarray: the values to compute difference with
        :return list of int: the difference
        """
        cped = self.copy() if on is None else IntArray(on=on, shape=self.shape)
        cped[other] = False
        return cped.on

    def less(self, value):
        """
        Get the values that are less than the input.

        :param value np.ndarray: values less than this one are returned
        :return list of int: the less than values
        """
        copied = self.copy()
        copied[value:] = False
        return copied.on
