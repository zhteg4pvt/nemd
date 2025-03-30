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

    def __new__(cls, values=None, mval=None):
        """
        :param values list: the int array values
        :param mval int: The maximum value of the bit array.
        """
        if mval is None:
            mval = max(values) if len(values) else -1
        array = np.zeros(mval + 1, dtype=bool)
        if values is not None and len(values):
            array[values] = True
        return np.asarray(array).view(cls)

    @property
    def values(self):
        """
        Return the indexes of on values.
        """
        return self.nonzero()[0]

    def index(self, values):
        """
        Return the on indexes.

        :param values: the values to retrieve the on indexes.
        :return np.ndarray: the on indexes.
        :raise KeyError: if some values are not on.
        """
        value_to_index = {x: i for i, x in enumerate(self.values)}
        try:
            return np.array([value_to_index[x] for x in values])
        except KeyError:
            raise KeyError(f"{values} not in {self.values}")

    def difference(self, other):
        """
        Get the values that are on but not in the other array.

        :param other np.ndarray: another other array
        :return list of int: the difference
        """
        copied = self.copy()
        copied[other] = False
        return copied.values
