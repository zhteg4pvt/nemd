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

    def __new__(cls, values=None, max_val=0):
        """
        :param values list: the int array values
        :param max_val int: The maximum value of the bit array.
        """
        if values is not None:
            max_val = max(values)
        array = np.zeros(max_val + 1, dtype=bool)
        if values is not None:
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
