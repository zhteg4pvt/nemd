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

    def __new__(cls, max_val=0, dtype=bool):
        """
        :param max_val: The maximum value of the bit array.
        :param dtype: The data type of the bit array.
        """
        array = np.zeros(max_val + 1, dtype=dtype)
        obj = np.asarray(array).view(cls)
        return obj

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
        """
        value_to_index = {x: i for i, x in enumerate(self.values)}
        try:
            return np.array([value_to_index[x] for x in values])
        except KeyError:
            raise KeyError(f"{values} not in {self.values}")
