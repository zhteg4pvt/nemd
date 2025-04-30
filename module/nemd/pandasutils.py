# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
pandas utilities.
"""
from nemd import pd


class DataFrame(pd.DataFrame):
    """
    New DataFrame object maintains the data type.
    """

    @classmethod
    @property
    def _constructor(cls):
        """
        Return the constructor of the class.

        :return (sub-)class of 'Block': the constructor of the class
        """
        return cls
