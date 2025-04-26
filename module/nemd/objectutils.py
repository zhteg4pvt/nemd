# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Object utilities.
"""
import re


class Object(object):
    """
    Class with name.
    """

    @classmethod
    @property
    def name(cls):
        """
        The class name.

        :return str: the name of the class.
        """
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '_'.join([x.lower() for x in words])
