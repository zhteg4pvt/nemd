# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Dict utilities.
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


class Dict(dict):
    """
    Class to store key value pairs by dot notation.
    """

    def __setattr__(self, key, value):
        """
        Set the key = value by dot notation.

        :param key str: the key of the key / value pair
        :param value object convertible to str: the value to be saved.
        """
        self[key] = value

    def setattr(self, key, value):
        """
        Set the key = value by dot notation.

        :param key str: the attribute name
        :param value any: the attribute value
        """
        super(Dict, self).__setattr__(key, value)

    def __getattr__(self, key):
        """
        Get the attribute from the class and the stored data.

        :param key str: the key to retrieve the value.
        :return any: the retrieved value.
        """
        try:
            return super().__getattr__(key)
        except AttributeError:
            return self[key]
