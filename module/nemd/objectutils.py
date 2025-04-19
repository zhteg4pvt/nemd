import re


class Object(object):

    @classmethod
    @property
    def name(cls):
        """
        The class name.

        :return str: the name of the class.
        """
        words = re.findall('[A-Z][^A-Z]*', cls.__name__)
        return '_'.join([x.lower() for x in words])
