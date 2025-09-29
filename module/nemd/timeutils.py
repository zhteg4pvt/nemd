# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
datetime utilities.
"""
import datetime
import re

HMS = '%H:%M:%S'


class Date(datetime.datetime):
    """
    Customized with default format.
    """
    HMS_MDY = f'{HMS} %m/%d/%Y'

    def strftime(self, fmt=HMS_MDY):
        """
        See parent.

        :param fmt str: the format to parse input time str
        """
        return super().strftime(fmt)

    @classmethod
    def strptime(cls, *args, fmt=HMS_MDY):
        """
        See parent.

        :param fmt str: the format to parse input time str
        """
        return super().strptime(*args, fmt)

    def __sub__(self, *args, **kwargs):
        """
        See parent.
        """
        return Delta(super().__sub__(*args, **kwargs))


class Delta(datetime.timedelta):
    """
    Customized with str format.
    """
    ZERO = Date.strptime('00:00:00', fmt=HMS)
    SEP = ' days, '
    NAN = 'nan'

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], datetime.timedelta):
            args, kwargs = tuple(), dict(seconds=args[0].total_seconds())
        return super().__new__(cls, *args, **kwargs)

    def toStr(self, fmt=HMS):
        """
        Convert a timedelta object to a string representation.

        :param fmt str: the format to print the time
        :return str: the string representation of the timedelta
        """
        formattd = (self.ZERO + self).strftime(fmt)
        if self.days:
            formattd = f"{self.days}{self.SEP}{formattd}"
        return formattd

    @classmethod
    def fromStr(cls, value, fmt=HMS, rex=re.compile(rf'^(\d+){SEP}([:\d]*)$')):
        """
        Convert a string representation of time to a timedelta object.

        :param value str: the string representation of time
        :param fmt str: the format to parse the input string excluding days
        :param rex re.Pattern: deltatime regular expression
        :return 'TimeDelta': the timedelta object based on input string
        """
        if value == cls.NAN:
            return
        matched = rex.match(value)
        if matched:
            value = matched.group(2)
        delta = Date.strptime(value, fmt=fmt) - cls.ZERO
        if matched:
            delta += cls(days=int(matched.group(1)))
        return cls(seconds=delta.total_seconds())
