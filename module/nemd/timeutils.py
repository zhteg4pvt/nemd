# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
datetime utilities.
"""
from datetime import datetime

Xx_FMT = '%X %x'
HMS_FMT = '%H:%M:%S'
HMS_ZERO = datetime.strptime('00:00:00', HMS_FMT)


def ctime(fmt=Xx_FMT):
    """
    Get current time.

    :param fmt str: the format to print current time
    :return str: current time
    """

    return datetime.now().strftime(fmt)


def dtime(strftime, fmt=Xx_FMT):
    """
    Get the datatime from str time.

    :param strftime str: the string representation of time
    :param fmt str: the format to parse input time str
    :return 'datetime.datetime': the datatime
    """

    return datetime.strptime(strftime, fmt)


def delta2str(delta, fmt=HMS_FMT):
    """
    Convert a timedelta object to a string representation.

    :param delta 'datetime.timedelta': the timedelta object to convert
    :param fmt str: the format to print the time
    :return str: the string representation of the timedelta
    """
    return (HMS_ZERO + delta).strftime(fmt)


def str2delta(value, fmt=HMS_FMT):
    """
    Convert a string representation of time to a timedelta object.

    :param value str: the string representation of time
    :param fmt str: the format to parse the input string
    :return 'datetime.timedelta': the timedelta object based on input string
    """
    return dtime(value, fmt=fmt) - HMS_ZERO
