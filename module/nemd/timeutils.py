# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
datetime utilities.
"""
import datetime
import re

HMS_FMT = '%H:%M:%S'
HMS_MDY = f'{HMS_FMT} %m/%d/%Y'
ZERO = datetime.datetime.strptime('00:00:00', HMS_FMT)


def ctime(fmt=HMS_MDY):
    """
    Get current time.

    :param fmt str: the time format
    :return str: current time
    """

    return datetime.datetime.now().strftime(fmt)


def dtime(strftime, fmt=HMS_MDY):
    """
    Get the datatime from str time.

    :param strftime str: the string representation of time
    :param fmt str: the format to parse input time str
    :return 'datetime.datetime': the datatime
    """
    return datetime.datetime.strptime(strftime, fmt)


def delta2str(delta, fmt=HMS_FMT):
    """
    Convert a timedelta object to a string representation.

    :param delta 'datetime.timedelta': the timedelta object to convert
    :param fmt str: the format to print the time
    :return str: the string representation of the timedelta
    """
    try:
        formattd = (ZERO + delta).strftime(fmt)
    except (TypeError, ValueError):
        return 'nan'
    if delta.days:
        formattd = f"{delta.days} days, {formattd}"
    return formattd


def str2delta(value, fmt=HMS_FMT, rex=re.compile(r'^(\d+) days, ([:\d]*)$')):
    """
    Convert a string representation of time to a timedelta object.

    :param value str: the string representation of time
    :param fmt str: the format to parse the input string excluding days
    :param rex re.Pattern: deltatime regular expression
    :return 'datetime.timedelta': the timedelta object based on input string
    """
    matched = rex.match(value)
    if matched:
        value = matched.group(2)
    delta = dtime(value, fmt=fmt) - ZERO
    if matched:
        delta += datetime.timedelta(days=int(matched.group(1)))
    return delta
