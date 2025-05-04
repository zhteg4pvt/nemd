# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Force fields, such as OPLS-UA, SW...
"""
from nemd import oplsua
from nemd import sw
from nemd import symbols


def get(name, *args, **kwargs):
    """
    Get the force field object.

    :param name str: Name of the force field.
    :return str or `oplsua.Parser`: the force field file or parser.
    """
    match name:
        case symbols.SW:
            return sw.get_file(*args, **kwargs)
        case symbols.OPLSUA:
            return oplsua.Parser.get(*args)
