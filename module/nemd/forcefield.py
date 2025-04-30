# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Different types of force fields, such as OPLS-UA, SW...
"""
from nemd import oplsua
from nemd import sw
from nemd import symbols


def get(name, *args, struct=None):
    """
    Get the force field object.

    :param name str: Name of the force field.
    :param struct `Struct`: the structure for additional information.
    :return str or `oplsua.Parser`: the force field file or parser.
    """
    if name == symbols.SW:
        return sw.get_file(args, struct=struct)
    if name == symbols.OPLSUA:
        return oplsua.Parser.get(wmodel=args[0])
