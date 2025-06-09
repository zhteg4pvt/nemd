# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module generates and parses the LAMMPS data file.
"""
from nemd import lmpatomic
from nemd import lmpfull
from nemd import lmpin


def get_reader(pathname, style=lmpin.In.ATOMIC):
    """
    Get the appropriate reader based on the style of the data file style.

    :param pathname str: the pathname of the data file
    :param style str: the style of the data file (default: ATOMIC)
    :return `Reader`: the corresponding data file reader
    """
    match lmpatomic.Reader.getStyle(pathname):
        case lmpin.In.ATOMIC:
            return lmpatomic.Reader
        case lmpfull.In.FULL:
            return lmpfull.Reader
        case _:
            raise ValueError(f'Unsupported style: {style}')


def read(pathname):
    """
    Read the data file and return the corresponding reader.

    :param pathname str: the pathname of the data file
    :return `Reader`: the corresponding reader loaded with the data file
    """
    reader = get_reader(pathname)
    return reader(pathname)
