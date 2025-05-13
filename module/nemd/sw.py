# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Stillinger-Weber force field.
"""
import functools

from nemd import envutils

SI = 'Si'
NAME_ELEMENTS = {SI: {SI}}


@functools.cache
def get_file(*args):
    """
    Get the force field file for the given elements.

    https://docs.lammps.org/pair_sw.html

    :return str: the force field pathname.
    """
    elements = set(args)
    name = next((x for x, y in NAME_ELEMENTS.items() if y == elements), None)
    if name:
        return envutils.get_data('potentials', f'{name}.sw', module='lammps')
