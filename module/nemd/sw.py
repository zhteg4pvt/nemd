# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Stillinger-Weber force field.
"""
import functools

from nemd import envutils

SI = 'Si'
SW_ELEMENTS = [SI]
NAME_ELEMENTS = {SI: [SI]}


@functools.cache
def get_file(elements, struct=None):
    """
    Get the force field file for the given elements.

    https://docs.lammps.org/pair_sw.html

    :param elements tuple: the elements to be included in the force field.
    :param struct `stillinger.Struct`: the structure to retrieve elements from.
    :return str: the force field pathname.
    """
    if not elements:
        elements = struct.masses.element
    elements = set(elements)
    for name, supported in NAME_ELEMENTS.items():
        if elements.difference(supported):
            continue
        return envutils.get_data('potentials', f'{name}.sw', module='lammps')
