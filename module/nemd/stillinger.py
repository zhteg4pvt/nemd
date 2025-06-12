# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
LAMMPS input file generator for Stillinger force field.
"""
import functools

from nemd import lmpatomic
from nemd import lmpin
from nemd import pbc
from nemd import symbols


class Script(lmpin.Script):

    def data(self):
        """
        Write pair coefficients in addition. (see parent)
        """
        super().data()
        self.append(f"{self.PAIR_COEFF} * * {self.struct.ff} "
                    f"{symbols.SPACE.join(self.struct.masses.comment)}\n")


class Struct(lmpatomic.Struct):
    """
    The Stillinger structure.
    """
    V_UNITS = lmpin.Script.METAL
    V_ATOM_STYLE = lmpin.Script.ATOMIC
    V_PAIR_STYLE = lmpin.Script.SW

    def write(self):
        """
        Write out a LAMMPS datafile or return the content.
        """
        # FIXME: crystal mixture and interface
        self.box = pbc.Box.fromParams(*self.mols[0].vecs)
        with open(self.datafile, 'w') as self.hdl:
            self.hdl.write(f"{self.DESCR.format(style=self.V_ATOM_STYLE)}\n\n")
            self.atoms.writeCount(self.hdl)
            self.hdl.write("\n")
            self.masses.writeCount(self.hdl)
            self.hdl.write("\n")
            self.box.write(self.hdl)
            self.masses.write(self.hdl)
            self.atoms.write(self.hdl)

    @property
    @functools.cache
    def script(self):
        """
        Get the LAMMPS in-script writer.

        :return `Script`: the in-script.
        """
        return Script(struct=self)
