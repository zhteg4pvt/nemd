# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
LAMMPS input file generator for Stillinger force field.
"""
from nemd import lammpsin
from nemd import lmpatomic
from nemd import pbc
from nemd import symbols


class Struct(lmpatomic.Struct, lammpsin.In):

    V_UNITS = lammpsin.In.METAL
    V_ATOM_STYLE = lammpsin.In.ATOMIC
    V_PAIR_STYLE = lammpsin.In.SW

    def __init__(self, struct=None, options=None, **kwargs):
        """
        :param struct Struct: struct object with moelcules and conformers.
        :param ff 'OplsParser': the force field class.
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(struct=struct, **kwargs)
        lammpsin.In.__init__(self, options=options, **kwargs)

    def coeff(self):
        """
        Write pair coefficients when data file doesn't contain the coefficients.
        """
        elements = symbols.SPACE.join(self.masses.comment).strip()
        self.fh.write(f"{self.PAIR_COEFF} * * {self.ff} {elements}\n")

    @property
    def box(self):
        """
        Write box information.
        """
        # FIXME: crystal mixture and interface
        return pbc.Box.fromParams(*self.mols[0].vecs)
