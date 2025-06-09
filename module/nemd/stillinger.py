# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
LAMMPS input file generator for Stillinger force field.
"""
from nemd import lmpatomic
from nemd import lmpin
from nemd import pbc
from nemd import symbols


class Struct(lmpatomic.Struct, lmpin.In):
    """
    The Stillinger structure.
    """
    V_UNITS = lmpin.In.METAL
    V_ATOM_STYLE = lmpin.In.ATOMIC
    V_PAIR_STYLE = lmpin.In.SW

    def __init__(self, *args, options=None, **kwargs):
        """
        :param options 'argparse.Namespace': parsed command line options.
        """
        super().__init__(*args, options=options, **kwargs)
        lmpin.In.__init__(self, options=options)

    def coeff(self):
        """
        See parent.
        """
        elements = symbols.SPACE.join(self.masses.comment).strip()
        self.fh.write(f"{self.PAIR_COEFF} * * {self.ff} {elements}\n")

    def writeData(self):
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