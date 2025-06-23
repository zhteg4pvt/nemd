# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver builds a single molecule.
"""
import sys

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import polymutils
from nemd import structutils


class Single(logutils.Base):
    """
    Build a structure with single molecule.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver':  Parsed command-line options
        """
        super().__init__(options=options, **kwargs)
        self.mol = None
        self.struct = None

    def run(self):
        """
        Main method to build the cell.
        """
        self.setMol()
        self.setStruct()
        self.logSubstruct()
        self.write()

    def setMol(self):
        """
        Build the molecule.
        """
        for cru, cru_num, mol_num in zip(self.options.cru,
                                         self.options.cru_num,
                                         self.options.mol_num):
            moieties = polymutils.Moieties(cru,
                                           cru_num=cru_num,
                                           mol_num=mol_num,
                                           options=self.options,
                                           logger=self.logger)
            moieties.run()
            self.mol = moieties.mols[0]

    def setStruct(self):
        """
        Build the structure.
        """
        self.struct = structutils.GriddedStruct.fromMols([self.mol],
                                                         options=self.options)
        self.struct.run()

    def logSubstruct(self):
        """
        Log substructure information.
        """
        if not (self.options.substruct and self.options.substruct[1] is None):
            return
        val = self.struct.mols[0].GetConformer().measure()
        self.log(f"{self.options.substruct[0]} "
                 f"{'matches no substructure' if val is None else val}")

    def write(self):
        """
        Write the data file.
        """
        self.struct.write()
        self.log(f'Data file written into {self.struct.outfile}')
        jobutils.Job.reg(self.struct.outfile)
        self.struct.script.write()
        self.log(f'In script written into {self.struct.script.outfile}')
        jobutils.Job.reg(self.struct.script.outfile, file=True)


def main(argv):
    parser = parserutils.MolBldr(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        cell = Single(options, logger=logger)
        cell.run()


if __name__ == "__main__":
    main(sys.argv[1:])
