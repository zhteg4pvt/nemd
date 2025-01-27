# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver builds a single molecule.
"""
import sys

from nemd import jobutils
from nemd import logutils
from nemd import polymutils
from nemd import structutils
from nemd import task


class Grid(logutils.Base):
    """
    A grid cell with a fixed number of molecules.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(**kwargs)
        self.options = options
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
        Build polymer from monomers if provided.
        """
        moieties = polymutils.Moieties(self.options.cru[0],
                                       cru_num=self.options.cru_num[0],
                                       options=self.options)
        moieties.setUp()
        self.mol = polymutils.Mol(moieties.mols[0],
                                  self.options.mol_num[0],
                                  moieties=moieties,
                                  options=self.options,
                                  logger=self.logger)

    def setStruct(self):
        """
        Build gridded cell.
        """
        self.struct = structutils.GriddedStruct.fromMols([self.mol],
                                                         options=self.options)
        self.struct.run()

    def logSubstruct(self):
        """
        Log substructure information.
        """
        if not self.options.substruct or self.options.substruct[1] is not None:
            return
        val = self.struct.molecules[0].GetConformer().measure()
        if val is not None:
            self.log(f"{self.options.substruct[0]} {val}")
            return
        self.log(
            f'{self.options.substruct[0]} does not match any substructures')

    def write(self):
        """
        Write amorphous cell into data file.
        """
        self.struct.writeData()
        self.struct.writeIn()
        self.log(f'Data file written into {self.struct.datafile}')
        self.log(f'In script written into {self.struct.inscript}')
        jobutils.add_outfile(self.struct.datafile)
        jobutils.add_outfile(self.struct.inscript, file=True)


def main(argv):
    parser = task.MolBldrJob.get_parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        cell = Grid(options, logger=logger)
        cell.run()


if __name__ == "__main__":
    main(sys.argv[1:])
