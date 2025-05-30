# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This amorphous builder driver builds polymers from constitutional repeat units,
and packs molecules as an amorphous cell.
"""
import sys

from nemd import jobutils
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import polymutils
from nemd import structutils


class Amorphous(logutils.Base):
    """
    Build amorphous structure from molecules.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver':  Parsed command-line options
        """
        super().__init__(**kwargs)
        self.options = options
        self.mols = []
        self.struct = None

    def run(self):
        """
        Main method to build the cell.
        """
        self.setMols()
        self.setGridded()
        self.setPacked()
        self.setGrowed()
        self.write()

    def setMols(self):
        """
        Build polymer from monomers if provided.
        """
        for cru, cru_num, mol_num in zip(self.options.cru,
                                         self.options.cru_num,
                                         self.options.mol_num):
            moieties = polymutils.Moieties(cru,
                                           cru_num=cru_num,
                                           options=self.options,
                                           logger=self.logger)
            moieties.run()
            self.mols.extend(moieties.getEmbedMols(mol_num))

    def setGridded(self):
        """
        Build gridded cell.
        """
        if self.options.method != parserutils.AmorpBldr.GRID:
            return

        self.struct = structutils.GriddedStruct.fromMols(self.mols,
                                                         options=self.options)
        self.struct.run()

    def setPacked(self):
        """
        Build packed cell.

        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        if self.options.method != parserutils.AmorpBldr.PACK:
            return
        self.create(ClassStruct=structutils.PackedStruct)

    def setGrowed(self):
        """
        Build packed cell.
        """
        if self.options.method != parserutils.AmorpBldr.GROW:
            return
        self.create(ClassStruct=structutils.GrownStruct)

    def create(self, ClassStruct=None, mini_density=0.001, num=5):
        """
        Create amorphous cell.

        :param ClassStruct 'Struct': the structure class.
        :param mini_density float: the minium density for liquid and solid.
        :param num int: the number of densities to try.
        """
        if ClassStruct is None:
            ClassStruct = structutils.PackedStruct
        self.struct = ClassStruct.fromMols(self.mols, options=self.options)
        step = min([0.1, self.options.density / num])
        while self.struct.density >= min([mini_density, step]):
            if self.struct.run():
                return
            self.struct.density -= step
            self.log(f'Density is reduced to {self.struct.density:.4f} g/cm^3')
        self.error(f"Amorphous structure cannot be built with density as low "
                   f"as {self.struct.density} g/cm^3")

    def write(self):
        """
        Write amorphous cell into data file.
        """
        self.struct.writeData()
        for warning in self.struct.getWarnings():
            self.warning(f'{warning}')
        self.struct.writeIn()
        self.log(f'Data file written into {self.struct.datafile}')
        self.log(f'In script written into {self.struct.inscript}')
        jobutils.Job.reg(self.struct.datafile)
        jobutils.Job.reg(self.struct.inscript, file=True)


def main(argv):
    parser = parserutils.AmorpBldr(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        cell = Amorphous(options, logger=logger)
        cell.run()


if __name__ == "__main__":
    main(sys.argv[1:])
