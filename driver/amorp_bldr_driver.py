# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This amorphous builder driver builds polymers from constitutional repeat units,
and packs molecules as an amorphous cell.
"""
from nemd import jobutils
from nemd import logutils
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
        super().__init__(options=options, **kwargs)
        self.mols = []
        self.struct = None

    def run(self):
        """
        Main method to build the amorphous structure.
        """
        self.setMols()
        self.setStruct()
        self.build()
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
                                           mol_num=mol_num,
                                           options=self.options,
                                           logger=self.logger)
            moieties.run()
            self.mols.extend(moieties.mols)

    def setStruct(self):
        """
        Build the structure.
        """
        match self.options.method:
            case parserutils.AmorpBldr.GRID:
                Struct = structutils.GriddedStruct
                # FIXME: GriddedStruct supports target density
            case parserutils.AmorpBldr.PACK:
                Struct = structutils.PackedStruct
            case parserutils.AmorpBldr.GROW:
                Struct = structutils.GrownStruct
        self.struct = Struct.fromMols(self.mols, options=self.options)

    def build(self, mini=1E-10, num=5):
        """
        Build the amorphous conformers.

        :param mini float: the minium density.
        :param num int: the number of different densities to try.
        """
        step = min([0.1, self.options.density / num])
        while self.struct.density >= min([mini, step]):
            if self.struct.run():
                return
            self.struct.density -= step
            self.log(f'Density is reduced to {self.struct.density:.4f} g/cm^3')
        self.error(f"Amorphous structure cannot be built with density as low "
                   f"as {self.struct.density} g/cm^3")

    def write(self):
        """
        Write the data file.
        """
        self.struct.write()
        for warning in self.struct.getWarnings():
            self.warning(f'{warning}')
        self.log(f'Data file written into {self.struct.outfile}')
        jobutils.Job.reg(self.struct.outfile)
        self.struct.script.write()
        self.log(f'In script written into {self.struct.script.outfile}')
        jobutils.Job.reg(self.struct.script.outfile, file=True)


if __name__ == "__main__":
    logutils.Script.run(Amorphous, parserutils.AmorpBldr(descr=__doc__))
