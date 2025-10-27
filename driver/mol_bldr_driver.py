# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver builds a single molecule.
"""
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import polymutils
from nemd import structutils


class Single(logutils.Base):
    """
    Build a structure with single molecule.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mol = None
        self.struct = None

    def run(self):
        """
        Main method to build the cell.
        """
        self.setMol()
        self.setStruct()
        self.logSubstruct()

    def setMol(self):
        """
        Set the molecule.
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
        Set the structure.
        """
        self.struct = structutils.GriddedStruct.fromMols([self.mol],
                                                         options=self.options,
                                                         logger=self.logger)
        self.struct.run()
        self.struct.write()

    def logSubstruct(self):
        """
        Log substructure information.
        """
        if self.options.substruct and len(self.options.substruct) == 1:
            measured = self.struct.mols[0].GetConformer().measure()
            if measured is None:
                measured = 'matches no substructure.'
            self.log(f"{self.options.substruct[0]} {measured}")


if __name__ == "__main__":
    logutils.Script.run(Single, parserutils.MolBldr(descr=__doc__))
