import os
from unittest import mock

import mol_bldr_driver as driver
import pytest


class TestGrid:

    @pytest.fixture
    def grid(self, argv):
        options = driver.validate_options(argv)
        return driver.Grid(options, logger=mock.Mock())

    @pytest.mark.parametrize("argv,num", [(['[Ar]'], 1),
                                          (['*C*', '-cru_num', '4'], 4)])
    def testSetMol(self, grid, num):
        grid.setMol()
        assert grid.mol.GetNumAtoms() == num

    @pytest.mark.parametrize("argv,num", [(['[Ar]'], 1),
                                          (['*C*', '-cru_num', '4'], 4)])
    def testSetStruct(self, grid, num):
        grid.setMol()
        grid.setStruct()
        assert grid.struct.atoms.shape[0] == num

    @mock.patch('mol_bldr_driver.log')
    @pytest.mark.parametrize(
        "argv,called",
        [(['*C*', '-cru_num', '4', '-substruct', 'CCCC:45'], False),
         (['*C*', '-cru_num', '4', '-substruct', 'CCCC'], True),
         (['*C*', '-cru_num', '4', '-substruct', 'CCCCC'], True),
         (['*C*', '-cru_num', '4'], False)])
    def testLogSubstruct(self, log_mock, grid, called):
        grid.setMol()
        grid.setStruct()
        grid.logSubstruct()
        assert log_mock.called == called

    @pytest.mark.parametrize("argv", [(['*C*', '-cru_num', '4'])])
    def testWrite(self, grid, argv, tmp_dir):
        grid.setMol()
        grid.setStruct()
        grid.write()
        assert os.path.exists(grid.struct.inscript)
        assert os.path.exists(grid.struct.datafile)
