import os

import mol_bldr_driver as driver
import pytest

from nemd import parserutils


class TestSingle:

    @pytest.fixture
    def single(self, argv, logger):
        options = parserutils.MolBldr().parse_args(argv)
        return driver.Single(options, logger=logger)

    @pytest.mark.parametrize("argv", [(['*C*', '-cru_num', '4'])])
    def testRun(self, single, argv, tmp_dir):
        single.run()
        assert os.path.exists(single.struct.script.outfile)
        assert os.path.exists(single.struct.outfile)

    @pytest.mark.parametrize("argv,expected", [(['[Ar]'], 1),
                                               (['*C*', '-cru_num', '4'], 4)])
    def testSetMol(self, single, expected):
        single.setMol()
        assert expected == single.mol.GetNumAtoms()

    @pytest.mark.parametrize("argv,expected", [(['[Ar]'], 1),
                                               (['*C*', '-cru_num', '4'], 4)])
    def testSetStruct(self, single, expected, tmp_dir):
        single.setMol()
        single.setStruct()
        assert expected == single.struct.atoms.shape[0]
        assert os.path.exists(single.struct.script.outfile)
        assert os.path.exists(single.struct.outfile)

    @pytest.mark.parametrize('argv', [['*C*', '-cru_num', '4', '-seed', '1']])
    @pytest.mark.parametrize("substruct,expected",
                             [(['CCCC', 45], False),
                              (['CCCC'], 'CCCC dihedral (degree): -0.00'),
                              (['CCCCC'], 'CCCCC matches no substructure.'),
                              (None, False)])
    def testLogSubstruct(self, single, substruct, called, tmp_dir):
        single.options.substruct = substruct
        single.setMol()
        single.setStruct()
        single.log = called
        single.logSubstruct()
