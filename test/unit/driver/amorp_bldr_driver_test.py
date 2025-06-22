import os
from unittest import mock

import amorp_bldr_driver as driver
import numpy as np
import pytest

from nemd import parserutils
from nemd import structutils


class TestAmorphous:

    @pytest.fixture
    def amorp(self, argv, logger):
        options = parserutils.AmorpBldr().parse_args(argv + ['-seed', '1'])
        return driver.Amorphous(options, logger=logger)

    @pytest.mark.parametrize(
        "argv,expected",
        [(['[Ar]', '-method', 'grid'], [1, 4, 4, 4]),
         (['*C*', '-cru_num', '4', '-mol_num', '2', '-method', 'grid'
           ], [8, 11.73, 13.23, 8]),
         ([
             '*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3', '-method',
             'grid'
         ], [17, 11.72, 13.23, 8]),
         (['[Ar]', '-method', 'pack'], [1, 5.10, 5.10, 5.10]),
         (['*C*', '-cru_num', '4', '-mol_num', '2', '-method', 'pack'
           ], [8, 7.28, 7.28, 7.28]),
         ([
             '*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3', '-method',
             'pack'
         ], [17, 8.27, 8.27, 8.27]),
         (['[Ar]', '-method', 'grow'], [1, 5.10, 5.10, 5.10]),
         (['*C*', '-cru_num', '4', '-mol_num', '2', '-method', 'grow'
           ], [8, 7.28, 7.28, 7.28]),
         ([
             '*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3', '-method',
             'grow'
         ], [17, 8.27, 8.27, 8.27])])
    def testRun(self, amorp, expected, tmp_dir):
        amorp.run()
        to_compare = [amorp.struct.atom_total, *amorp.struct.box.hi]
        assert np.allclose(expected, to_compare, atol=0.01)

    @pytest.mark.parametrize(
        "argv,expected",
        [(['[Ar]'], [1]), (['*C*', '-cru_num', '4', '-mol_num', '2'], [2]),
         (['*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3'], [2, 3])])
    def testSetMols(self, amorp, expected):
        amorp.setMols()
        assert expected == [x.GetNumConformers() for x in amorp.mols]

    @pytest.mark.parametrize(
        "argv,expected",
        [(['[Ar]', '-method', 'grid'], structutils.GriddedStruct),
         (['[Ar]', '-method', 'pack'], structutils.PackedStruct),
         (['[Ar]', '-method', 'grow'], structutils.GrownStruct)])
    def testSetStruct(self, amorp, expected):
        amorp.setStruct()
        assert isinstance(amorp.struct, expected)

    @mock.patch('nemd.structutils.PackedStruct.run')
    @mock.patch('amorp_bldr_driver.Amorphous.error')
    @pytest.mark.parametrize('argv', [['[Ar]']])
    @pytest.mark.parametrize("density,success,expected",
                             [(2.65, 2.63, 2.55), (2.5, 0.25, 0.2),
                              (0.4, 0.25, 0.24), (0.2, 0.05, 0.04),
                              (0.005, 0.0019, 0.001), (0.0005, 0., False)])
    def testbuild(self, err_mock, run_mock, amorp, density, success, expected):
        amorp.options.density = density
        amorp.setMols()
        amorp.setStruct()
        run_mock.side_effect = lambda: amorp.struct.density <= success
        amorp.build()
        np.testing.assert_almost_equal(amorp.struct.density, expected)
        assert bool(expected) == (not err_mock.called)

    @pytest.mark.parametrize('argv', [['[Ar]']])
    def testWrite(self, argv, amorp, tmp_dir):
        amorp.setMols()
        amorp.setStruct()
        amorp.write()
        assert os.path.exists(amorp.struct.script.outfile)
        assert os.path.exists(amorp.struct.outfile)
