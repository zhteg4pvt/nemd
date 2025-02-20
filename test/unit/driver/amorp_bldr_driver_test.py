import os
from unittest import mock

import amorp_bldr_driver as driver
import numpy as np
import pytest

from nemd import structutils


class TestAmorphous:

    @pytest.fixture
    def amorp(self, argv):
        options = driver.validate_options(argv)
        return driver.Amorphous(options, logger=mock.Mock())

    @pytest.mark.parametrize(
        "argv,nums",
        [(['[Ar]'], [1]), (['*C*', '-cru_num', '4', '-mol_num', '2'], [2]),
         (['*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3'], [2, 3])])
    def testSetMols(self, amorp, nums):
        amorp.setMols()
        assert len(nums) == len(amorp.mols)
        for num, mol in zip(nums, amorp.mols):
            assert num == mol.GetNumConformers()

    @pytest.mark.parametrize(
        "argv,edges,num",
        [(['[Ar]'], [4, 4, 4], 1),
         (['*C*', '-cru_num', '4', '-mol_num', '2'], [11.56, 9.59, 13.06], 8),
         (['*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3'
           ], [11.56, 9.59, 13.06], 17)])
    def testSetGridded(self, amorp, edges, num):
        amorp.setMols()
        amorp.options.cell = 'grid'
        amorp.setGridded()
        assert num == amorp.struct.atom_total
        assert np.allclose(edges, amorp.struct.box.hi, atol=0.01)

    @pytest.mark.parametrize(
        "argv,edges,num",
        [(['[Ar]'], [5.10, 5.10, 5.10], 1),
         (['*C*', '-cru_num', '4', '-mol_num', '2'], [7.28, 7.28, 7.28], 8),
         (['*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3'
           ], [8.27, 8.27, 8.27], 17)])
    def testSetPacked(self, amorp, edges, num):
        amorp.setMols()
        amorp.options.cell = 'pack'
        amorp.setPacked()
        assert num == amorp.struct.atom_total
        assert np.allclose(edges, amorp.struct.box.hi, atol=0.01)

    @pytest.mark.parametrize(
        "argv,edges,num",
        [(['[Ar]'], [5.10, 5.10, 5.10], 1),
         (['*C*', '-cru_num', '4', '-mol_num', '2'], [7.28, 7.28, 7.28], 8),
         (['*C*', 'O', '-cru_num', '4', '1', '-mol_num', '2', '3'
           ], [8.27, 8.27, 8.27], 17)])
    def testSetGrowed(self, amorp, edges, num):
        amorp.setMols()
        amorp.options.cell = 'grow'
        amorp.setGrowed()
        assert num == amorp.struct.atom_total
        assert np.allclose(edges, amorp.struct.box.hi, atol=0.01)

    @mock.patch('nemd.structutils.PackedStruct.runWithDensity')
    @mock.patch('amorp_bldr_driver.error')
    @pytest.mark.parametrize("argv,threshold,density",
                             [(['[Ar]', '-density', '2.65'], 2.63, 2.55),
                              (['[Ar]', '-density', '2.5'], 0.25, 0.2),
                              (['[Ar]', '-density', '0.4'], 0.25, 0.24),
                              (['[Ar]', '-density', '0.2'], 0.05, 0.04),
                              (['[Ar]', '-density', '0.005'], 0.0019, 0.001),
                              (['[Ar]', '-density', '0.0005'], 0., 0.0001)])
    def testCreate(self, error_mock, run_mock, amorp, threshold, density):
        amorp.setMols()
        run_mock.side_effect = lambda x: x > threshold and (
            _ for _ in ()).throw(structutils.DensityError)
        amorp.create()
        run_mock.assert_called_with(pytest.approx(density, rel=1e-4))
        failed = threshold < 0.001 and threshold < amorp.options.density / 5
        assert failed == error_mock.called

    @pytest.mark.parametrize('argv', [(['[Ar]'])])
    def testWrite(self, argv, amorp, tmp_dir):
        amorp.setMols()
        amorp.create()
        amorp.write()
        assert os.path.exists(amorp.struct.inscript)
        assert os.path.exists(amorp.struct.datafile)
