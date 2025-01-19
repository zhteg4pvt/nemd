import os

import dispersion_driver as driver
import pytest

from nemd import np


class TestDispersion:

    @pytest.fixture
    def disp(self, argv):
        options = driver.validate_options(argv)
        return driver.Dispersion(options)

    @pytest.mark.parametrize(
        "argv,vol", [(['-name', 'Si'], 160.16),
                     (['-name', 'Si', '-scale_factor', '1.1'], 213.18)])
    def testBuildCell(self, disp, vol):
        disp.buildCell()
        np.testing.assert_almost_equal(vol, disp.crystal.volume, decimal=2)

    @pytest.mark.parametrize(
        "argv,num", [(['-name', 'Si'], 8),
                     (['-name', 'Si', '-dimension', '2', '1', '1'], 16)])
    def testWriteDataFile(self, disp, num, tmp_dir):
        disp.buildCell()
        disp.writeDataFile()
        assert os.path.exists('dispersion.data')
        assert num == disp.struct.atom_total

    @pytest.mark.parametrize("argv",
                             [(['-name', 'Si']),
                              (['-name', 'Si', '-dimension', '2', '1', '1']),
                              (['-name', 'Si', '-scale_factor', '1.1'])])
    def testWriteDispersion(self, disp, tmp_dir):
        disp.buildCell()
        disp.writeDataFile()
        disp.writeDispersion()
        assert os.path.exists(disp.outfile)

    @pytest.mark.parametrize("argv", [(['-name', 'Si'])])
    def testPlotDispersion(self, disp, tmp_dir):
        disp.buildCell()
        disp.writeDataFile()
        disp.writeDispersion()
        disp.plotDispersion()
        assert os.path.exists('dispersion.png')
