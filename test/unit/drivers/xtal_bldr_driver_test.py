import os
from unittest import mock

import pytest
import xtal_bldr_driver as driver


class TestCrystal:

    @pytest.fixture
    def crystal(self, argv):
        options = driver.validate_options(argv)
        return driver.Crystal(options, logger=mock.Mock())

    @pytest.mark.parametrize("argv", [(['-scale_factor', '2'])])
    def testRun(self, crystal, tmp_dir):
        crystal.run()
        assert os.path.exists(crystal.struct.inscript)
        assert os.path.exists(crystal.struct.datafile)