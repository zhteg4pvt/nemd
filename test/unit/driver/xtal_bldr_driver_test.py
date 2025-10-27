import os

import numpy as np
import pytest
import xtal_bldr_driver as driver

from nemd import parserutils


class TestCrystal:

    @pytest.fixture
    def crystal(self, argv, logger):
        options = parserutils.XtalBldr().parse_args(argv)
        return driver.Crystal(options=options, logger=logger)

    @pytest.mark.parametrize("argv,expected",
                             [(['-scale_factor', '2'], (8, 10.8614)),
                              (['-dimension', '3'], (216., 16.2921))])
    def testRun(self, crystal, expected, tmp_dir):
        crystal.run()
        to_compare = [crystal.struct.atom_total, crystal.struct.box.hi.max()]
        np.testing.assert_almost_equal(to_compare, expected)
        assert os.path.exists(crystal.struct.script.outfile)
        assert os.path.exists(crystal.struct.outfile)
