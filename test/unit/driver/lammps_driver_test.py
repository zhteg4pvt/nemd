import os
import re
import shutil

import lammps_driver as driver
import pytest

from nemd import envutils
from nemd import parserutils

SI_DIR = envutils.test_data('si')
SI_IN = os.path.join(SI_DIR, 'crystal_builder.in')
MISS_DATA_IN = envutils.test_data('ar', 'single.in')


class TestLammps:

    @pytest.fixture
    def raw(self, args, logger, tmp_dir):
        options = parserutils.Lammps().parse_args(args)
        return driver.Lammps(options, logger=logger)

    @pytest.fixture
    def lmp(self, infile, tmp, logger, dirname, tmp_dir):
        if tmp:
            test_data = envutils.test_data(dirname)
            shutil.copytree(test_data, tmp, dirs_exist_ok=True)
            infile = os.path.join(tmp, os.path.basename(infile))
        options = parserutils.Lammps().parse_args([infile])
        return driver.Lammps(options, logger=logger)

    @pytest.mark.parametrize('dirname,read_data',
                             [(SI_DIR, re.compile(r"read_data\s+(\S+)"))])
    @pytest.mark.parametrize('infile,tmp,expected',
                             [('crystal_builder.in', 'tmp', 'tmp'),
                              ('crystal_builder.in', os.curdir, ''),
                              (SI_IN, None, SI_DIR)])
    def testSetUp(self, lmp, tmp, expected, read_data):
        lmp.setUp()
        if tmp is None:
            expected = os.path.dirname(lmp.options.inscript)
        expected = os.path.join(expected, 'crystal_builder.data')
        assert expected == read_data.search(lmp.cont).group(1)

    @pytest.mark.parametrize('args,expected', [([SI_IN], 0)])
    def testRun(self, raw, expected):
        assert expected == raw.run().returncode
