import os
import re
import shutil
from unittest import mock

import lammps_driver as driver
import pytest

from nemd import envutils
from nemd import parserutils


class TestLammps:

    SI_DIR = envutils.test_data('si')
    SI_IN = os.path.join(SI_DIR, 'crystal_builder.in')
    AR_IN = envutils.test_data('ar', 'ar100.in')

    @pytest.fixture
    def raw(self, args, logger, tmp_dir):
        options = parserutils.Lammps().parse_args(args)
        return driver.Lammps(options, logger=logger)

    @pytest.fixture
    def lmp(self, infile, tmp, logger, dirname, tmp_dir):
        if not os.path.isabs(infile):
            test_data = envutils.test_data(dirname)
            shutil.copytree(test_data, tmp, dirs_exist_ok=True)
        if tmp:
            infile = os.path.join(tmp, infile)
        options = parserutils.Lammps().parse_args([infile])
        return driver.Lammps(options, logger=logger)

    @pytest.mark.parametrize('dirname', [SI_DIR])
    @pytest.mark.parametrize('infile,tmp,expected',
                             [('crystal_builder.in', 'tmp', 'tmp'),
                              ('crystal_builder.in', os.curdir, ''),
                              (SI_IN, None, SI_DIR)])
    def testSetUp(self, lmp, tmp, expected):
        lmp.setUp()
        if tmp is None:
            expected = os.path.dirname(lmp.options.inscript)
        expected = os.path.join(expected, 'crystal_builder.data')
        assert expected == re.search(r"read_data\s+(\S+)", lmp.cont).group(1)

    @pytest.mark.parametrize('dirname', [SI_DIR])
    @pytest.mark.parametrize('infile,tmp,expected',
                             [('crystal_builder.in', 'tmp', False),
                              ('crystal_builder.in', os.curdir, False),
                              (SI_IN, None, True)])
    def testParent(self, lmp, expected):
        assert expected == lmp.parent.is_absolute()

    @pytest.mark.parametrize('dirname', [SI_DIR])
    @pytest.mark.parametrize('infile,tmp,expected',
                             [('crystal_builder.in', 'tmp', False),
                              (SI_IN, None, True), (AR_IN, None, None)])
    def testSetPair(self, lmp, expected):
        lmp.setPair()
        match = re.search(r"pair_coeff\s+\S+\s+\S+\s+(\S+)", lmp.cont)
        assert (match is None) if expected is None else os.path.exists(
            match.group(1))

    @pytest.mark.parametrize('args,expected', [([SI_IN], 11), ([AR_IN], 22)])
    def testCont(self, raw, expected):
        assert expected == len(raw.cont.split('\n'))

    @pytest.mark.parametrize('args,expected', [([SI_IN], True),
                                               ([AR_IN], True)])
    def testAddPath(self, raw, expected):
        raw.addPath(re.compile(r"read_data\s+(\S+)"))
        data_file = re.search(r"read_data\s+(\S+)", raw.cont).group(1)
        assert expected == os.path.isabs(data_file)

    @pytest.mark.parametrize('args,expected',
                             [([SI_IN], 'crystal_builder.in')])
    def testWrite(self, raw, expected):
        raw.write()
        assert os.path.exists(expected)

    @pytest.mark.parametrize('args,expected', [([SI_IN], 0)])
    def testRun(self, raw, expected):
        assert expected == raw.run().returncode

    @pytest.mark.parametrize('args', [[SI_IN]])
    @pytest.mark.parametrize('returncode,expected', [(0, 11), (1, 6)])
    def testArgs(self, raw, returncode, expected):
        return_value = mock.Mock(returncode=returncode)
        with mock.patch('nemd.process.Process.run', return_value=return_value):
            assert expected == len(raw.args)
