import os
import re
import shutil
from unittest import mock

import conftest
import lammps_driver as driver
import pytest

from nemd import envutils
from nemd import parserutils


@conftest.require_src
class TestLammps:

    SI_DIR = envutils.Src().test('si')
    SI_IN = envutils.Src().test('si', 'crystal_builder.in')
    AR_IN = envutils.Src().test('ar', 'ar100.in')
    EMPTY_IN = envutils.Src().test('ar', 'empty.in')
    WRONG_IN = envutils.Src().test('ar', 'error.in')

    @pytest.fixture
    def raw(self, args, logger, tmp_dir):
        options = parserutils.Lammps().parse_args(args)
        return driver.Lammps(options, logger=logger)

    @pytest.fixture
    def lmp(self, infile, tmp, logger, dirname, tmp_dir):
        if not os.path.isabs(infile):
            test_data = envutils.Src().test(dirname)
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

    @pytest.mark.parametrize('dirname', ['he'])
    @pytest.mark.parametrize('args,expected', [([SI_IN], True),
                                               ([AR_IN], True),
                                               ([EMPTY_IN], None),
                                               (['mol_bldr.in'], False)])
    def testAddPath(self, args, expected, logger, copied):
        options = parserutils.Lammps().parse_args(args)
        lmp = driver.Lammps(options, logger=logger)
        lmp.addPath(re.compile(r"read_data\s+(\S+)"))
        match = re.search(r"read_data\s+(\S+)", lmp.cont)
        assert expected == (match and os.path.isabs(match.group(1)))

    @pytest.mark.parametrize('args,expected',
                             [([SI_IN], 'crystal_builder.in')])
    def testWrite(self, raw, expected):
        raw.write()
        assert os.path.exists(expected)

    @pytest.mark.parametrize(
        'args,expected',
        [([SI_IN], None), ([EMPTY_IN], None),
         ([WRONG_IN], 'Unknown command: error (src/input.cpp:314)')])
    def testRun(self, raw, expected, called):
        raw.error = called
        raw.run()
        assert bool(expected) == bool(raw.proc.returncode)

    @pytest.mark.parametrize('args', [[SI_IN]])
    @pytest.mark.parametrize('returncode,expected', [(0, 9), (1, 4)])
    def testArgs(self, raw, returncode, expected):
        with mock.patch('subprocess.run',
                        return_value=mock.Mock(returncode=returncode)):
            assert expected == len(raw.args)
