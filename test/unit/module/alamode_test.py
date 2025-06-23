import os.path
import types

import pytest

from nemd import alamode
from nemd import envutils
from nemd import parserutils

OPTIONS = parserutils.XtalBldr().parse_args(
    ['-JOBNAME', 'dispersion', '-temp', '0'])
DISPLACE_DAT = envutils.test_data('0044', 'displace', 'dispersion1.lammps')


@pytest.fixture
def crystal(mode):
    return alamode.Crystal.fromDatabase(OPTIONS, mode=mode)


class TestScript:

    @pytest.fixture
    def script(self):
        return alamode.Script(struct=types.SimpleNamespace(options=OPTIONS))

    def testDump(self, script):
        script.dump(1, 'all', 'custom', 1000, 'dispersion.custom', 'id')
        assert 'id xu yu zu fx fy fz' in script[0]

    def testDumpModify(self, script):
        script.dump_modify(1)
        assert "format float '%20.15f'" in script[0]


@pytest.mark.parametrize('jobname,files', [('dispersion', [DISPLACE_DAT])])
class TestLmp:

    @pytest.fixture
    def lmp(self, jobname, files, tmp_dir):
        options = parserutils.XtalBldr().parse_args(['-JOBNAME', 'dispersion'])
        mols = [alamode.Crystal.fromDatabase(options).mol]
        struct = alamode.Struct.fromMols(mols, options=options)
        return alamode.Lmp(struct, jobname=jobname, files=files)

    @pytest.mark.parametrize('expected', [('.custom')])
    def testExt(self, lmp, expected):
        assert expected == lmp.ext

    def testSetUp(self, lmp, tmp_dir):
        lmp.setUp()
        assert os.path.isfile(lmp.struct.outfile)
        assert os.path.isfile(lmp.struct.script.outfile)


class TestFunc:
    DAT = envutils.test_data('0044', 'dispersion.data')
    PATT = envutils.test_data('0044', 'suggest', 'dispersion.pattern_HARMONIC')
    STRUCT = alamode.Struct.fromMols(
        [alamode.Crystal.fromDatabase(OPTIONS).mol], options=OPTIONS)

    @pytest.mark.parametrize(
        'obj,kwargs,expected',
        [(alamode.Crystal.fromDatabase(OPTIONS), {}, 1), (STRUCT, {}, 1),
         ('displace', dict(files=[DAT, PATT], jobname='dispersion'), 1),
         ('unknown', {}, ValueError)])
    def testExe(self, obj, kwargs, expected, raises, tmp_dir):
        with raises:
            assert expected == len(alamode.exe(obj, **kwargs))


class TestCrystal:

    @pytest.mark.parametrize('mode,expected',
                             [('suggest', 34), ('optimize', 38),
                              ('phonons', 23)])
    def testWrite(self, crystal, expected, tmp_dir):
        crystal.write()
        assert os.path.exists(crystal.outfile)
        with open(crystal.outfile, 'r') as fh:
            assert expected == len(fh.readlines())


class TestGeneral:

    @pytest.fixture
    def general(self, crystal):
        return alamode.General(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', 5), ('optimize', 5),
                                               ('phonons', 6)])
    def testSetUp(self, general, expected):
        assert expected == len(general)

    @pytest.mark.parametrize('mode,expected', [('suggest', 8), ('optimize', 8),
                                               ('phonons', 9)])
    def testWrite(self, general, expected, tmp_line):
        with tmp_line() as (fh, lines):
            general.write(fh)
        assert expected == len(lines)

    @pytest.mark.parametrize('val,expected', [('hi', 'hi'), (1.2, 1.2),
                                              ((1, 2), '1 2')])
    def testFormat(self, val, expected):
        assert expected == alamode.General.format(val)


class TestOptimize:

    @pytest.fixture
    def optimize(self, crystal):
        return alamode.Optimize(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', 0), ('optimize', 1),
                                               ('phonons', 0)])
    def testSetUp(self, optimize, expected):
        assert expected == len(optimize)


class TestInteraction:

    @pytest.fixture
    def interaction(self, crystal):
        return alamode.Interaction(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', 1), ('optimize', 1),
                                               ('phonons', 0)])
    def testSetUp(self, interaction, expected):
        assert expected == len(interaction)


class TestCutoff:

    @pytest.fixture
    def cutoff(self, crystal):
        return alamode.Cutoff(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', 1), ('optimize', 1),
                                               ('phonons', 0)])
    def testSetUp(self, cutoff, expected):
        assert expected == len(cutoff)

    @pytest.mark.parametrize('mode,expected', [('suggest', '  Si-Si 7.3'),
                                               ('optimize', '  Si-Si 7.3'),
                                               ('phonons', None)])
    def testWrite(self, cutoff, expected, tmp_line):
        with tmp_line() as (fh, lines):
            cutoff.write(fh)
        assert (not lines) if expected is None else (expected in lines)


class TestCell:

    @pytest.fixture
    def cell(self, crystal):
        return alamode.Cell.fromCrystal(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', 7)])
    def testWrite(self, cell, expected, tmp_line):
        with tmp_line() as (fh, lines):
            cell.write(fh)
        assert expected == len(lines)

    @pytest.mark.parametrize('mode,expected', [('suggest', (3, 3))])
    def testFromCrystal(self, cell, expected):
        assert expected == cell.shape


class TestPosition:

    @pytest.fixture
    def position(self, crystal):
        return alamode.Position.fromCrystal(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', (8, 4)),
                                               ('optimize', (8, 4)),
                                               ('phonons', (0, 0))])
    def testFromCrystal(self, position, expected):
        assert expected == position.shape


class TestKpoint:

    @pytest.fixture
    def kpoint(self, crystal):
        return alamode.Kpoint.fromCrystal(crystal)

    @pytest.mark.parametrize('mode,expected', [('suggest', (0, 0)),
                                               ('optimize', (0, 0)),
                                               ('phonons', (3, 9))])
    def testFromCrystal(self, kpoint, expected):
        assert expected == kpoint.shape
