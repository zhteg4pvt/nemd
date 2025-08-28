import glob
import os.path
from unittest import mock

import pytest

from nemd import oplsua
from nemd.data.ff.oplsua import update


@pytest.fixture
def lines():
    return update.Raw.getLines()


class TestBase:

    def testToParquet(self, tmp_dir):
        base = update.Base()
        with mock.patch.object(oplsua, 'DIRNAME', os.curdir):
            base.to_parquet()
            assert os.path.isfile('base.parquet')


class TestSmiles:

    def testInit(self):
        assert (35, 5) == update.Smiles().shape

    @pytest.mark.parametrize('smiles,expected', [('CC(C)O', [1, 3, 1, 2, 1])])
    def testGetDeg(self, mol, expected):
        assert expected == [update.Smiles.getDeg(x) for x in mol.GetAtoms()]


class TestCharge:

    def testInit(self, lines):
        assert (216, 1) == update.Charge(lines).shape

    def testGetLines(self, lines):
        assert 216 == len(update.Charge.getLines(lines))


class TestVdw:

    def testInit(self, lines):
        assert (216, 2) == update.Vdw(lines).shape


class TestAtom:

    def testInit(self, lines):
        assert (216, 6) == update.Atom(lines).shape


class TestImproper:

    @pytest.fixture
    def improper(self, lines):
        return update.Improper(lines)

    def testInit(self, improper):
        assert (76, 7) == improper.shape


class TestBond:

    @pytest.fixture
    def bond(self, lines):
        return update.Bond(lines)

    def testInit(self, bond):
        assert (151, 4) == bond.shape
        assert 13 == len(bond.TMAP)
        assert 3 == len(bond.MAP)

    def testToNpy(self, bond, tmp_dir):
        with mock.patch.object(oplsua, 'DIRNAME', os.curdir):
            bond.to_npy()
        assert os.path.isfile('bond.npy')


class TestAngle:

    @pytest.fixture
    def angle(self, lines):
        return update.Angle(lines)

    def testInit(self, angle):
        assert (310, 5) == angle.shape
        assert 10 == len(angle.TMAP)
        assert 3 == len(angle.MAP)


class TestDihedral:

    @pytest.fixture
    def dihedral(self, lines):
        return update.Dihedral(lines)

    def testInit(self, dihedral):
        assert (631, 8) == dihedral.shape
        assert 12 == len(dihedral.TMAP)
        assert 6 == len(dihedral.MAP)

    @pytest.mark.parametrize('line,expected', [
        ('torsion       6   66   69   46',
         'torsion 6 66 69 46 0.0 0.0 0.0 0.0'),
        ('torsion      51   53   54   56            1.700  180.0  2',
         'torsion 51 53 54 56 0.0 3.4 0.0 0.0'),
        ('torsion       2    1   70   20            0.100  180.0  3',
         'torsion 2 1 70 20 0.0 0.0 -0.2 0.0'),
        ('torsion      10   61   62   63            0.100    0.0  2      0.725    0.0  3',
         'torsion 10 61 62 63 0.0 -0.2 1.45 0.0')
    ])
    def testFormat(self, line, expected):
        assert expected == update.Dihedral.format(line)


class TestRaw:

    @pytest.fixture
    def raw(self):
        return update.Raw()

    def testInit(self, raw):
        assert (35, 5) == raw.smiles.shape
        assert (216, 1) == raw.charges.shape
        assert (216, 2) == raw.vdws.shape
        assert (216, 6) == raw.atoms.shape
        assert (310, 5) == raw.angles.shape
        assert (631, 8) == raw.dihedrals.shape
        assert (76, 7) == raw.impropers.shape

    def testGetLines(self, lines):
        assert 2811 == len(lines)

    def testWrite(self, raw, tmp_dir):
        with mock.patch.object(oplsua, 'DIRNAME', os.curdir):
            raw.write()
            assert 3 == len(glob.glob('*.npy'))
            assert 8 == len(glob.glob('*.parquet'))