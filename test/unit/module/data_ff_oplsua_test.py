from unittest import mock

import pytest

from nemd import oplsua
from nemd.data.ff.oplsua import update


class TestSmiles:

    def testRead(self, tmp_dir):
        smiles = update.Smiles.read()
        assert (oplsua.Smiles.load() == smiles).all().all()
        with mock.patch.object(update.Smiles, 'parquet', 'file'):
            smiles.to_parquet()
            saved = smiles.load()
        assert (smiles == saved).all().all()


class TestBond:

    def testRead(self):
        assert (oplsua.Bond.load() == update.Bond.read()).all().all()


class TestAngle:

    def testRead(self):
        assert (oplsua.Angle.load() == update.Angle.read()).all().all()


class TestDihedral:

    def testRead(self):
        assert (oplsua.Dihedral.load() == update.Dihedral.read()).all().all()
