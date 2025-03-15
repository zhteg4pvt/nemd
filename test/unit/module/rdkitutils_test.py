import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import pytestutils
from nemd import rdkitutils


class TestFunction:

    @pytestutils.Raises
    @pytest.mark.parametrize('smiles,expected', [('CN(C)C=O', 5),
                                                 ('dafd', ValueError)])
    def testMolFromSmiles(self, smiles, expected):
        assert expected == rdkitutils.MolFromSmiles(smiles).GetNumAtoms()

    def testCaptureLogging(self):
        mol = Chem.MolFromSmiles("[Mg+2]")
        with rdkitutils.capture_logging() as log:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        assert 2 == len(log)
