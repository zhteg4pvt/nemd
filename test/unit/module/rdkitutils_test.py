import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import rdkitutils


class TestFunction:

    @pytest.mark.parametrize('smiles,raise_type,is_raise',
                             [('CN(C)C=O', None, False),
                              ('dafd', ValueError, True)])
    def testMolFromSmiles(self, smiles, check_raise):
        with check_raise():
            rdkitutils.MolFromSmiles(smiles)

    def testCaptureLogging(self):
        mol = Chem.MolFromSmiles("[Mg+2]")
        with rdkitutils.capture_logging() as log:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            assert 2 == len(log)
