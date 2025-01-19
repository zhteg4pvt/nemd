import rdkit

from nemd import rdkitutils


class TestFunction:

    def testCaptureLogging(self):
        mol = rdkit.Chem.MolFromSmiles("[Mg+2]")
        with rdkitutils.capture_logging() as log:
            rdkit.Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
            assert 2 == len(log)
