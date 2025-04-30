import pytest

from nemd import oplsua


class TestParser:

    @pytest.fixture
    def parser(self):
        return oplsua.Parser()

    @pytest.mark.parametrize('smiles,expected',
                             [('[Ar]', [210]),
                              ('CC(C)O', [83, 107, 83, 103, 104]),
                              ('[Br-].[Mg+2].[Br-]', [207, 207, 201])])
    def testType(self, parser, mol, expected):
        parser.type(mol)
        assert expected == [x.GetIntProp('type_id') for x in mol.GetAtoms()]

    def testAtoms(self, parser):
        assert (215, 6) == parser.atoms.shape

    def testVdws(self, parser):
        assert (215, 2) == parser.vdws.shape

    def testCharges(self, parser):
        assert (215, 1) == parser.charges.shape

    def testBonds(self, parser):
        assert (150, 4) == parser.bonds.shape

    def testAngles(self, parser):
        assert (309, 5) == parser.angles.shape

    def testImpropers(self, parser):
        assert (76, 7) == parser.impropers.shape

    def testDihedrals(self, parser):
        assert (630, 8) == parser.dihedrals.shape

    @pytest.mark.parametrize('smiles,expected',
                             [('[Ar]', 39.948), ('CC(C)O', 60.096),
                              ('[Br-].[Mg+2].[Br-]', 184.113)])
    def testMolecularWeight(self, parser, mol, expected):
        parser.type(mol)
        assert expected == parser.molecular_weight(mol)

    @pytest.mark.parametrize('wmodel', [('SPC'), ('SPCE'), ('TIP3P')])
    def testGet(self, wmodel):
        parser = oplsua.Parser.get(wmodel=wmodel)
        assert wmodel == parser.typer.wmodel


class TestTyper:

    @pytest.fixture
    def typer(self, smiles, mol):
        typer = oplsua.Typer()
        typer.setUp(mol)
        return typer

    @pytest.mark.parametrize('smiles,expected', [('[Ar]', 1), ('CC(C)O', 5),
                                                 ('[Br-].[Mg+2].[Br-]', 3)])
    def testSetUp(self, typer, expected):
        assert expected == typer.mx

    @pytest.mark.parametrize('smiles,expected',
                             [('C', 80), ('O', 77), ('CO', 105), ('CCO', 106),
                              ('CC(C)', 85), ('CCCO', 106), ('CC(C)O', 107),
                              ('CC(=O)C', 128), ('CCC(=O)CC', 129),
                              ('CC(C)(C)', 87), ('CC(=O)C(C)(C)', 128),
                              ('[Li+].[F-]', 205), ('[K+].[Br-]', 207),
                              ('[Na+].[Cl-]', 206),
                              ('[Br-].[Mg+2].[Br-]', 207),
                              ('[Rb+].[Cl-].[Cs+].[Br-]', 207),
                              ('[Ca+2].[Sr+2].[Ba+2].[Cl-]', 206),
                              ('[He].[Ne].[Ar].[Kr].[Xe]', 212)])
    def testDoTyping(self, typer, expected):
        typer.doTyping()
        atoms = typer.mol.GetAtoms()
        assert expected == max(x.GetIntProp('type_id') for x in atoms)

    def testSmiles(self):
        assert (33, 6) == oplsua.Typer().smiles.shape

    @pytest.mark.parametrize('smiles,expected', [('CC(C)O', [1, 3, 1, 2, 1])])
    def testGetDeg(self, typer, expected):
        assert expected == [typer.getDeg(x) for x in typer.mol.GetAtoms()]

    @pytest.mark.parametrize(
        'smiles,matches',
        [('CC(=O)C(C)(C)', [((2, ), 26, 0, []),
                            ((0, 1, 2, 3), 24, 1, [0, 1, 2]),
                            ((1, 3, 4, 5), 21, 2, [3, 4, 5])])])
    def testMark(self, typer, matches):
        for match, idx, res_num, expected in matches:
            aids = list(typer.mark(match, typer.smiles.loc[idx], res_num))
            assert expected == aids
            if not aids:
                return
            atoms = [typer.mol.GetAtomWithIdx(x) for x in aids]
            res_nums = set([x.GetIntProp('res_num') for x in atoms]).pop()
            assert res_num == res_nums

    @pytest.mark.parametrize('smiles,expected',
                             [('C', 0), ('CCCO', 1),
                              ('[Ca+2].[Sr+2].[Ba+2].[Cl-]', 3),
                              ('[He].[Ne].[Ar].[Kr].[Xe]', 4)])
    def testSetResNum(self, typer, expected):
        typer.doTyping()
        typer.setResNum()
        res_nums = [x.GetIntProp('res_num') for x in typer.mol.GetAtoms()]
        assert expected == max(res_nums)
