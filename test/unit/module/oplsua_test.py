import pytest

from nemd import oplsua


class TestBase:

    def testParquet(self):
        assert oplsua.Base.parquet.endswith('base.parquet')

    def testNpy(self):
        assert oplsua.Base.npy.endswith('base.npy')


class TestCharge:

    def testInit(self):
        assert (215, 1) == oplsua.Charge().shape


class TestSmiles:

    def testInit(self):
        smiles = oplsua.Smiles()
        assert (35, 5) == smiles.shape
        assert {134: 135} == smiles.hs.iloc[1]


class TestVdw:

    def testInit(self):
        assert (215, 2) == oplsua.Vdw().shape


class TestAtom:

    @pytest.fixture
    def atom(self):
        return oplsua.Atom()

    def testInit(self, atom):
        assert (215, 6) == atom.shape

    def testAtomicNumber(self, atom):
        assert (215, ) == atom.atomic_number.shape

    def testConnectivity(self, atom):
        assert (215, ) == atom.connectivity.shape


class TestBond:

    @pytest.fixture
    def bonds(self):
        return oplsua.Bond(atoms=oplsua.Atom())

    @pytest.fixture
    def atoms(self, mol, idx):
        bond = mol.GetBondWithIdx(idx)
        return [bond.GetBeginAtom(), bond.GetEndAtom()]

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(1, 142), (2, 148)])
    def testBond(self, bonds, mol, atoms, expected):
        oplsua.Typer().type(mol)
        assert expected == bonds.getMatched(atoms)

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(1, (106, 83)), (3, (103, 104))])
    def testGetTypes(self, bonds, atoms, mol, expected):
        oplsua.Typer().type(mol)
        assert expected == bonds.getTypes(atoms)

    def testMaps(self, bonds):
        assert [13, 6] == [len(x) for x in bonds.maps]

    @pytest.mark.parametrize('tids,expected', [((106, 83), (106, 83)),
                                               ((85, 106), (85, 85))])
    def testGetCtype(self, bonds, tids, expected):
        assert expected == bonds.getCtype(tids)

    def testRowMap(self, bonds):
        assert (3, 150) == bonds.row.shape

    @pytest.mark.parametrize('smiles,idx,tids,expected',
                             [('CC(C)O', 1, (106, 83), 142)])
    def testGetPartial(self, bonds, atoms, tids, expected):
        assert expected == bonds.getPartial(tids, atoms)

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(0, 4), (2, 4)])
    def testGetConn(self, mol, idx, expected):
        assert expected == oplsua.Bond.getConn(mol.GetAtomWithIdx(idx))

    def testHasH(self, bonds):
        assert (150, ) == bonds.has_h.shape


class TestDihedral:

    @pytest.fixture
    def dihedrals(self):
        return oplsua.Dihedral()

    @pytest.mark.parametrize('tids,expected',
                             [((10, 25, 75, 23), (10, 25, 75, 23)),
                              ((83, 106, 8, 106), (83, 8, 8, 106))])
    def testGetCtype(self, dihedrals, tids, expected):
        assert expected == dihedrals.getCtype(tids)


class TestImproper:

    @pytest.fixture
    def improper(self):
        return oplsua.Improper(atoms=oplsua.Atom())

    @pytest.fixture
    def atoms(self, mol, ids):
        return [mol.GetAtomWithIdx(x) for x in ids]

    @pytest.mark.parametrize('smiles,ids,expected', [('CC(C)C', [0,2,1,3], 30)])
    def testGetMatched(self, improper, atoms, expected):
        assert expected == improper.getMatched(atoms)

    def testRow(self, improper):
        assert 11 == len(improper.row)


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
