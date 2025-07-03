import pytest

from nemd import np
from nemd import oplsua


class TestBase:

    def testParquet(self):
        assert oplsua.Base.parquet.endswith('base.parquet')

    def testNpy(self):
        assert oplsua.Base.npy.endswith('base.npy')


class TestCharge:

    def testInit(self):
        assert (216, 1) == oplsua.Charge().shape


class TestSmiles:

    def testInit(self):
        smiles = oplsua.Smiles()
        assert (35, 5) == smiles.shape
        assert {135: 136} == smiles.hs.iloc[1]


class TestVdw:

    def testInit(self):
        assert (216, 2) == oplsua.Vdw().shape


class TestAtom:

    @pytest.fixture
    def atom(self):
        return oplsua.Atom()

    def testInit(self, atom):
        assert (216, 6) == atom.shape

    def testAtomicNumber(self, atom):
        assert (216, ) == atom.atomic_number.shape

    def testConnectivity(self, atom):
        assert (216, ) == atom.connectivity.shape


class TestBondIndex:

    @pytest.fixture
    def bond(self):
        return oplsua.Bond().row

    @pytest.mark.parametrize('row,expected', [((83, 86), 142), ((86, 83), 142),
                                              ((86, 84), 0),
                                              ((86, 1000), None), ((0, 1), 0)])
    def testIndex(self, bond, row, expected):
        assert expected == bond.index(row)

    def testGetBlock(self, bond):
        assert 151 == bond.getBlock().size

    def testGetCsr(self, bond):
        assert (215, 216) == oplsua.BondIndex.getCsr(bond).shape

    @pytest.mark.parametrize('row,expected', [((107, 84), 4)])
    def testGetFlipped(self, bond, row, expected):
        indexes, head_tail = bond.getFlipped(row)
        assert expected == indexes.shape[0] == head_tail.shape[0]

    @pytest.mark.parametrize('row,expected', [((107, 84), 2)])
    def testGetPartial(self, bond, row, expected):
        indexes, head_tail = bond.getPartial(row)
        assert expected == indexes.shape[0] == head_tail.shape[0]

    def testHeadTail(self, bond):
        assert (151, 2) == bond.head_tail.shape

    def testFlipped(self):
        indexes = np.array([[1]])
        head_tail = np.array([[3, 4]])
        indexes, head_tail = oplsua.BondIndex.flipped(indexes, head_tail)
        np.testing.assert_equal(indexes, np.array([[1], [1]]))
        np.testing.assert_equal(head_tail, np.array([[3, 4], [4, 3]]))


class TestAngleIndex:

    @pytest.fixture
    def angle(self):
        return oplsua.Angle().row

    @pytest.mark.parametrize('row,expected', [((105, 107, 107), 1)])
    def testGetBlock(self, angle, row, expected):
        assert expected == angle.getBlock(row).data.size

    @pytest.mark.parametrize('row,expected', [((105, 107, 107), (1, 1, 2))])
    def testGetPartial(self, angle, row, expected):
        indexes, head_tail = angle.getPartial(row)
        assert expected == (*indexes.shape, *head_tail.shape)


class TestDihedralIndex:

    @pytest.fixture
    def dihedral(self):
        return oplsua.Dihedral().row

    @pytest.mark.parametrize('row,expected', [((2, 1, 3, 5), 94)])
    def testGetBlock(self, dihedral, row, expected):
        assert expected == dihedral.getBlock(row).data.size

    @pytest.mark.parametrize('stop,expected', [(False, 630), (True, 631)])
    def testGetRange(self, dihedral, stop, expected):
        assert expected == dihedral.getRange(stop=stop).max()

    @pytest.mark.parametrize('row,expected',
                             [((105, 104, 107, 83), (630, 105, 83)),
                              ((83, 107, 104, 105), (630, 83, 105)),
                              ((1, 30, 30, 17), (503, 503, 1, 17, 17, 1))])
    def testGetFlipped(self, dihedral, row, expected):
        indexes, head_tail = dihedral.getFlipped(row)
        assert expected == (*indexes, *head_tail.flatten())


class TestBond:

    @pytest.fixture
    def bonds(self):
        return oplsua.Bond(atoms=oplsua.Atom())

    @pytest.fixture
    def atoms(self, mol, idx):
        bond = mol.GetBondWithIdx(idx)
        return [bond.GetBeginAtom(), bond.GetEndAtom()]

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(1, 143), (2, 149)])
    def testBond(self, bonds, mol, atoms, expected):
        oplsua.Typer().type(mol)
        assert expected == bonds.match(atoms)

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(1, (107, 84)), (3, (104, 105))])
    def testGetTypes(self, bonds, atoms, mol, expected):
        oplsua.Typer().type(mol)
        assert expected == bonds.getTypes(atoms)

    def testMaps(self, bonds):
        assert [13, 6] == [len(x) for x in bonds.maps]

    @pytest.mark.parametrize('tids,expected', [((107, 84), (107, 84)),
                                               ((86, 107), (86, 86))])
    def testGetCtype(self, bonds, tids, expected):
        assert expected == bonds.getCtype(tids)

    def testRowMap(self, bonds):
        assert (3, 151) == bonds.row.shape

    @pytest.mark.parametrize('smiles,idx,tids,expected',
                             [('CC(C)O', 1, (107, 84), 143)])
    def testGetPartial(self, bonds, atoms, tids, expected):
        assert expected == bonds.getPartial(tids, atoms)

    @pytest.mark.parametrize('smiles', ['CC(C)O'])
    @pytest.mark.parametrize('idx,expected', [(0, 4), (2, 4)])
    def testGetConn(self, mol, idx, expected):
        assert expected == oplsua.Bond.getConn(mol.GetAtomWithIdx(idx))

    def testHasH(self, bonds):
        assert (151, ) == bonds.has_h.shape


class TestDihedral:

    @pytest.fixture
    def dihedrals(self):
        return oplsua.Dihedral()

    @pytest.mark.parametrize('tids,expected',
                             [((11, 26, 76, 24), (11, 26, 76, 24)),
                              ((84, 107, 9, 107), (84, 9, 9, 107))])
    def testGetCtype(self, dihedrals, tids, expected):
        assert expected == dihedrals.getCtype(tids)


class TestImproper:

    @pytest.fixture
    def improper(self):
        return oplsua.Improper(atoms=oplsua.Atom())

    @pytest.fixture
    def atoms(self, mol, ids):
        return [mol.GetAtomWithIdx(x) for x in ids]

    @pytest.mark.parametrize('smiles,ids,expected',
                             [('CC(C)C', [0, 2, 1, 3], 30)])
    def testGetMatched(self, improper, atoms, expected):
        assert expected == improper.match(atoms)

    def testRow(self, improper):
        assert 11 == len(improper.row)

    def testConnAtomic(self, improper):
        assert (76, 5) == improper.conn_atomic.shape
        for conn_atomic, index in improper.row.items():
            ids = (improper.conn_atomic == conn_atomic).all(axis=1)
            assert list(ids).index(True) == index
            parms = improper.loc[ids].drop(columns=improper.ID_COLS)
            assert 1 == np.unique(parms, axis=0).shape[0]


class TestParser:

    @pytest.fixture
    def parser(self):
        return oplsua.Parser()

    @pytest.mark.parametrize('smiles,expected',
                             [('[Ar]', [211]),
                              ('CC(C)O', [84, 108, 84, 104, 105]),
                              ('[Br-].[Mg+2].[Br-]', [208, 208, 202])])
    def testType(self, parser, mol, expected):
        parser.type(mol)
        assert expected == [x.GetIntProp('type_id') for x in mol.GetAtoms()]

    def testAtoms(self, parser):
        assert (216, 6) == parser.atoms.shape

    def testVdws(self, parser):
        assert (216, 2) == parser.vdws.shape

    def testCharges(self, parser):
        assert (216, 1) == parser.charges.shape

    def testBonds(self, parser):
        assert (151, 4) == parser.bonds.shape

    def testAngles(self, parser):
        assert (310, 5) == parser.angles.shape

    def testImpropers(self, parser):
        assert (76, 7) == parser.impropers.shape

    def testDihedrals(self, parser):
        assert (631, 8) == parser.dihedrals.shape

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
    def typer(self, mol):
        typer = oplsua.Typer()
        typer.setUp(mol)
        return typer

    @pytest.mark.parametrize('smiles,expected', [('[Ar]', 1), ('CC(C)O', 5),
                                                 ('[Br-].[Mg+2].[Br-]', 3)])
    def testSetUp(self, typer, expected):
        assert expected == typer.mx

    @pytest.mark.parametrize('smiles,expected',
                             [('C', 81), ('O', 78), ('CO', 106), ('CCO', 107),
                              ('CC(C)', 86), ('CCCO', 107), ('CC(C)O', 108),
                              ('CC(=O)C', 129), ('CCC(=O)CC', 130),
                              ('CC(C)(C)', 88), ('CC(=O)C(C)(C)', 129),
                              ('[Li+].[F-]', 206), ('[K+].[Br-]', 208),
                              ('[Na+].[Cl-]', 207),
                              ('[Br-].[Mg+2].[Br-]', 208),
                              ('[Rb+].[Cl-].[Cs+].[Br-]', 208),
                              ('[Ca+2].[Sr+2].[Ba+2].[Cl-]', 207),
                              ('[He].[Ne].[Ar].[Kr].[Xe]', 213)])
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
