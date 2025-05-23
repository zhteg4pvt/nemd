import pytest

from nemd import np
from nemd import structure


class TestConformer:

    @pytest.fixture
    def conf(self, mol):
        return structure.Conf(mol=mol)

    @pytest.mark.parametrize('smiles,expected', [('O', True), (None, False)])
    def testHasOwningMol(self, conf, expected):
        assert expected == conf.HasOwningMol()

    @pytest.mark.parametrize('smiles', ['O', None])
    def testGetOwningMol(self, conf, mol):
        assert mol == conf.GetOwningMol()

    @pytest.mark.parametrize('smiles', ['O', None])
    @pytest.mark.parametrize('xyz', [np.zeros((3, 3))])
    def testSetPositions(self, conf, xyz):
        conf.setPositions(xyz)
        np.testing.assert_equal(xyz, conf.GetPositions())


class TestMol:

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('polym,vecs',
                             [(None, None), (False, None),
                              (True, (5.43, 5.43, 5.43, 90.0, 90.0, 90.0))])
    def testSetUp(self, mol, polym, vecs):
        np.testing.assert_equal(mol.aids, [0, 1, 2])
        assert (False, False) == (mol.polym, mol.vecs)
        mol = structure.Mol(mol, polym=polym, vecs=vecs)
        expected = tuple(False if x is None else x for x in [polym, vecs])
        assert expected == (mol.polym, mol.vecs)
        mol = structure.Mol(mol)
        assert expected == (mol.polym, mol.vecs)

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('cnum,expected', [(1, 1), (0, 0)])
    def testGetConformers(self, emol, expected):
        assert expected == len(emol.GetConformers())

    @pytest.mark.parametrize('smiles,cnum,expected1,expected2',
                             [('O', 1, [[0, 2]], [[0, 2], [1, 5]])])
    def testAppend(self, emol, expected1, expected2):
        assert expected1 == [[x.gid, x.gids.max()] for x in emol.confs]
        emol.append(emol.confs[0])
        assert expected2 == [[x.gid, x.gids.max()] for x in emol.confs]

    @pytest.mark.parametrize('smiles,cnum,expected1,expected2',
                             [('O', 1, [[0, 2]], [[1, 5]])])
    def testShift(self, emol, expected1, expected2):
        assert expected1 == [[x.gid, x.gids.max()] for x in emol.confs]
        emol.shift(emol.confs[0])
        assert expected2 == [[x.gid, x.gids.max()] for x in emol.confs]

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('cnum,expected', [(1, 1), (2, 2)])
    def testAddConformer(self, emol, expected):
        assert expected == len(emol.confs)

    @pytest.mark.parametrize('smiles,cnum', [('O', 2)])
    @pytest.mark.parametrize('idx,expected', [(0, 0), (1, 1)])
    def testGetConformer(self, emol, idx, expected):
        emol.AddConformer(emol.confs[0])
        assert expected == emol.GetConformer(idx).gid

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('cnum,expected', [(1, 1), (0, 0)])
    def testGetNumConformers(self, emol, expected):
        assert expected == emol.GetNumConformers()

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('cnum,randomSeed,clearConfs,expected',
                             [(1, -1, True, 1), (1, 2**31, False, 2)])
    def testEmbedMolecule(self, emol, randomSeed, clearConfs, expected):
        emol.EmbedMolecule(randomSeed=randomSeed, clearConfs=clearConfs)
        assert expected == len(emol.confs)

    @pytest.mark.parametrize('smiles', ['c1=cc=cc=c1'])
    @pytest.mark.parametrize('cnum,randomSeed,numConfs,clearConfs,expected',
                             [(1, -1, 1, False, (2, True)),
                              (2, 2**31, 2, True, (2, True)),
                              (2, 1, 2, True, (2, False))])
    def testEmbedMultipleConfs(self, emol, randomSeed, numConfs, clearConfs,
                               expected):
        emol.EmbedMultipleConfs(randomSeed=randomSeed,
                                numConfs=numConfs,
                                clearConfs=clearConfs)
        xyzs1, xyzs2 = [x.GetPositions() for x in emol.confs]
        assert expected == (len(emol.confs), (xyzs1 == xyzs2).all())

    @pytest.mark.parametrize('smiles,united,expected', [('O', True, 3),
                                                        ('C', True, 1),
                                                        ('C', False, 5)])
    def testMolFromSmiles(sel, smiles, united, expected):
        mol = structure.Mol.MolFromSmiles(smiles, united=united)
        assert expected == mol.GetNumAtoms()

    @pytest.mark.parametrize(
        'smiles,expected',
        [('O', [3, 2]), ['C', [1, 0]], ['C1CCCCC1', [6, 6]]])
    def testGraph(sel, mol, expected):
        assert expected == [len(x) for x in [mol.graph.nodes, mol.graph.edges]]

    @pytest.mark.parametrize('smiles', ['CCCC'])
    @pytest.mark.parametrize('bond,expected', [((0, 1), False), ((1, 2), True),
                                               ((2, 1), True)])
    def testIsRotatable(sel, mol, bond, expected):
        assert expected == mol.isRotatable(bond)

    @pytest.mark.parametrize('smiles,expected',
                             [('CCCC', 1), ['C=C', 0], ['C1CCCCC1', 0]])
    def testRotatable(sel, mol, expected):
        assert expected == len(mol.rotatable)


class TestStruct:

    RING = structure.Mol.MolFromSmiles('C1CCCCC1')
    RING.EmbedMolecule()
    RING.EmbedMolecule(clearConfs=False)

    @pytest.fixture
    def struct(self, smiles):
        mols = [structure.Mol.MolFromSmiles(x) for x in smiles]
        for mol in mols:
            mol.EmbedMolecule()
        return structure.Struct.fromMols(mols)

    @pytest.mark.parametrize(
        'smiles,expected', [([], [1, range(2), range(12)]),
                            (['O', 'CC'], [3, range(4), range(17)])])
    def testSetUp(self, struct, expected):
        struct.setUp([self.RING])
        mol_num, gid, gids = expected
        assert mol_num == len(struct.mols)
        assert list(gid) == [x.gid for x in struct.conf]
        assert list(gids) == [y for x in struct.conf for y in x.gids]

    @pytest.mark.parametrize(
        'smiles,expected', [([], [0, range(0), range(0)]),
                            (['O', 'CC'], [2, range(2), range(5)])])
    def testFromMols(self, struct, expected):
        mol_num, gid, gids = expected
        assert mol_num == len(struct.mols)
        assert list(gid) == [x.gid for x in struct.conf]
        assert list(gids) == [y for x in struct.conf for y in x.gids]

    @pytest.mark.parametrize('smiles,expected', [([], 0), (['O', 'CC'], 2)])
    def testConformer(self, struct, expected):
        assert expected == len(list(struct.conf))

    @pytest.mark.parametrize('smiles,expected', [([], 12), (['O', 'CC'], 17)])
    def testConformer(self, struct, expected):
        struct.setUp([self.RING])
        assert expected == struct.atom_total
