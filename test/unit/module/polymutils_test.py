from unittest import mock

import pytest
from rdkit import Chem

from nemd import np
from nemd import parserutils
from nemd import polymutils


@pytest.fixture
def moieties(args):
    options = parserutils.MolBase().parse_args(args)
    for cru, cru_num, mol_num in zip(options.cru, options.cru_num,
                                     options.mol_num):
        return polymutils.Moieties(cru,
                                   cru_num=cru_num,
                                   mol_num=mol_num,
                                   options=options)


class TestBond:

    @pytest.fixture
    def bond(self, mol):
        return polymutils.Bond(mol.GetBonds()[0])

    @pytest.mark.parametrize('smiles,idx,expected',
                             [('Cl*.*C[*:1]', 0, (0, 2, 1, 0)),
                              ('CC*.*CC[*:1]', 2, (0, 0, 1, 3))])
    def testHash(self, smiles, idx, expected):
        bond = polymutils.Moieties(smiles).polym.GetBonds()[idx]
        assert expected == polymutils.Bond(bond).hash

    @pytest.mark.parametrize('smiles,expected', [('Cl*.*C[*:1]', 3)])
    def testBegin(self, bond, expected):
        bond.begin = expected
        assert expected == bond.begin

    @pytest.mark.parametrize('smiles,expected', [('Cl*.*C[*:1]', 3)])
    def testEnd(self, bond, expected):
        bond.end = expected
        assert expected == bond.end

    @pytest.mark.parametrize('smiles,expected', [('Cl*.*C[*:1]', [1.1, 2, 3])])
    def testVec(self, bond, expected):
        bond.vec = expected
        assert expected == bond.vec

    @pytest.mark.parametrize('smiles,expected', [('Cl*.*C[*:1]', [0, 1.2, 3])])
    def testXyz(self, bond, expected):
        bond.xyz = expected
        assert expected == bond.xyz


@pytest.mark.parametrize('smiles', ['*CC*'])
class TestConf:

    @pytest.fixture
    def conf(self, smiles):
        chain = polymutils.Moiety.MolFromSmiles(smiles)
        chain.EmbedMolecule()
        return chain.GetConformer()

    @pytest.mark.parametrize('bond,prop,aids',
                             [(None, None, [0, 1, 2, 3]),
                              (0, (0, [1, 2, 3], [-1, 1, 2]), [1])])
    def testGetAligned(self, conf, bond, prop, aids):
        if bond is not None:
            bond = polymutils.Bond(conf.mol.GetBondWithIdx(bond))
            bond.end, bond.xyz, bond.vec = prop
        xyz = conf.getAligned(bond=bond)
        if bond:
            vec = xyz[bond.end] - xyz[aids]
            np.testing.assert_almost_equal(np.cross(vec, bond.vec), 0)
            np.testing.assert_almost_equal(xyz[aids][0], bond.xyz)
        else:
            np.testing.assert_almost_equal(xyz[aids].mean(), 0)

    @pytest.mark.parametrize('cap,aids', [(None, [0, 1, 2, 3]), (0, [1])])
    def testTranslated(self, conf, cap, aids):
        xyz = conf.translated(cap=cap)[aids]
        np.testing.assert_almost_equal(xyz.mean(), 0)


class TestMoiety:

    @pytest.fixture
    def moiety(self, mol, info):
        return polymutils.Moiety(mol, info=info)

    @pytest.mark.parametrize('smiles', ['C'])
    @pytest.mark.parametrize('info,expected',
                             [(None, None),
                              (dict(res_num=1, serial=3), (1, 3))])
    def testSetup(self, moiety, expected):
        info = next(x.GetMonomerInfo() for x in moiety.GetAtoms())
        if info:
            info = (info.GetResidueNumber(), info.GetSerialNumber())
        assert expected == info

    @pytest.mark.parametrize('smiles,info,expected', [('*C[*:1]', None, [0])])
    def testHead(self, moiety, expected):
        assert expected == [x.GetIdx() for x in moiety.head]

    @pytest.mark.parametrize('smiles,info,expected', [('*C[*:1]', None, [2])])
    def testTail(self, moiety, expected):
        assert expected == [x.GetIdx() for x in moiety.tail]

    @pytest.mark.parametrize('info', [dict(res_num=0)])
    @pytest.mark.parametrize('smiles,other,expected',
                             [('*C[*:1]', '*C*', '*CC*'), ('', '*C*', '*C'),
                              ('*C[*:1]', '', '*C')])
    def testBond(self, moiety, other, expected):
        mol = polymutils.Moiety.MolFromSmiles(other, info=dict(res_num=0))
        assert expected == moiety.bond(mol).smiles

    @pytest.mark.parametrize('smiles,info,expected',
                             [('*C[*:1]', dict(res_num=0), 1)])
    def testNew(self, moiety, expected):
        nmoiety = moiety.new(info=dict(res_num=expected))
        for atom in nmoiety.GetAtoms():
            assert expected == atom.GetMonomerInfo().GetResidueNumber()

    @pytest.mark.parametrize('smiles,info,expected',
                             [('*CCCC[*:1]', dict(res_num=0), 1)])
    def testEmbedMolecule(self, moiety, expected):
        moiety.EmbedMolecule()
        value = moiety.GetConformer().measure([0, 1, 2, 3]) % 180
        np.testing.assert_almost_equal(value, 0)


class TestEditableMol:

    @pytest.fixture
    def editable(self, mol):
        return polymutils.EditableMol(mol)

    @pytest.mark.parametrize('smiles,aids,expected', [('CCCCl', (3, 0), 'CC')])
    def testRemoveAtoms(self, editable, aids, expected):
        editable.removeAtoms(aids)
        assert expected == Chem.MolToSmiles(editable.GetMol())

    @pytest.mark.parametrize('smiles,pairs,expected',
                             [('CCC*.*O*.Cl*', ((0, 6), (8, 4)), 'CCCOCl')])
    def testAddBonds(self, editable, pairs, expected):
        mol = editable.GetMol()
        pairs = [[mol.GetAtomWithIdx(y) for y in x] for x in pairs]
        for idx, cap in enumerate(y for x in pairs for y in x):
            cap.SetIntProp('maid', idx)
        editable.addBonds(pairs)
        assert expected == Chem.MolToSmiles(editable.GetMol())


class TestSequence:

    @pytest.fixture
    def seq(self, moieties):
        return moieties.getSeq()

    @pytest.mark.parametrize('args,expected',
                             [(['*CO*', '-cru_num', '3', '-seed', '1'], 8)])
    def testBuild(self, seq, expected):
        assert expected == seq.build().GetNumAtoms()


class TestMoieties:

    @pytest.fixture
    def moieties(self, args):
        options = parserutils.MolBase().parse_args(args)
        for cru, cru_num, mol_num in zip(options.cru, options.cru_num,
                                         options.mol_num):
            return polymutils.Moieties(cru,
                                       cru_num=cru_num,
                                       mol_num=mol_num,
                                       options=options,
                                       logger=mock.MagicMock())

    @pytest.mark.parametrize('args,expected', [(['C'], (1, 0)),
                                               (['C.Cl'], (2, 0)),
                                               (['C*.*C*.Cl*'], (3, 7))])
    def testSetUp(self, moieties, expected):
        atoms = [y for x in moieties for y in x.GetAtoms()]
        amid_num = len([x for x in atoms if x.HasProp('maid')])
        assert expected == (len(moieties), amid_num)

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C[*:1].*C*'], 2),
                                               (['*C*'], 0)])
    def testInr(self, moieties, expected):
        assert expected == moieties.inr.GetNumAtoms()

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C*.*C*'], 2),
                                               (['*C*'], 0)])
    def testTer(self, moieties, expected):
        assert expected == moieties.ter.GetNumAtoms()

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C.Cl'], 0),
                                               (['*C*'], 1)])
    def testMers(self, moieties, expected):
        assert expected == len(moieties.mers)

    @pytest.mark.parametrize('args,expected', [(['C'], 1), (['C.Cl'], 2),
                                               (['*C*'], 1)])
    def testRun(self, moieties, expected):
        moieties.run()
        assert expected == sum(x.GetNumConformers() for x in moieties.mols)

    @pytest.mark.parametrize('args,expected', [(['C'], 1), (['C.Cl'], 2),
                                               (['*C*'], 0)])
    def testMols(self, moieties, expected):
        assert expected == len(moieties.mols)

    @pytest.mark.parametrize('args,expected', [(['*C*', '-cru_num', '3'], 3)])
    def testGetSeq(self, moieties, expected):
        assert expected == len(moieties.getSeq())

    @pytest.mark.parametrize('args,hashed,expected',
                             [(['*CO*', '-cru_num', '3', '-seed', '1'],
                               (0, 1, 0, 2), 1.3977891330176333)])
    def testGetLength(self, moieties, hashed, expected):
        np.testing.assert_almost_equal(moieties.getLength(hashed), expected)
