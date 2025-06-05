from unittest import mock

import pytest
from rdkit import Chem

from nemd import np
from nemd import parserutils
from nemd import polymutils


@pytest.fixture
def moiety(mol):
    moiety = polymutils.Moiety(mol, info=dict(res_num=0))
    moiety.setMaids()
    return moiety


@pytest.fixture
def moieties(args):
    options = parserutils.MolBase().parse_args(args)
    for cru, cru_num, mol_num in zip(options.cru, options.cru_num,
                                     options.mol_num):
        return polymutils.Moieties(cru,
                                   cru_num=cru_num,
                                   mol_num=mol_num,
                                   options=options,
                                   logger=mock.MagicMock())


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
    def other_moiety(self, other):
        other = polymutils.Moiety.MolFromSmiles(other, info=dict(res_num=0))
        other.setMaids()
        return other

    @pytest.mark.parametrize('smiles', ['C'])
    @pytest.mark.parametrize('info,expected',
                             [(None, None),
                              (dict(res_num=1, serial=3), (1, 3))])
    def testSetup(self, mol, info, expected):
        moiety = polymutils.Moiety(mol, info=info)
        info = next(x.GetMonomerInfo() for x in moiety.GetAtoms())
        assert expected == ((info.GetResidueNumber(),
                             info.GetSerialNumber()) if info else None)

    @pytest.mark.parametrize('smiles,expected', [('C', [0]),
                                                 ('*C[*:1]', [0, 1, 2])])
    def testSetMaids(self, moiety, expected):
        assert expected == [x.GetIntProp('maid') for x in moiety.GetAtoms()]

    @pytest.mark.parametrize('smiles,other,expected',
                             [('C[*:1]', '*C[*:1]', 'CC[*:1]'),
                              ('', '*C[*:1]', 'C[*:1]'),
                              ('*C[*:1]', '', '*C')])
    def testBond(self, moiety, other_moiety, expected):
        assert expected == moiety.bond(other_moiety).smiles

    @pytest.mark.parametrize('smiles,tail,other,head,expected',
                             [('C[*:1]', None, '*C[*:1]', None,
                               ('CC[*:1]', 0, 1, 1)),
                              ('C[*:1]', 1, '*C[*:1]', 2, ('*CC', 0, 1, 1)),
                              ('', None, '*C[*:1]', None, None),
                              ('*C[*:1]', None, '', None, None)])
    def testCombine(self, moiety, tail, other_moiety, head, expected):
        chain = moiety.combine(other_moiety, tail=tail, head=head)
        if expected is None:
            assert chain is None
            return
        res = [x.GetMonomerInfo().GetResidueNumber() for x in chain.GetAtoms()]
        assert expected == (chain.smiles, *res)

    @pytest.mark.parametrize('smiles,expected', [('*C[*:1]', [2])])
    def testTail(self, moiety, expected):
        assert expected == [x.GetIdx() for x in moiety.tail]

    @pytest.mark.parametrize('smiles,expected', [('*C[*:1]', [0])])
    def testHead(self, moiety, expected):
        assert expected == [x.GetIdx() for x in moiety.head]

    @pytest.mark.parametrize('smiles,expected', [('*C[*:1]', False),
                                                 ('', True)])
    def testEmpty(self, moiety, expected):
        assert expected == moiety.empty

    @pytest.mark.parametrize('smiles,expected', [('*C[*:1]', 1)])
    def testNew(self, moiety, expected):
        new = moiety.new(info=dict(res_num=expected))
        for atom in new.GetAtoms():
            assert expected == atom.GetMonomerInfo().GetResidueNumber()

    @pytest.mark.parametrize('smiles,delta,expected', [('*C[*:1]', 2, [2, 4])])
    def testIncrRes(self, moiety, delta, expected):
        max_res = moiety.incrRes(delta=delta)
        assert expected == [max_res, moiety.incrRes(delta=delta)]

    @pytest.mark.parametrize('smiles,other,expected',
                             [('C[*:1]', '*C[*:1]', None),
                              ('', '*C[*:1]', 'C[*:1]'), ('*C[*:1]', '', '*C'),
                              ('', '', None)])
    def testCap(self, moiety, other_moiety, expected):
        capped = moiety.cap(other_moiety)
        assert expected == (capped.smiles if capped else None)

    @pytest.mark.parametrize('smiles,aids,expected',
                             [('*CCCC[*:1]', [0, 1, 2, 3], 1)])
    def testEmbedMolecule(self, moiety, aids, expected):
        moiety.EmbedMolecule()
        measured = moiety.GetConformer().measure(aids)
        np.testing.assert_almost_equal(measured % 180, 0)


class TestEditableMol:

    @pytest.fixture
    def editable(self, mol):
        return polymutils.EditableMol(mol)

    @pytest.mark.parametrize('smiles,aids,expected', [('CCCCl', (3, 0), 'CC')])
    def testRemoveAtoms(self, editable, aids, expected):
        editable.removeAtoms(aids)
        assert expected == Chem.MolToSmiles(editable.GetMol())

    @pytest.mark.parametrize('smiles,pairs,expected',
                             [('CCC*.*O*.Cl*', [(0, 6), (8, 4)], 'CCCOCl')])
    def testAddBonds(self, moiety, pairs, expected):
        for idx, aids in enumerate(pairs):
            pairs[idx] = [moiety.GetAtomWithIdx(x) for x in aids]
        editable = polymutils.EditableMol(moiety)
        chain = editable.addBonds(pairs)
        assert expected == Chem.MolToSmiles(chain.GetMol())


class TestSequence:

    @pytest.mark.parametrize(
        'args,expected', [(['*CO*', '-cru_num', '3'], '*[C]O[C]O[C]O[*:1]')])
    def testBuild(self, moieties, expected):
        mers = [moieties[0] for _ in range(moieties.options.cru_num[0])]
        assert expected == polymutils.Sequence(mers).build().smiles


class TestMoieties:

    @pytest.mark.parametrize('args,expected', [(['C'], (1, 0)),
                                               (['C.Cl'], (2, 0)),
                                               (['C*.*C*.Cl*'], (3, 7))])
    def testSetUp(self, moieties, expected):
        vals = (1 for x in moieties for y in x.GetAtoms() if y.HasProp('maid'))
        assert expected == (len(moieties), sum(vals))

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C[*:1].*C*'], 2),
                                               (['*C*'], 0)])
    def testInr(self, moieties, expected):
        assert expected == moieties.inr.GetNumAtoms()

    @pytest.mark.parametrize('args', [['*CC*.[*:1]C[*:1].C*']])
    @pytest.mark.parametrize('role,expected',
                             [('initiator', 3), ('monomer', 4),
                              ('terminator', 2), ('regular', 0)])
    def testGetMoiety(self, moieties, role, expected):
        assert expected == moieties.getMoiety(role).GetNumAtoms()

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
    def testPolym(self, moieties, expected):
        assert expected == moieties.polym.GetNumAtoms()

    @pytest.mark.parametrize('args,hashed,expected',
                             [(['*CO*', '-cru_num', '3', '-seed', '1'],
                               (0, 0, 0, 3), 1.3742332)])
    def testGetLength(self, moieties, hashed, expected):
        np.testing.assert_almost_equal(moieties.getLength(hashed), expected)
