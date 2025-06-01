from unittest import mock

import pytest

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
