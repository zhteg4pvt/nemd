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
                                       options=options)

    @pytest.mark.parametrize('args,expected', [(['C'], 1), (['C.Cl'], 2),
                                               (['C*.*C*.Cl*'], 3)])
    def testSetUp(self, moieties, expected):
        assert expected == len(moieties)

    @pytest.mark.parametrize('args,expected', [(['C'], 1), (['C.Cl'], 2),
                                               (['*C*'], 1)])
    def testRun(self, moieties, expected):
        moieties.run()
        assert expected == sum(x.GetNumConformers() for x in moieties.mols)

    @pytest.mark.parametrize('args,expected', [(['C'], 1), (['C.Cl'], 2),
                                               (['*C*'], 0)])
    def testMols(self, moieties, expected):
        assert expected == len(moieties.mols)

    @pytest.mark.parametrize('args,expected', [(['*C*'], [0, 1, 2])])
    def testSetMaids(self, moieties, expected):
        moieties.setMaids()
        maids = [y.GetIntProp('maid') for x in moieties for y in x.GetAtoms()]
        assert expected == maids

    @pytest.mark.parametrize('args,expected', [(['*C*', '-cru_num', '3'], 3)])
    def testGetSeq(self, moieties, expected):
        assert expected == len(moieties.getSeq())

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C.Cl'], 0),
                                               (['*C*'], 1)])
    def testMers(self, moieties, expected):
        assert expected == len(moieties.mers)

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C[*:1].*C*'], 2),
                                               (['*C*'], 0)])
    def testInr(self, moieties, expected):
        assert expected == moieties.inr.GetNumAtoms()

    @pytest.mark.parametrize('args,expected', [(['C'], 0), (['C*.*C*'], 2),
                                               (['*C*'], 0)])
    def testTer(self, moieties, expected):
        assert expected == moieties.ter.GetNumAtoms()

    @pytest.mark.parametrize('args,hashed,expected',
                             [(['*CO*', '-cru_num', '3', '-seed', '1'],
                               (0, 1, 0, 2), 1.3977891330176333)])
    def testGetLength(self, moieties, hashed, expected):
        np.testing.assert_almost_equal(moieties.getLength(hashed), expected)
