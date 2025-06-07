import pytest

from nemd import np
from nemd import parserutils
from nemd import xtal


class TestCrystal:

    @pytest.fixture
    def xtl(self, args):
        options = parserutils.XtalBldr().parse_args(args)
        return xtal.Crystal.fromDatabase(options)

    @pytest.mark.parametrize(
        'args,expected', [(['-name', 'Si'], ['Si', 5.4307, 5.4307, 5.4307]),
                          (['-name', 'Si', '-scale_factor', '1.1', '1', '0.9'
                            ], ['Si', 5.97377, 5.4307, 4.88763])])
    def testFromDatabase(self, xtl, expected):
        assert expected == [xtl.chemical_formula, *xtl.lattice_parameters[:3]]

    @pytest.mark.parametrize(
        'args,expected', [(['-name', 'Si'], 8),
                          (['-name', 'Si', '-dimension', '2', '1', '3'], 48)])
    def testSuperCell(self, xtl, expected):
        assert expected == len(xtl.supercell.atoms)

    @pytest.mark.parametrize('args,expected',
                             [(['-name', 'Si'], [8, 2.0365125])])
    def testMol(self, xtl, expected):
        assert expected[0] == xtl.mol.GetNumAtoms()
        xyz = xtl.mol.GetConformer().GetPositions()
        np.testing.assert_almost_equal(xyz.mean(), expected[1])
