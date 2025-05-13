import copy
import os
import types

import numpy as np
import pytest

from nemd import envutils
from nemd import lmpatomic
from nemd import numpyutils
from nemd import parserutils
from nemd import table


class TestBase:

    @pytest.fixture
    def base(self):
        return lmpatomic.Base()

    def testWriteCount(self, base, tmp_dir):
        with open('file', 'w') as fh:
            base.writeCount(fh)
        assert os.path.exists('file')

    @pytest.mark.parametrize('lines,expected', [(['1 2'], [0, 2])])
    def testFromLines(self, base, lines, expected):
        base = base.fromLines(lines)
        assert expected == [*base.index, *base.values]


class TestMass:

    ATOM = table.TABLE.reset_index()

    @pytest.fixture
    def masses(self, indices):
        return lmpatomic.Mass.fromAtoms(self.ATOM.iloc[indices])

    @pytest.mark.parametrize('indices', [([1, 2]), ([4, 2, 112])])
    def testFromAtoms(self, masses, indices):
        assert len(indices) == masses.shape[0]

    @pytest.mark.parametrize('indices,expected',
                             [([1, 2], ['H', 'He']),
                              ([4, 2, 112], ['Be', 'He', 'Cn'])])
    def testElement(self, masses, expected):
        np.testing.assert_equal(masses.element, expected)

    @pytest.mark.parametrize('indices', [([1, 2]), ([4, 2, 112])])
    def testWrite(self, masses, tmp_dir):
        with open('file', 'w') as fh:
            masses.write(fh)
        assert os.path.isfile('file')

    @pytest.mark.parametrize('lines',
                             [(['1 1.0080 # H #\n', '2 4.0030 # He #\n'])])
    def testFromLines(self, lines):
        assert len(lines) == lmpatomic.Mass.fromLines(lines).shape[0]


class TestId:

    @pytest.fixture
    def ids(self, tmol):
        return lmpatomic.Id.fromAtoms(tmol.GetAtoms())

    @pytest.mark.parametrize('smiles,expected',
                             [('O', [[0, 77], [1, 78], [2, 78]])])
    def testFromAtoms(self, ids, expected):
        np.testing.assert_equal(ids.values, expected)

    @pytest.mark.parametrize('smiles', ['O'])
    @pytest.mark.parametrize('gids,expected', [([0, 1, 2], [0, 1, 2]),
                                               ([11, 12, 13], [11, 12, 13])])
    def testToNumpy(self, ids, gids, expected):
        gids = ids.to_numpy(np.array(gids))[:, 0]
        np.testing.assert_equal(gids, expected)

    @pytest.mark.parametrize(
        'arrays,on,expected',
        [(([[1, 2]], [[3, 4]]), [1, 2, 4], [[1, 1], [3, 2]])])
    def testConcatenate(self, arrays, on, expected):
        type_map = numpyutils.IntArray(on=on)
        ids = lmpatomic.Id.concatenate(arrays, type_map)
        np.testing.assert_equal(ids.values, expected)


class TestAtom:

    @pytest.mark.parametrize('atomic,expected', [([[0, 0, 0.1, 0.2, 0.3]], 1)])
    def testWrite(self, atomic, expected, tmp_dir):
        with open('file', 'w') as fh:
            lmpatomic.Atom(atomic).write(fh)
        assert os.path.isfile('file')


class TestConformer:

    @pytest.mark.parametrize(
        'smiles,expected',
        [('O', [[0, 8], [1, 1], [2, 1], [3, 8], [4, 1], [5, 1]])])
    def testIds(self, smiles, expected):
        mol = lmpatomic.Mol.MolFromSmiles(smiles)
        mol.EmbedMolecule()
        mol.EmbedMolecule(clearConfs=False)
        ids = np.concatenate([x.ids for x in mol.GetConformers()])
        np.testing.assert_equal(ids, expected)


class TestMol:

    @pytest.fixture
    def mol(self, smiles):
        return lmpatomic.Mol.MolFromSmiles(smiles)

    @pytest.mark.parametrize('smiles,expected', [('O', [8, 1, 1])])
    def testType(self, mol, expected):
        assert expected == [x.GetIntProp('type_id') for x in mol.GetAtoms()]

    @pytest.mark.parametrize('smiles,expected',
                             [('O', [[0, 8], [1, 1], [2, 1]])])
    def testIds(self, mol, expected):
        np.testing.assert_equal(mol.ids.values, expected)

    @pytest.mark.parametrize('smiles,class_type', [('[Si]', str),
                                                   ('C', types.NoneType)])
    def testFf(self, mol, smiles, class_type):
        options = parserutils.MolBase().parse_args([smiles])
        struct = lmpatomic.Struct.fromMols([mol], options=options)
        assert isinstance(struct.mols[0].ff, class_type)


@pytest.mark.parametrize('smiles', ['[Si]'])
class TestStruct:

    @pytest.fixture
    def struct(self, emol):
        return lmpatomic.Struct.fromMols([emol])

    @pytest.mark.parametrize('cnum,expected', [(1, [14])])
    def testSetTypeMap(self, struct, expected):
        np.testing.assert_equal(struct.atm_types.on, expected)

    @pytest.mark.parametrize('cnum,expected', [(0, (0, 5)), (1, (1, 5)),
                                               (2, (2, 5))])
    def testAtoms(self, struct, expected):
        assert expected == struct.atoms.shape

    @pytest.mark.parametrize('cnum,expected', [(0, 0), (1, 1), (2, 2)])
    def testGetAtomic(self, struct, expected):
        assert expected == len(list(struct.getAtomic()))

    @pytest.mark.parametrize('cnum,expected', [(0, (0, 2)), (1, (1, 2)),
                                               (2, (2, 2))])
    def testIds(self, struct, expected):
        assert expected == struct.ids.shape

    @pytest.mark.parametrize('cnum,expected', [(0, 0), (1, 1), (2, 2)])
    def testGetPositions(self, struct, expected):
        assert expected == len(struct.GetPositions())

    @pytest.mark.parametrize('cnum,expected', [(0, str)])
    def testFf(self, struct, expected):
        assert isinstance(struct.ff, expected)


@pytest.mark.parametrize(
    'data_file', [envutils.test_data('0033_test', 'crystal_builder.data')])
class TestReader:

    @pytest.fixture
    def rdr(self, data_file):
        return lmpatomic.Reader(data_file)

    def testLines(self, rdr):
        assert 3 == len(rdr.lines)

    @pytest.mark.parametrize('line', ['Masses', 'Atoms'])
    def testNameRe(self, rdr, line):
        assert rdr.name_re.match(line)

    @pytest.mark.parametrize('line', ['48 atoms', '1 atom types'])
    def testCountRe(self, rdr, line):
        assert rdr.count_re.match(line)

    @pytest.mark.parametrize(
        'line', ['0 5.1592 xlo xhi', '0 10.3183 ylo yhi', '0 15.4775 zlo zhi'])
    def testBoxRe(self, rdr, line):
        assert rdr.box_re.match(line)

    @pytest.mark.parametrize('line', ['0.0000 0.0000 0.0000 xy xz yz'])
    def testTiltRe(self, rdr, line):
        assert rdr.tilt_re.match(line)

    def testBox(self, rdr):
        assert (3, 4) == rdr.box.shape
        assert [0, 0, 0] == rdr.box.tilt

    def testMasses(self, rdr):
        assert (1, 2) == rdr.masses.shape

    def testElements(self, rdr):
        assert (48, 1) == rdr.elements.shape

    def testAtoms(self, rdr):
        assert (48, 5) == rdr.atoms.shape

    def testFromLines(self, rdr):
        assert (48, 4) == rdr.fromLines(lmpatomic.Atom).shape

    @pytest.mark.parametrize('atol,rtol,expected', [(1e-08, 1e-05, False),
                                                    (1e-06, 1e-05, True)])
    def testFromLines(self, rdr, atol, rtol, expected):
        other = copy.deepcopy(rdr)
        other.box.lo = 1e-07
        assert expected == rdr.allClose(other, atol=atol, rtol=rtol)

    def testGetStyle(self, rdr):
        assert 'atomic' == rdr.getStyle(rdr.data_file)
