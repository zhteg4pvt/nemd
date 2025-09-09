import copy
from unittest import mock

import conftest
import numpy as np
import pytest
from rdkit import Chem

from nemd import envutils
from nemd import lmpatomic
from nemd import lmpfull
from nemd import numpyutils
from nemd import oplsua
from nemd import parserutils
from nemd import pbc
from nemd import rdkitutils

PARSER = oplsua.Parser.get()


class TestMass:

    @pytest.fixture
    def atoms(self, indices, ff=PARSER):
        return ff.atoms.loc[indices]

    @pytest.mark.parametrize('indices,expected', [([77, 78, 78], (3, 2))])
    def testFromAtoms(self, atoms, expected):
        assert expected == lmpfull.Mass.fromAtoms(atoms).shape


class TestId:

    @pytest.fixture
    def ids(self, mol):
        return lmpfull.Mol(mol).ids

    @pytest.mark.parametrize('smiles,expected', [('O', (3, 3))])
    def testFromAtoms(self, ids, expected):
        assert expected == ids.shape

    @pytest.mark.parametrize('smiles,gids,expected',
                             [('O', [4, 5, 6], (3, 3))])
    def testToNumpy(self, ids, gids, expected):
        assert expected == ids.to_numpy(np.array(gids)).shape


class TestBond:

    @pytest.fixture
    def bonds(self, mol):
        return lmpfull.Mol(mol).bonds

    @pytest.mark.parametrize('smiles,expected', [('O', (2, 3))])
    def testFromAtoms(self, bonds, expected):
        assert expected == bonds.shape

    @pytest.mark.parametrize('smiles,expected', [('O', [(0, 1), (0, 2)])])
    def testGetPair(self, bonds, expected):
        assert expected == bonds.getPairs()


class TestAngle:

    @pytest.fixture
    def angles(self, mol):
        return lmpfull.Mol(mol).angles

    @pytest.mark.parametrize('smiles,expected', [('O', (1, 4)),
                                                 ('CC(C)C', (3, 4))])
    def testFromAtoms(self, angles, expected):
        assert expected == angles.shape

    @pytest.mark.parametrize(
        'angle,improper,expected',
        [([[297, 1, 0, 2]], None, (1, 4)),
         ([[304, 0, 1, 2], [304, 0, 1, 3], [304, 2, 1, 3]], [[30, 0, 2, 1, 3]],
          (2, 4))])
    def testDropLowest(self, angle, improper, expected):
        angles = lmpfull.Angle(angle)
        imprps = lmpfull.Improper(improper) if improper else lmpfull.Improper()
        angles.dropLowest(imprps.getAngles(), PARSER.angles.ene.to_dict())
        assert expected == angles.shape

    @pytest.mark.parametrize('smiles,expected', [('O', (4, 1)),
                                                 ('CC(C)C', (4, 3))])
    def testRow(self, angles, expected):
        assert expected == angles.row.shape


class TestImproper:

    @pytest.fixture
    def impropers(self, mol):
        return lmpfull.Mol(mol).impropers

    @pytest.mark.parametrize('smiles,expected', [('O', (0, 5)),
                                                 ('CC(C)C', (1, 5))])
    def testFromAtoms(self, impropers, expected):
        assert expected == impropers.shape

    @pytest.mark.parametrize('smiles,expected', [('O', 0), ('CC(C)C', 3)])
    def testGetPairs(self, impropers, expected):
        assert expected == len(impropers.getPairs())

    @pytest.mark.parametrize('smiles,expected', [('O', 0), ('CC(C)C', 1)])
    def testGetAngles(self, impropers, expected):
        assert expected == len(impropers.getAngles())


@pytest.mark.parametrize('smiles', [('CCC(C)C')])
class TestConformer:

    @pytest.fixture
    def conf(self, mol):
        mol = lmpfull.Mol(mol)
        mol.EmbedMolecule(randomSeed=1)
        return mol.GetConformer()

    def testIds(self, conf):
        assert (5, 3) == conf.ids.shape

    def testBonds(self, conf):
        assert (4, 3) == conf.bonds.shape

    def testAngles(self, conf):
        assert (4, 4) == conf.angles.shape

    def testDihedrals(self, conf):
        assert (2, 5) == conf.dihedrals.shape

    def testImpropers(self, conf):
        assert (1, 5) == conf.impropers.shape

    @pytest.mark.parametrize('aids,val', [((2, 3), 2.12), ((0, 1, 2), 121),
                                          ((0, 1, 2, 4), 171)])
    def testSetGeo(self, conf, aids, val):
        conf.setGeo(aids, val)
        np.testing.assert_almost_equal(conf.measure(aids), val)

    @pytest.mark.parametrize('args,expected',
                             [([], None),
                              (['C'], [1.9976941, -0.034285, -0.3056107]),
                              (['CC'], 1.526), (['CC', '1.6'], 1.6),
                              (['CCC'], 112.4), (['CCC', '120'], 120),
                              (['CCCC'], 141.3842856), (['CCCC', '30'], 30)])
    def testMeasure(self, conf, smiles, args, expected):
        args = [smiles, '-substruct'] + args if args else [smiles]
        options = parserutils.MolBase().parse_args(args)
        strt = lmpfull.Struct.fromMols([conf.GetOwningMol()], options=options)
        numpyutils.assert_almost_equal(next(strt.conf).measure(), expected)


class TestMol:

    @pytest.fixture
    def fmol(self, emol):
        return lmpfull.Mol(emol)

    @pytest.mark.parametrize('smiles,cnum', [('C', 1)])
    def testFf(self, fmol):
        assert isinstance(fmol.ff, oplsua.Parser)

    @pytest.mark.parametrize('smiles,cnum,expected', [('O', 0, [77, 78, 78])])
    def testType(self, fmol, expected):
        assert expected == [x.GetIntProp('type_id') for x in fmol.GetAtoms()]

    @pytest.mark.parametrize('smiles,cnum,aids,expected',
                             [('O', 1, [0, 1], 0.9572)])
    def testSetInternal(self, fmol, aids, expected):
        length = Chem.rdMolTransforms.GetBondLength(fmol.GetConformer(), *aids)
        np.testing.assert_almost_equal(length, expected)

    @pytest.mark.parametrize('smiles,cnum', [('CCC(C)C', 1)])
    @pytest.mark.parametrize('args,expected', [([], None),
                                               (['CC', '1.6'], 1.6),
                                               (['CCC', '120'], 120),
                                               (['CCCC', '30'], 30)])
    def testSetSubstruct(self, fmol, smiles, args, expected):
        args = [smiles, '-substruct'] + args if args else [smiles]
        options = parserutils.MolBase().parse_args(args)
        struct = lmpfull.Struct.fromMols([fmol], options=options)
        numpyutils.assert_almost_equal(next(struct.conf).measure(), expected)

    @pytest.mark.parametrize('smiles,cnum', [('CCC(C)C', 1)])
    @pytest.mark.parametrize('args,expected', [([], []), (['O'], []),
                                               (['CC'], [0, 1]),
                                               (['CCC'], [0, 1, 2]),
                                               (['CCCC'], [0, 1, 2, 3])])
    def testGetSubstructMatch(self, fmol, smiles, args, expected):
        args = [smiles, '-substruct'] + args if args else [smiles]
        options = parserutils.MolBase().parse_args(args)
        struct = lmpfull.Struct.fromMols([fmol], options=options)
        match = struct.mols[0].getSubstructMatch()
        np.testing.assert_almost_equal(match.values, expected)

    @pytest.mark.parametrize('smiles,cnum,aids,expected',
                             [('O', 2, [0, 1], 0.9572)])
    def testUpdateAll(self, fmol, aids, expected):
        for conf in fmol.GetConformers():
            length = Chem.rdMolTransforms.GetBondLength(conf, *aids)
            np.testing.assert_almost_equal(length, expected)

    @pytest.mark.parametrize('smiles,cnum,expected',
                             [('O', 0, [[-0.834], [0.417], [0.417]])])
    def testCharges(self, fmol, expected):
        np.testing.assert_almost_equal(fmol.charges, expected)

    @pytest.mark.parametrize('smiles,cnum,expected', [('O', 0, (2, 3))])
    def testBonds(self, fmol, expected):
        assert expected == fmol.bonds.shape

    @pytest.mark.parametrize('smiles,cnum,expected', [('O', 0, (1, 4))])
    def testAngles(self, fmol, expected):
        assert expected == fmol.angles.shape

    @pytest.mark.parametrize('smiles,cnum,expected', [('O', 0, 1)])
    def testGetAngle(self, fmol, expected):
        assert expected == len(list(fmol.getAngle()))

    @pytest.mark.parametrize('smiles,cnum,expected', [('CCCC', 0, (1, 5))])
    def testDihedrals(self, fmol, expected):
        assert expected == fmol.dihedrals.shape

    @pytest.mark.parametrize('smiles,cnum,expected', [('CCCC', 0, 1)])
    def testGetDehedral(self, fmol, expected):
        assert expected == len(list(fmol.getDehedral()))

    @pytest.mark.parametrize('smiles,cnum,expected', [('CC(C)C', 0, (1, 5))])
    def testImpropers(self, fmol, expected):
        assert expected == fmol.impropers.shape

    @pytest.mark.parametrize('cnum', [0])
    @pytest.mark.parametrize('smiles,expected', [('CC(C)C', 1),
                                                 ('[C]N([C])[C]=O', 1)])
    def testGetImproper(self, fmol, expected):
        assert expected == len(list(fmol.getImproper()))

    @pytest.mark.parametrize('cnum', [0])
    @pytest.mark.parametrize('smiles,expected', [('CC(C)C', 58.124),
                                                 ('[C]N([C])[C]=O', 73.095)])
    def testMolecularWeight(self, fmol, expected):
        assert expected == fmol.molecular_weight

    @pytest.mark.parametrize('cnum', [0])
    @pytest.mark.parametrize('smiles,expected', [('CC(C)C', (4, 1)),
                                                 ('OC(C)CC', (6, 1)),
                                                 ('[C]N([C])[C]=O', (5, 1))])
    def testNbrCharge(self, fmol, expected):
        assert expected == fmol.nbr_charge.shape


class TestScript:

    @pytest.fixture
    def script(self, smiles, emol):
        options = parserutils.AmorpBldr().parse_args([smiles])
        return lmpfull.Struct.fromMols([emol], options=options).script

    @pytest.mark.parametrize('smiles,cnum', [('O', 1)])
    def testSetup(self, script):
        script.setup()
        assert 7 == len(script)

    @pytest.mark.parametrize('cnum', [1])
    @pytest.mark.parametrize('smiles,expected', [('O', 3), ('C', 2)])
    def testPair(self, script, expected):
        script.pair()
        assert expected == len(script)

    @pytest.mark.parametrize('cnum', [1])
    @pytest.mark.parametrize('smiles,expected', [('O', 0)])
    def testCoeff(self, script, expected):
        script.coeff()
        assert expected == len(script)

    @pytest.mark.parametrize('smiles', ['CCCC'])
    @pytest.mark.parametrize('substruct,cnum,expected', [
        (None, 1, None),
        (['CCC'], 1, None),
        (['CCC', 120], 2, [
            'fix rest all restrain angle 1 2 3 2000 2000 120',
            'minimize 1.0e-6 1.0e-6 1000 10000'
        ]),
        (['CCC', 120], 1, [
            'fix rest all restrain angle 1 2 3 2000 2000 120',
            'minimize 1.0e-6 1.0e-6 1000000 10000000'
        ]),
        (['CCCC', 120], 1, [
            'fix rest all restrain dihedral 1 2 3 4 -2000 -2000 120',
            'minimize 1.0e-6 1.0e-6 1000000 10000000'
        ]),
    ])
    def testMinimize(self, script, substruct, expected):
        script.struct.options.substruct = substruct
        script.minimize()
        assert all(x in script
                   for x in expected) if expected else (2 == len(script))

    @pytest.mark.parametrize('cnum', [1])
    @pytest.mark.parametrize(
        'smiles,expected',
        [('C', None), ('O', 'fix rigid all shake 0.0001 10 10000 b 1 a 1')])
    def testMinimizeShake(self, script, expected):
        script.minimize()
        assert (expected in script) if expected else (2 == len(script))


class TestStruct:

    @pytest.fixture
    def struct(self, smiless, logger):
        options = parserutils.AmorpBldr().parse_args(
            [*smiless, '-JOBNAME', 'name'])
        mols = [lmpfull.Mol.MolFromSmiles(x) for x in smiless]
        for cnum, mol in enumerate(mols):
            with rdkitutils.capture_logging():
                mol.EmbedMultipleConfs(numConfs=cnum)
        return lmpfull.Struct.fromMols(mols, options=options, logger=logger)

    @pytest.mark.parametrize('smiless,expected',
                             [(['O'], [216, 151, 310, 631, 76])])
    def testInit(self, struct, expected):
        struct_types = [
            struct.atm_types, struct.bnd_types, struct.ang_types,
            struct.dihe_types, struct.impr_types
        ]
        assert expected == [len(x) for x in struct_types]

    @pytest.mark.parametrize('smiless,expected',
                             [(['O'], [2, 1, 1, 0, 0]),
                              (['CC(C)CC'], [4, 2, 2, 1, 1])])
    def testSetTypeMap(self, struct, expected):
        ons = [
            struct.atm_types.on, struct.bnd_types.on, struct.ang_types.on,
            struct.dihe_types.on, struct.impr_types.on
        ]
        assert expected == [len(x) for x in ons]

    @pytest.mark.parametrize('smiless,expected', [(['O', '[Na+]'], [
        'WARNING: The system has a net charge of 1.0000',
        'Data file written into name.data', 'In script written into name.in'
    ])])
    def testWrite(self, struct, expected, tmp_dir):
        struct.write()
        struct.logger.log.assert_has_calls([mock.call(x) for x in expected])

    @pytest.mark.parametrize('smiless,expected', [(['O', 'O'], 47),
                                                  (['[Ar]', 'O'], 49)])
    def testWriteData(self, struct, expected, tmp_dir):
        struct.writeData()
        with open(struct.outfile) as fh:
            assert expected == len(fh.readlines())

    @pytest.mark.parametrize('smiless,expected', [(['O', 'CC(C)CC'], (5, 7))])
    def testAtoms(self, struct, expected):
        assert expected == struct.atoms.shape

    @pytest.mark.parametrize(
        'smiless,expected',
        [(['O'], []), (['O', 'CC(C)CC'], [0.0]),
         (['O', 'CC(C)CC', 'C(=O)O'], [0.0, -0.5, -0.58, 0.55, 0.45])])
    def testCharges(self, struct, expected):
        assert set(expected) == set(struct.charges.flatten().tolist())

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 3)),
                                                  (['O', 'CC(C)CC'], (4, 3))])
    def testBonds(self, struct, expected):
        assert expected == struct.bonds.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 4)),
                                                  (['O', 'CC(C)CC'], (4, 4))])
    def testAngles(self, struct, expected):
        assert expected == struct.angles.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 5)),
                                                  (['O', 'CC(C)CC'], (2, 5))])
    def testDihedrals(self, struct, expected):
        assert expected == struct.dihedrals.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 5)),
                                                  (['O', 'CC(C)CC'], (1, 5))])
    def testImpropers(self, struct, expected):
        assert expected == struct.impropers.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (1, 2)),
                                                  (['O', 'CC(C)CC'], (6, 2))])
    def testMasses(self, struct, expected):
        assert expected == struct.masses.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (1, 2)),
                                                  (['O', 'CC(C)CC'], (6, 2))])
    def testPairCoeffs(self, struct, expected):
        assert expected == struct.pair_coeffs.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 2)),
                                                  (['O', 'CC(C)CC'], (3, 2))])
    def testBondCoeffs(self, struct, expected):
        assert expected == struct.bond_coeffs.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 2)),
                                                  (['O', 'CC(C)CC'], (3, 2))])
    def testAngleCoeffs(self, struct, expected):
        assert expected == struct.angle_coeffs.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 4)),
                                                  (['O', 'CC(C)CC'], (1, 4))])
    def testDihedralCoeffs(self, struct, expected):
        assert expected == struct.dihedral_coeffs.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], (0, 3)),
                                                  (['O', 'CC(C)CC'], (1, 3))])
    def testImproperCoeffs(self, struct, expected):
        assert expected == struct.improper_coeffs.shape

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], 0),
                                                  (['O', 'CC(C)CC'], 72.151)])
    def testMolecularWeight(self, struct, expected):
        assert expected == struct.molecular_weight

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], False),
                                                  (['[Ar]', 'O'], True)])
    def testHasCharge(self, struct, expected):
        assert expected == struct.hasCharge()

    @pytest.mark.parametrize('smiless,expected', [(['[Ar]'], ''),
                                                  (['[Ar]', 'O'], '1')])
    def testGetRigid(self, struct, expected):
        assert expected == struct.getRigid(struct.bnd_types, struct.ff.bonds)

    @pytest.mark.parametrize(
        'smiless,edge,expected',
        [(['[Na+]', 'O'], None, None), (['[Na+]', 'O'], 40, None),
         (['[Na+]', 'O'], 10, 'Box span / 2 (5.00 Å) < 11.00 Å (cutoff)'),
         (['[Ar]', '[Na+]'], None, 'The system has a net charge of 1.0000')])
    def testGetWarnings(self, struct, edge, expected):
        if edge:
            struct.box = pbc.Box.fromParams(edge)
        assert expected == next(struct.getWarnings(), None)

    @pytest.mark.parametrize('smiles,args,expected',
                             [('O', None, 'TIP3P'),
                              ('O', ['-force_field', 'OPLSUA', 'SPC'], 'SPC')])
    def testFf(self, mol, smiles, args, expected, options=None):
        if args:
            options = parserutils.MolBase().parse_args([smiles] + args)
        struct = lmpfull.Struct.fromMols([mol], options=options)
        assert expected == struct.ff.wmodel


@conftest.require_src
class TestReader:

    @pytest.fixture
    def rdr(self, args):
        return lmpfull.Reader.fromTest(*args)

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (4, 2))])
    def testPairCoeffs(self, rdr, expected):
        assert expected == rdr.pair_coeffs.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (3, 2)),
                              (['0000', 'original.data'], (0, 2))])
    def testBondCoeffs(self, rdr, expected):
        assert expected == rdr.bond_coeffs.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (2, 2)),
                              (['0000', 'original.data'], (0, 2))])
    def testAngleCoeffs(self, rdr, expected):
        assert expected == rdr.angle_coeffs.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (2, 4)),
                              (['0000', 'original.data'], (0, 4))])
    def testDihedralCoeffs(self, rdr, expected):
        assert expected == rdr.dihedral_coeffs.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (1, 3)),
                              (['0000', 'original.data'], (0, 3))])
    def testImproperCoeffs(self, rdr, expected):
        assert expected == rdr.improper_coeffs.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (5, 3)),
                              (['0000', 'original.data'], (0, 3))])
    def testBonds(self, rdr, expected):
        assert expected == rdr.bonds.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (4, 4)),
                              (['0000', 'original.data'], (0, 4))])
    def testAngles(self, rdr, expected):
        assert expected == rdr.angles.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (4, 5)),
                              (['0000', 'original.data'], (0, 5))])
    def testDihedrals(self, rdr, expected):
        assert expected == rdr.dihedrals.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], (1, 5)),
                              (['0000', 'original.data'], (0, 5))])
    def testImpropers(self, rdr, expected):
        assert expected == rdr.impropers.shape

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], 1),
                              (['0000', 'original.data'], 10)])
    def testMols(self, rdr, expected):
        assert expected == len(rdr.mols)

    @pytest.mark.parametrize('args,expected',
                             [(['0022_test', 'amorp_bldr.data'], 86.134),
                              (['0000', 'original.data'], 160.43)])
    def testMolecularWeight(self, rdr, expected):
        np.testing.assert_almost_equal(rdr.molecular_weight, expected)

    @pytest.mark.parametrize('args', [(['0022_test', 'amorp_bldr.data'])])
    @pytest.mark.parametrize('attr', [
        None, 'pair_coeffs', 'bond_coeffs', 'angle_coeffs', 'dihedral_coeffs',
        'improper_coeffs', 'atoms', 'bonds', 'angles', 'dihedrals', 'impropers'
    ])
    def testAllClose(self, rdr, attr):
        other = copy.deepcopy(rdr)
        if attr:
            attr = getattr(rdr, attr)
            attr += 1
        assert (attr is None) == rdr.allClose(other)

    @pytest.mark.parametrize(
        'args,expected', [(['0022_test', 'amorp_bldr.data'], lmpfull.Reader),
                          (['si', 'crystal_builder.data'], lmpatomic.Reader),
                          (['0000', 'check'], ValueError)])
    def testRead(self, args, expected, raises):
        with raises:
            rdr = lmpfull.Reader.read(envutils.test_data(*args))
            assert isinstance(rdr, expected)
