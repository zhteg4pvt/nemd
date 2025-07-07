import itertools
from unittest import mock

import numpy as np
import pytest
import scipy
from rdkit import Chem

from nemd import envutils
from nemd import parserutils
from nemd import structutils


@pytest.mark.parametrize('smiles,cnum,seed', [('O', 1, 0)])
class TestConf:

    @pytest.fixture
    def conf(self, smol):
        return structutils.GriddedMol(smol).GetConformer()

    @pytest.mark.parametrize('aids,ignoreHs,expected',
                             [(None, False, [-0.0254707, -0.0234309, 0.]),
                              ([0, 1], False, [-0.3970275, 0.0979114, 0.]),
                              (None, True, [-0.00081616, 0.36637843, 0.])])
    def testCentroid(self, conf, aids, ignoreHs, expected):
        centroid = conf.centroid(aids=aids, ignoreHs=ignoreHs)
        np.testing.assert_almost_equal(centroid, expected)

    @pytest.mark.parametrize('axis,angle,expected', [('x', 180, [1, -1, -1]),
                                                     ('z', 180, [-1, -1, 1])])
    def testRotate(self, conf, axis, angle, expected):
        oxyz = conf.GetPositions()
        rotation = scipy.spatial.transform.Rotation.from_euler(axis, [angle],
                                                               degrees=True)
        conf.rotate(rotation)
        xyz = conf.GetPositions()
        for idx, factor in enumerate(expected):
            np.testing.assert_almost_equal(oxyz[:, idx], factor * xyz[:, idx])

    @pytest.mark.parametrize('vec', [(1, 2, 3)])
    def testTranslate(self, conf, vec):
        oxyz = conf.GetPositions()
        conf.translate(vec)
        xyz = conf.GetPositions()
        np.testing.assert_almost_equal(oxyz, xyz - vec)


@pytest.mark.parametrize('smiles,cnum,seed', [('O', 2, 0)])
class TestPackedConf:

    @pytest.fixture
    def conf(self, smol, random_seed):
        struct = structutils.PackedStruct.fromMols([smol])
        struct.setFrame()
        struct.dist.set([0, 1, 2])
        return next(itertools.islice(struct.conf, 1, 2))

    def testSetConformer(self, conf):
        conf.setConformer()
        assert conf.mol.struct.dist.gids.all()
        np.testing.assert_almost_equal(conf.GetPositions().mean(), 3.0663129)

    @pytest.mark.parametrize('gids', [([2])])
    def testCheckClash(self, conf, gids):
        assert conf.checkClash([3, 4, 5]) is None
        conf.setConformer()
        assert conf.checkClash([3, 4, 5]) is True

    @pytest.mark.parametrize('aids', [([0, 1])])
    def testRotateRandomly(self, conf, aids):
        oxyz = conf.GetPositions()
        obond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        conf.rotateRandomly()
        assert not (oxyz == conf.GetPositions()).all()
        bond = Chem.rdMolTransforms.GetBondLength(conf, *aids)
        np.testing.assert_almost_equal(bond, obond)

    def testRest(self, conf):
        conf.setConformer()
        oxyz = conf.GetPositions()
        conf.reset()
        assert not (oxyz == conf.GetPositions()).all()


@pytest.mark.parametrize('smiles,cnum,seed', [('CCCCC(CC)CC', 2, 0)])
class TestGrownConf:

    @pytest.fixture
    def conf(self, smol, random_seed):
        struct = structutils.GrownStruct.fromMols([smol])
        struct.setFrame()
        struct.dist.set([0, 1, 2])
        return next(itertools.islice(struct.conf, 1, 2))

    def testSetUp(self, conf):
        np.testing.assert_almost_equal(conf.init, [9, 10, 11])
        assert 5 == len(list(conf.frag.next()))
        assert 1 == len(conf.frags)

    def testGrow(self, conf):
        while conf.frags:
            conf.grow()
        assert 1 == conf.failed
        assert conf.mol.struct.dist.gids[conf.gids].all()

    def testSetConformer(self, conf):
        conf.setConformer()
        assert 6 == conf.mol.struct.dist.gids.on.size

    def testCentroid(self, conf):
        np.testing.assert_almost_equal(conf.centroid(),
                                       [2.64317873, -0.42000759, -0.21083656])

    @pytest.mark.parametrize('dihe,expected', [((0, 1, 2, 3), 6),
                                               ((1, 2, 3, 4), 5)])
    def testGetSwingAtoms(self, conf, dihe, expected):
        assert expected == len(conf.getSwingAtoms(dihe))

    def testReset(self, conf):
        while conf.frags:
            conf.grow()
        conf.reset()
        assert 0 == conf.failed
        assert len(conf.frags)
        assert 36 == len(conf.frag.vals)


@pytest.mark.parametrize('smiles,cnum,seed', [('O', 2, 0)])
class TestGriddedMol:

    @pytest.fixture
    def grid(self, smol):
        return structutils.GriddedMol(smol)

    @pytest.mark.parametrize('size,expected', [([10, 10, 10], 4)])
    def testSetBox(self, grid, size, expected):
        grid.setBox(np.array(size))
        assert expected == np.prod(grid.num) == len(grid.vecs)

    @pytest.mark.parametrize('size,num,expected', [([10, 10, 10], 20, (2, 19)),
                                                   ([6, 5, 5], 20, (2, 18))])
    def testSetConformers(self, grid, size, num, expected):
        grid.setBox(np.array(size))
        unused = grid.setConformers(np.random.rand(num, 3))
        assert expected == (len(grid.vectors), len(unused))

    def testSize(self, grid):
        np.testing.assert_almost_equal(grid.size, [5.5108817, 4.6324939, 4.])


class TestPackedMol:

    @pytest.mark.parametrize('smiles,cnum,expected', [('O', 2, (3, 3))])
    def testSetUp(self, emol, expected):
        for conf in structutils.PackedMol(emol).confs:
            assert expected == conf.oxyz.shape


@pytest.mark.parametrize('cnum,seed', [(2, 0)])
class TestGrownMol:

    @pytest.fixture
    def grown(self, smol, random_seed):
        return structutils.GrownMol(smol)

    @pytest.mark.parametrize('smiles', ['CCCCC(CC)CC'])
    def testSetUp(self, grown):
        assert all(x.frag for x in grown.confs)

    @pytest.mark.parametrize('smiles,expected', [('CCCCC(CC)CC', 18)])
    def testShift(self, grown, expected):
        grown.shift(grown.confs[0])
        gids = [y for x in grown.confs for y in x.init]
        gids += [z for x in grown.confs for y in x.frag.next() for z in y.ids]
        assert set(range(expected, expected * 2)) == set(gids)

    @pytest.mark.parametrize('smiles,expected', [('CCCCC(CC)CC', True),
                                                 ('C', False)])
    def testFrag(self, grown, expected):
        assert expected == bool(grown.frag)

    @pytest.mark.parametrize('smiles,expected', [('CCCCC(CC)CC', [0, 1, 2]),
                                                 ('C', [0])])
    def testInit(self, grown, expected):
        np.testing.assert_equal(grown.init, expected)

    @pytest.mark.parametrize('smiles', ['CCCCC(CC)CC'])
    @pytest.mark.parametrize('sources,targets,polym_ht,expected',
                             [((None, ), (None, ), None, 4),
                              ((None, ), (None, ), (1, 4), 1),
                              ((0, 1), (4, 5), (1, 4), 3)])
    def testGetDihes(self, grown, sources, targets, polym_ht, expected):
        if polym_ht:
            grown.polym = True
            for idx in polym_ht:
                grown.GetAtomWithIdx(idx).SetBoolProp('polym_ht', True)
        assert expected == len(grown.getDihes(sources, targets))

    @pytest.mark.parametrize('smiles', ['CCCCC(CC)CC'])
    @pytest.mark.parametrize('source,target,expected', [(None, None, 7),
                                                        (None, 5, 6),
                                                        (3, None, 4),
                                                        (1, 5, 5)])
    def testFindPath(self, grown, source, target, expected):
        assert expected == len(grown.findPath(source=source, target=target))


class TestStruct:

    @pytest.mark.parametrize('density', [0.5, 1])
    def testInit(self, density):
        parser = parserutils.AmorpBldr()
        options = parser.parse_args(['C', '-density', str(density)])
        assert density == structutils.Struct(options=options).density


@pytest.mark.parametrize('smiles,cnum,seed', [(('CCCCC(CC)CC', 'C'), 1, 0)])
class TestGriddedStruct:

    @pytest.fixture
    def struct(self, mols, random_seed):
        return structutils.GriddedStruct.fromMols(mols)

    @pytest.mark.parametrize('expected',
                             [[20.88765862, 15.8452838, 11.16757977]])
    def testSetBox(self, struct, expected):
        np.testing.assert_almost_equal(struct.box.hi.values, expected)

    @pytest.mark.parametrize('expected',
                             [[10.44382931, 7.9226419, 5.58378988]])
    def testSize(self, struct, expected):
        np.testing.assert_almost_equal(struct.size, expected)

    @pytest.mark.parametrize('expected',
                             [[3.74465398, 2.40594602, 0.45998894]])
    def testSetConformers(self, struct, expected):
        struct.setConformers()
        xyz = np.concatenate([x.GetPositions() for x in struct.conf])
        np.testing.assert_almost_equal(xyz.max(axis=0), expected)

    @pytest.mark.parametrize('expected', [0.02351117892377739])
    def testSetDensity(self, struct, expected):
        struct.setDensity()
        np.testing.assert_almost_equal(struct.density, expected)

    @pytest.mark.parametrize('expected', [[14.1884832, 10.3285875, 0.459989]])
    def testGetPositions(self, struct, expected):
        struct.setConformers()
        xyz = struct.GetPositions()
        np.testing.assert_almost_equal(xyz.max(axis=0), expected)


class TestBox:

    @pytest.mark.parametrize('seed,al,size', [(0, 10, 1000), (0, 20, 100)])
    def testGetPoint(self, al, size, random_seed):
        box = structutils.Box.fromParams(al)
        points = box.getPoints(size=size)
        assert size == points.shape[0] == np.unique(points, axis=0).shape[0]
        assert (box.lo.min() <= points.min()).all()
        assert (box.hi.max() >= points.max()).all()


@pytest.mark.parametrize('smiles,cnum,seed', [(('CCCCCC'), 2, 0)])
class TestPackFrame:

    @pytest.mark.parametrize(
        'file,expected',
        [(envutils.test_data('hexane_liquid', 'dump.custom'), 1000)])
    def testGetPoint(self, frm, mols, expected):
        options = parserutils.AmorpBldr().parse_args(['CCCCCC'])
        struct = structutils.PackedStruct.fromMols(mols, options=options)
        pnts = structutils.PackFrame(frm, struct=struct).getPoints()
        assert expected == pnts.shape[0] == np.unique(pnts, axis=0).shape[0]
        assert (frm.box.lo.min() <= pnts.min()).all()
        assert (frm.box.hi.max() >= pnts.max()).all()


@pytest.mark.parametrize('smiles,cnum,seed', [(('CCCCC(CC)CC', 'C'), 2, 0)])
class TestGrownFrame:

    @pytest.fixture
    def dist(self, mols, random_seed):
        struct = structutils.GrownStruct.fromMols(mols)
        struct.run()
        return struct.dist

    @pytest.mark.parametrize('grp,expected', [(None, 16), ([0, 1], 36)])
    def testGetDists(self, dist, grp, expected):
        assert expected == dist.getDists(grp=grp).shape[0]


@pytest.mark.parametrize('smiles,cnum,seed', [(('CCCCC(CC)CC', 'C'), 2, 0)])
class TestPackedStruct:

    @pytest.fixture
    def struct(self, mols, random_seed):
        return structutils.PackedStruct.fromMols(mols)

    def testRun(self, struct):
        assert struct.run()

    def testSetBox(self, struct):
        np.testing.assert_almost_equal(struct.box.hi.max(), 9.859626871423261)

    def testSetFrame(self, struct):
        struct.setFrame()
        assert 20 == struct.dist.shape[0]

    @pytest.mark.parametrize('possible,expected', [(True, (True, [4])),
                                                   (False, (None, []))])
    def testSetFrame(self, struct, possible, expected):
        struct.setFrame()
        with mock.patch.object(struct, 'isPossible', return_value=possible):
            assert expected[0] == struct.setConformers()
        assert expected[1] == struct.placed

    @pytest.mark.parametrize('idx,expected', [(None, (4, 20)), (2, (2, 0))])
    def testAttempt(self, struct, idx, expected):
        if idx is not None:
            conf = next(itertools.islice(struct.conf, idx, idx + 1), None)
            conf.setConformer = mock.Mock(side_effect=structutils.ConfError)
        struct.setFrame()
        struct.attempt()
        assert expected == (struct.placed[0], len(struct.dist.gids.on))

    @pytest.mark.parametrize('placed,intvl,expected',
                             [([], 5, True), ([1, 1, 1, 1], 4, 0.0),
                              ([1, 2, 3, 4], 4, True)])
    def testIsPossible(self, struct, placed, intvl, expected):
        struct.placed = placed
        assert expected == struct.isPossible(intvl=intvl)

    @pytest.mark.parametrize('expected', [4])
    def testConfTotal(self, struct, expected):
        assert expected == struct.conf_total

    def testReset(self, struct):
        struct.setFrame()
        oxyz = struct.GetPositions()
        struct.attempt()
        struct.reset()
        assert not len(struct.dist.gids.on)
        np.testing.assert_almost_equal(struct.GetPositions(), oxyz)
        np.testing.assert_almost_equal(struct.dist, oxyz)


@pytest.mark.parametrize('smiles,cnum,seed', [(('CCCC', 'CC'), 2, 0)])
class TestGrownStruct:

    @pytest.fixture
    def struct(self, mols, random_seed):
        return structutils.GrownStruct.fromMols(mols)

    @pytest.mark.parametrize('pidx,gidx,expected', [(None, None, 4),
                                                    (3, None, 1),
                                                    (None, 1, 3)])
    def testAttempt(self, struct, pidx, gidx, expected):
        if pidx is not None:
            conf = next(itertools.islice(struct.conf, pidx, pidx + 1), None)
            conf.setConformer = mock.Mock(side_effect=structutils.ConfError)
        if gidx is not None:
            conf = next(itertools.islice(struct.conf, gidx, gidx + 1), None)
            conf.grow = mock.Mock(side_effect=structutils.ConfError)
        struct.setFrame()
        struct.attempt()
        assert expected == struct.placed[0]


@pytest.mark.parametrize('cnum', [2])
class TestFragment:

    @pytest.fixture
    def frag(self, emol, dihe):
        mol = structutils.GrownMol(emol)
        return structutils.Fragment(mol.GetConformer(0), dihe)

    @pytest.fixture
    def new(self, frag):
        frag.setNfrags()
        return frag.new(frag.conf.mol.GetConformer(1))

    @pytest.mark.parametrize('smiles,dihe,expected',
                             [('CCCC', (0, 1, 2, 3), [3]),
                              ('CCCCC(CC)CC',
                               (0, 1, 2, 3), [3, 4, 5, 6, 7, 8])])
    def testSetUp(self, frag, expected):
        np.testing.assert_equal(frag.ids, expected)

    @pytest.mark.parametrize('smiles,dihe,expected',
                             [('CCCC', (0, 1, 2, 3), 1),
                              ('CCCCC(CC)CC', (0, 1, 2, 3), 4),
                              ('CCC(CC)CCCC(CC)CC', (0, 1, 2, 5), 7)])
    def testSetNfrags(self, frag, expected):
        frag.setNfrags()
        assert expected == len(list(frag.next()))

    @pytest.mark.parametrize('smiles,dihe,expected',
                             [('CCCC', (0, 1, 2, 3), [7, 0]),
                              ('CCCCC(CC)CC', (0, 1, 2, 3), [12, 1])])
    def testNew(self, new, expected):
        assert expected == [*new.ids, len(new.nfrags)]
        assert 36 == new.ovals.shape[0] == len(new.vals)

    @pytest.mark.parametrize('smiles,dihe', [('CCCCC', (0, 1, 2, 3))])
    @pytest.mark.parametrize('partial,expected', [(False, 2), (True, 0)])
    def testNext(self, new, partial, expected):
        assert expected == len(list(new.next(partial=partial)))

    @pytest.mark.parametrize('smiles,dihe', [('CCCCC', (0, 1, 2, 3))])
    def testReset(self, new):
        new.vals.pop()
        assert 36 != len(new.vals)
        new.reset()
        assert 36 == len(new.vals)


@pytest.mark.parametrize('cnum', [2])
class TestFirst:

    @pytest.fixture
    def first(self, emol, dihe):
        mol = structutils.GrownMol(emol)
        return structutils.First(mol.GetConformer(0), dihe)

    @pytest.mark.parametrize('smiles,dihe,expected',
                             [('CCCC', (0, 1, 2, 3), 1),
                              ('CCCCC(CC)CC', (0, 1, 2, 3), 5),
                              ('CCC(CC)CCCC(CC)CC', (0, 1, 2, 5), 8)])
    def testSetUp(self, first, expected):
        assert expected == len(list(first.next()))

    @pytest.mark.parametrize('smiles,dihe,expected',
                             [('CCCC', (0, 1, 2, 3), 1),
                              ('CCCCC(CC)CC', (0, 1, 2, 3), 5),
                              ('CCC(CC)CCCC(CC)CC', (0, 1, 2, 5), 8)])
    def testNew(self, first, expected):
        conf = first.conf.mol.GetConformer(1)
        new = first.new(conf)
        for frag in new.next():
            assert conf == frag.conf
        assert expected == len(list(first.next()))
        assert conf.gids.max() == max([y for x in new.next() for y in x.ids])