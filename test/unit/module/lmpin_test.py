import os
import types

import pytest

from nemd import lmpin
from nemd import np
from nemd import parserutils


@pytest.mark.parametrize('smiles,cnum', [('C', 1)])
class TestSinglePoint:

    @pytest.fixture
    def single(self, smiles, emol, tmp_dir):
        args = [smiles, '-JOBNAME', 'name']
        options = parserutils.MolBase().parse_args(args)
        return lmpin.SinglePoint(struct=types.SimpleNamespace(options=options))

    def testWrite(self, single, tmp_dir):
        single.write()
        os.path.isfile(single.outfile)

    def testSetUp(self, single):
        single.setUp()
        assert 12 == len(single)

    def testSetup(self, single):
        single.setup()
        assert ['units metal', 'atom_style atomic'] == single

    @pytest.mark.parametrize('args,expected',
                             [(('units', 'real'), 'units real')])
    def testJoin(self, single, args, expected):
        single.join(*args)
        assert expected == single[0]

    def testPair(self, single):
        single.pair()
        assert 'pair_style sw' == single[0]

    def testData(self, single):
        single.data()
        assert 'read_data name.data' == single[0]

    def testTraj(self, single):
        single.traj()
        assert 2 == len(single)

    @pytest.mark.parametrize('args', [(1, 'all', 'custom', 10, 'name')])
    @pytest.mark.parametrize(
        'xyz,force,expected',
        [(True, False, 'dump 1 all custom 10 name xu yu zu'),
         (False, True, 'dump 1 all custom 10 name fx fy fz'),
         (True, True, 'dump 1 all custom 10 name xu yu zu fx fy fz')])
    def testDump(self, single, args, xyz, force, expected):
        single.dump(*args, xyz=xyz, force=force)
        assert (expected in single) if expected else (not single)

    @pytest.mark.parametrize('args,sort,fmt,expected',
                             [((1, ), True, None, 'dump_modify 1 sort id'),
                              ((1, ), True, "float '%20.15f'",
                               "dump_modify 1 sort id format float '%20.15f'"),
                              ((1, ), False, "float '%20.15f'",
                               "dump_modify 1 format float '%20.15f'"),
                              ((1, ), False, None, None)])
    def testDumpModify(self, single, args, sort, fmt, expected):
        single.dump_modify(*args, sort=sort, fmt=fmt)
        assert (expected == single[0]) if expected else (not single)

    @pytest.mark.parametrize(
        'no_minimize,geo,val,expected',
        [(False, None, None, 'minimize 1.0e-6 1.0e-8 1000000 10000000'),
         (True, None, None, None),
         (True, 'dihedral 1 2 3 4', 120,
          'fix rest all restrain dihedral 1 2 3 4 -2000.0 -2000.0 120')])
    def testMinimize(self, single, no_minimize, geo, val, expected):
        single.no_minimize = no_minimize
        single.options.substruct = [None, val]
        single.minimize(geo=geo)
        assert (expected in single) if expected else (2 == len(single))

    @pytest.mark.parametrize('temp,expected', [(300, 3), (0, 0)])
    def testTimestep(self, single, expected, temp):
        single.options.temp = temp
        single.timestep()
        assert expected == len(single)

    @pytest.mark.parametrize('unit,expected', [('real', 1e-12),
                                               ('metal', 1e-9)])
    def testTimeUnit(self, single, unit, expected):
        np.testing.assert_almost_equal(single.time_unit(unit), expected)

    def testSimulation(self, single):
        single.simulation()
        assert 'run 0' == single[0]

    @pytest.mark.parametrize('nstep,expected', [(0, 'run 0'), (1.0, 'run 1')])
    def testRunStep(self, single, nstep, expected):
        single.run_step(nstep=nstep)
        assert expected == single[0]

    @pytest.mark.parametrize('data,expected', [([], 1), (['run 0'], 2),
                                               (['fix %s'], 2),
                                               ([['fix %s']], 3),
                                               ([['fix %s', 'fix %s']], 5)])
    def testFinalize(self, single, data, expected):
        single.extend(data)
        single.finalize()
        assert expected == len(single)

    @pytest.mark.parametrize(
        'name,expr,bracked,quoted,expected',
        [('xl', 'xhi - ylo', False, True, 'variable xl equal "xhi - ylo"'),
         ('modulus', 'modulus', True, False,
          'variable modulus equal ${modulus}')])
    def testEqual(self, single, name, expr, bracked, quoted, expected):
        single.equal(name, expr, bracked=bracked, quoted=quoted)
        assert expected == single[0]

    @pytest.mark.parametrize('args,expected', [
        (('fact', 'python', 'getBdryFact'), 'variable fact python getBdryFact')
    ])
    def testVariable(self, single, args, expected):
        single.variable(*args)
        assert expected == single[0]

    @pytest.mark.parametrize('newline,expected', [(False, [[]]),
                                                  (True, [[], ''])])
    def testBlock(self, single, newline, expected):
        with single.block(newline=newline) as blk:
            with blk.block(newline=newline):
                pass
        assert expected == single


@pytest.mark.parametrize('cnum', [1])
class TestRampUp:
    ARGS = ('-relax_time', '0.01', '-prod_time', '0.02', '-stemp', '2',
            '-temp', '3', '-press', '4')

    @pytest.fixture
    def ramp_up(self, smiles, args, emol, tmp_dir):
        args = (smiles, ) + args
        options = parserutils.MolBase().parse_args(args)
        kwargs = dict(options=options, atom_total=emol.GetNumAtoms())
        return lmpin.RampUp(struct=types.SimpleNamespace(**kwargs))

    @pytest.mark.parametrize('smiles', ['CC'])
    @pytest.mark.parametrize('args,expected',
                             [(('-relax_time', '1'), 1000000.0),
                              (('-relax_time', '0'), 0.0),
                              (('-relax_time', '0.0001'), 1000.0)])
    def testInit(self, ramp_up, expected):
        assert expected == ramp_up.relax_step

    @pytest.mark.parametrize('smiles,args,expected',
                             [('CC', (), 9), ('CC', ('-temp', '0'), 1),
                              ('C', (), 1)])
    def testSimulations(self, ramp_up, expected):
        ramp_up.simulation()
        assert expected == len(ramp_up)

    @pytest.mark.parametrize('smiles', ['CC'])
    @pytest.mark.parametrize(
        'args,expected',
        [(('-stemp', '20'), 'velocity all create 20'),
         (('-temp', '100', '-relax', '0'), 'velocity all create 100')])
    def testVelocity(self, ramp_up, expected):
        ramp_up.velocity()
        assert ramp_up[0].startswith(expected)

    @pytest.mark.parametrize(
        'smiles,args,expected',
        [('CC', ARGS, 'fix %s all temp/berendsen 2.0 2.0 100')])
    def testStartLow(self, ramp_up, expected):
        ramp_up.startLow()
        assert expected == ramp_up[0][0]

    @pytest.mark.parametrize('smiles,args,style', [('CC', (), 'berendsen')])
    @pytest.mark.parametrize(
        'nstep,stemp,temp,expected',
        [(1E4, None, 300, 'fix %s all temp/berendsen 300 300 100'),
         (10, 20, 50, 'fix %s all temp/berendsen 20 50 100'),
         (0, 20, 50, None)])
    def testNvt(self, ramp_up, nstep, stemp, temp, style, expected):
        ramp_up.nvt(nstep=nstep, stemp=stemp, temp=temp, style=style)
        assert (expected == ramp_up[0][0]) if expected else (not ramp_up)

    @pytest.mark.parametrize('smiles,args', [('CC', ())])
    def testFixAll(self, ramp_up):
        ramp_up.fix_all('temp/berendsen', 10, 10, 100)
        assert 'fix %s all temp/berendsen 10 10 100' == ramp_up[0]

    @pytest.mark.parametrize('smiles,args,expected', [('CC', ARGS, [
        'fix %s all press/berendsen iso 1 4.0 1000 modulus 10',
        'fix %s all temp/berendsen 2.0 3.0 100'
    ])])
    def testRampUp(self, ramp_up, expected):
        ramp_up.rampUp()
        assert expected == ramp_up[0][:2]

    @pytest.mark.parametrize('smiles,args,style', [('CC', (), 'berendsen')])
    @pytest.mark.parametrize('nstep,spress,press,modulus,expected', [
        (1E4, None, 1, 10,
         'fix %s all press/berendsen iso 1 1 1000 modulus 10'),
        (10, 0.1, 1, 2, 'fix %s all press/berendsen iso 0.1 1 1000 modulus 2'),
        (0, 0.1, 1, 2, None)
    ])
    def testNpt(self, ramp_up, nstep, spress, press, style, modulus, expected):
        ramp_up.npt(nstep=nstep,
                    spress=spress,
                    press=press,
                    modulus=modulus,
                    style=style)
        assert (expected == ramp_up[0][0]) if expected else (not ramp_up)

    @pytest.mark.parametrize('smiles,args,expected', [('CC', ARGS, [
        'fix %s all press/berendsen iso 4.0 4.0 1000 modulus 10',
        'fix %s all temp/berendsen 3.0 3.0 100'
    ])])
    def testRelaxation(self, ramp_up, expected):
        ramp_up.relaxation()
        assert expected == ramp_up[0][:2]

    @pytest.mark.parametrize('smiles,args', [('CC', ARGS)])
    @pytest.mark.parametrize(
        'prod_ens,expected',
        [('NVE', ['fix %s all nve', 'run 20000']),
         ('NVT', [
             'fix %s all temp/berendsen 3.0 3.0 100', 'fix %s all nve',
             'run 20000'
         ]),
         ('NPT', [
             'fix %s all press/berendsen iso 4.0 4.0 1000 modulus 10',
             'fix %s all temp/berendsen 3.0 3.0 100', 'fix %s all nve',
             'run 20000'
         ])])
    def testProduction(self, ramp_up, prod_ens, expected):
        ramp_up.options.prod_ens = prod_ens
        ramp_up.production()
        assert expected == ramp_up[0]

    @pytest.mark.parametrize('smiles,args', [('CC', ())])
    @pytest.mark.parametrize('nstep,expected',
                             [(1E4, ['fix %s all nve', 'run 10000']),
                              (0, None)])
    def testNve(self, ramp_up, nstep, expected):
        ramp_up.nve(nstep=nstep)
        assert (expected == ramp_up[0]) if expected else (not ramp_up)
