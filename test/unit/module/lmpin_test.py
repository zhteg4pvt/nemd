import pytest

from nemd import lmpin
from nemd import np
from nemd import parserutils
import types


@pytest.mark.parametrize('smiles,cnum', [('C', 1)])
class TestScript:

    @pytest.fixture
    def script(self, smiles, emol, tmp_dir):
        args = [smiles, '-JOBNAME', 'name']
        options = parserutils.MolBase().parse_args(args)
        return lmpin.Script(struct=types.SimpleNamespace(options=options))

    def testSetup(self, script, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.setup()
        assert 'units metal' in lines

    def testPair(self, script, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.pair()
        assert 'pair_style sw' in lines

    def testData(self, script, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.data()
        assert 'read_data name.data' in lines

    @pytest.mark.parametrize(
        'xyz,force,expected',
        [(True, False, 'dump 1 all custom 1000 name.custom.gz id xu yu zu'),
         (False, True, 'dump 1 all custom 1000 name.custom.gz id fx fy fz'),
         (False, False, None),
         (True, True,
          'dump 1 all custom 1000 name.custom.gz id xu yu zu fx fy fz')])
    def testTraj(self, script, xyz, force, expected, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.traj(xyz=xyz, force=force)
        assert (expected in lines) if expected else (not lines)

    @pytest.mark.parametrize(
        'sort,fmt,expected',
        [(True, None, 'dump_modify 1 sort id'),
         (True, "float '%20.15f'",
          "dump_modify 1 sort id format float '%20.15f'"),
         (False, "float '%20.15f'", "dump_modify 1 format float '%20.15f'"),
         (False, None, None)])
    def testTrajModify(self, script, sort, fmt, expected, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.traj(sort=sort, fmt=fmt)
        assert (expected in lines) if expected else (1 == len(lines))

    @pytest.mark.parametrize(
        'no_minimize,geo,val,expected',
        [(False, None, None, 'minimize 1.0e-6 1.0e-8 1000000 10000000'),
         (True, None, None, None),
         (True, 'dihedral 1 2 3 4', 120,
          'fix rest all restrain dihedral 1 2 3 4 -2000.0 -2000.0 120')])
    def testMinimize(self, script, no_minimize, geo, val, expected, tmp_line):
        script.no_minimize = no_minimize
        script.options.substruct = [None, val]
        with tmp_line() as (script.fh, lines):
            script.minimize(geo=geo)
        assert (expected in lines) if expected else (2 == len(lines))

    @pytest.mark.parametrize('temp,expected', [(300, 4), (0, 0)])
    def testTimestep(self, script, expected, temp, tmp_line):
        script.options.temp = temp
        with tmp_line() as (script.fh, lines):
            script.timestep()
        assert expected == len(lines)

    @pytest.mark.parametrize('atom_total,expected', [(1, 3), (100, 147)])
    def testSimulation(self, script, atom_total, expected, tmp_line):
        with tmp_line() as (script.fh, lines):
            script.simulation(atom_total=atom_total)
        assert expected == len(lines)

    @pytest.mark.parametrize('unit,expected', [('real', 1e-12),
                                               ('metal', 1e-9)])
    def testTimeUnit(self, script, unit, expected):
        np.testing.assert_almost_equal(script.time_unit(unit), expected)


# class TestFixWriter:
#
#     @pytest.fixture
#     def fix_writer(self):
#         options = {x: y for x, y in self.getOptions()._get_kwargs()}
#         struct_info = types.SimpleNamespace(btypes=[2],
#                                             atypes=[1],
#                                             testing=False)
#         options = types.SimpleNamespace(**options, **struct_info.__dict__)
#         return lmpin.FixWriter(io.StringIO(), options=options)
#
#     @staticmethod
#     def getContents(obj):
#         obj.write()
#         return super(TestFixWriter, TestFixWriter).getContents(obj)
#
#     def testWriteFix(self, fix_writer):
#         fix_writer.fixShake()
#         assert 'b 2 a 1' in self.getContents(fix_writer)
#
#     def testVelocity(self, fix_writer):
#         fix_writer.velocity()
#         assert 'create' in self.getContents(fix_writer)
#
#     def testStartLow(self, fix_writer):
#         fix_writer.startLow()
#         assert 'temp/berendsen' in self.getContents(fix_writer)
#
#     def testRampUp(self, fix_writer):
#         fix_writer.rampUp()
#         assert 'loop' in self.getContents(fix_writer)
#
#     @pytest.mark.parametrize('prod_ens, args', [('NVT', True), ('NPT', False)])
#     def testRelaxAndDefrom(self, fix_writer, prod_ens, args):
#         fix_writer.options.prod_ens = prod_ens
#         fix_writer.relaxAndDefrom()
#         contain = 'change_box' in self.getContents(fix_writer)
#         assert contain == args
#
#     @pytest.mark.parametrize('prod_ens, args',
#                              [('NVE', 'nve'), ('NVT', 'temp'),
#                               (None, 'press')])
#     def testProduction(self, fix_writer, prod_ens, args):
#         fix_writer.options.prod_ens = prod_ens
#         fix_writer.production()
#         assert args in self.getContents(fix_writer)
