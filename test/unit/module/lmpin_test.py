from unittest import mock

import pytest

from nemd import lmpin
from nemd import parserutils


@pytest.mark.parametrize('args', [['C', '-JOBNAME', 'name']])
class TestIn:

    @pytest.fixture
    def lmp_in(self, args, tmp_dir):
        return lmpin.In(options=parserutils.MolBase().parse_args(args))

    def testSetup(self, lmp_in, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.setup()
        assert 'units metal' in lines

    def testPair(self, lmp_in, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.pair()
        assert 'pair_style sw' in lines

    def testData(self, lmp_in, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.data()
        assert 'read_data name.data' in lines

    def testCoeff(self, lmp_in, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.coeff('ff', ['Si'])
        assert 'pair_coeff * * ff Si' in lines

    @pytest.mark.parametrize(
        'xyz,force,expected',
        [(True, False, 'dump 1 all custom 1000 name.custom.gz id xu yu zu'),
         (False, True, 'dump 1 all custom 1000 name.custom.gz id fx fy fz'),
         (False, False, None),
         (True, True,
          'dump 1 all custom 1000 name.custom.gz id xu yu zu fx fy fz')])
    def testTraj(self, lmp_in, xyz, force, expected, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.traj(xyz=xyz, force=force)
        assert (expected in lines) if expected else (not lines)

    @pytest.mark.parametrize(
        'sort,fmt,expected',
        [(True, None, 'dump_modify 1 sort id'),
         (True, "float '%20.15f'",
          "dump_modify 1 sort id format float '%20.15f'"),
         (False, "float '%20.15f'", "dump_modify 1 format float '%20.15f'"),
         (False, None, None)])
    def testTrajModify(self, lmp_in, sort, fmt, expected, tmp_line):
        with tmp_line() as (lmp_in.fh, lines):
            lmp_in.traj(sort=sort, fmt=fmt)
        assert (expected in lines) if expected else (1 == len(lines))

    @pytest.mark.parametrize(
        'no_minimize,rest,expected',
        [(False, None, 'minimize 1.0e-6 1.0e-8 1000000 10000000'),
         (False,
          'fix rest all restrain dihedral 1 2 3 4 -2000.0 -2000.0 120.0\n',
          'minimize 1.0e-6 1.0e-8 1000000 10000000')])
    def testMinimize(self, args, no_minimize, rest, expected, tmp_line):
        if no_minimize:
            args += ['-no_minimize']
        options = parserutils.MolBase().parse_args(args)
        with mock.patch('nemd.lmpin.In.rest',
                        new_callable=mock.PropertyMock,
                        return_value=rest):
            lmp_in = lmpin.In(options=options)
            with tmp_line() as (lmp_in.fh, lines):
                lmp_in.minimize()
        assert (expected in lines) if expected else (not lines)
        if lines and rest:
            assert rest in lines


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
