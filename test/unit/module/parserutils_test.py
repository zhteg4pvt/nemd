import argparse
import os

import pytest

from nemd import envutils
from nemd import parserutils

AR_DIR = envutils.test_data('ar', 'gas')
TRAJ_FILE = os.path.join(AR_DIR, 'ar100.custom.gz')


@pytest.mark.parametrize('raise_type', [argparse.ArgumentTypeError])
class TestType:

    @pytest.mark.parametrize('file,is_raise', [('not_existing', True),
                                               (
                                                   TRAJ_FILE,
                                                   False,
                                               )])
    def testFile(self, file, check_raise):
        with check_raise():
            parserutils.type_file(file)

    @pytest.mark.parametrize('file,is_raise', [(AR_DIR, False),
                                               ('not_existing', True),
                                               (TRAJ_FILE, True)])
    def testDir(self, file, check_raise):
        with check_raise():
            parserutils.type_dir(file)

    @pytest.mark.parametrize('arg,val,is_raise', [('y', True, False),
                                                  ('n', False, False),
                                                  ('wa', None, True)])
    def testBool(self, arg, val, check_raise):
        with check_raise():
            assert val == parserutils.type_bool(arg)

    @pytest.mark.parametrize('arg,val,is_raise', [('123', 123, False),
                                                  ('5.6', 5.6, False),
                                                  ('wa', None, True),
                                                  ('None', None, True)])
    def testFloat(self, arg, val, check_raise):
        with check_raise():
            assert val == parserutils.type_float(arg)

    @pytest.mark.parametrize('bottom,top', [(100, 200)])
    @pytest.mark.parametrize('value,included_bottom,include_top,is_raise',
                             [(123, True, True, False),
                              (100, True, True, False),
                              (100, False, True, True),
                              (200, False, True, False),
                              (200, False, False, True)])
    def testRanged(self, value, bottom, top, included_bottom, include_top,
                   check_raise):
        with check_raise():
            parserutils.type_ranged(value,
                                    bottom=bottom,
                                    top=top,
                                    included_bottom=included_bottom,
                                    include_top=include_top)

    @pytest.mark.parametrize('bottom,top', [(1.12, 3.45)])
    @pytest.mark.parametrize('arg,included_bottom,include_top,is_raise',
                             [('2', True, True, False),
                              ('1.12', True, True, False),
                              ('1.12', False, True, True),
                              ('3.45', False, True, False),
                              ('3.45', False, False, True)])
    def testRangedFloat(self, arg, bottom, top, included_bottom, include_top,
                        check_raise):
        with check_raise():
            parserutils.type_ranged_float(arg,
                                          bottom=bottom,
                                          top=top,
                                          included_bottom=included_bottom,
                                          include_top=include_top)

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1.12', False),
                                              ('-1.12', True)])
    def testNonnegativeFloat(self, arg, check_raise):
        with check_raise():
            parserutils.type_nonnegative_float(arg)

    @pytest.mark.parametrize('arg,is_raise', [('0', True), ('1.12', False),
                                              ('-1.12', True)])
    def testPositiveFloat(self, arg, check_raise):
        with check_raise():
            parserutils.type_positive_float(arg)

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1.12', True),
                                              ('-1', False)])
    def testInt(self, arg, check_raise):
        with check_raise():
            parserutils.type_int(arg)

    @pytest.mark.parametrize('arg,is_raise', [('1', False), ('0', True),
                                              ('-1', True)])
    def testPositiveInt(self, arg, check_raise):
        with check_raise():
            parserutils.type_positive_int(arg)

    @pytest.mark.parametrize('arg,is_raise', [('0', False), ('1234', False),
                                              ('-1', True), (2**31, True)])
    def testRandomSeed(self, arg, check_raise):
        with check_raise():
            parserutils.type_random_seed(arg)

    @pytest.mark.parametrize('arg,is_raise', [('C', False),
                                              ('not_valid', True)])
    def testTypeSmiles(self, arg, check_raise):
        with check_raise():
            parserutils.type_smiles(arg)

    @pytest.mark.parametrize('arg,allow_reg,is_raise', [('C', True, False),
                                                        ('C', False, True),
                                                        ('*C*', False, False),
                                                        ('*C*.C', False, True),
                                                        ('C*', True, True)])
    def testCruSmiles(self, arg, allow_reg, check_raise):
        with check_raise():
            parserutils.type_cru_smiles(arg, allow_reg=allow_reg)

    @pytest.mark.parametrize('arg,is_raise', [('0.2', False), ('0', True),
                                              ('0.99', False), ('1', True)])
    def testLastPctType(self, arg, check_raise):
        with check_raise():
            parserutils.LastPct.type(arg)


class TestLastPct:

    @pytest.mark.parametrize('data', [[0, 1, 2, 3, 4]])
    @pytest.mark.parametrize('arg,buffer,sidx', [('0.7', 0, 2), ('0.6', 0, 2),
                                                 ('0.6', 1, 1)])
    def testLastPctGetSidx(self, data, arg, buffer, sidx):
        ptc = parserutils.LastPct.type(arg)
        assert sidx == ptc.getSidx(data, buffer=buffer)
