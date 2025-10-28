import os

import conftest
import ml_reg_driver as driver
import numpy as np
import pytest
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.utils import validation

from nemd import envutils
from nemd import test

SRC = envutils.Src()
POS_CSV = SRC.test('ml', 'position_salaries.csv')
SEL_CSV = SRC.test('ml', 'model_selection.csv')


@conftest.require_src
class TestReg:

    @pytest.fixture
    def reg(self, method, logger):
        args = [POS_CSV, '-method', method, '-seed', '0']
        options = driver.ArgumentParser().parse_args(args)
        reg = driver.Regression(options=options, logger=logger)
        reg.read()
        reg.split()
        reg.setRegs()
        return reg

    @pytest.mark.parametrize('method,expected',
                             [('lr', (linear_model.LinearRegression, None)),
                              ('svr', (svm.SVR, 2)),
                              ('poly', (linear_model.LinearRegression, None)),
                              ('dt', (tree.DecisionTreeRegressor, None)),
                              ('rfr', (ensemble.RandomForestRegressor, None))])
    def testSetUp(self, reg, expected):
        assert isinstance(reg.regs[0].reg, expected[0])
        assert expected[1] == (reg.scs and len(reg.scs))

    @pytest.mark.parametrize('method', ['lr', 'svr', 'poly', 'dt', 'rfr'])
    def testFit(self, reg):
        reg.regs[0].fit(reg.xtrain, reg.ytrain)
        validation.check_is_fitted(reg.regs[0].reg)

    @pytest.mark.parametrize('method,expected', [('lr', 43), ('svr', 0.0),
                                                 ('poly', 346.0)])
    def testOpr(self, reg, expected):
        assert expected == reg.regs[0].opr(reg.xtrain).sum()

    @pytest.mark.parametrize('method,expected', [('lr', False),
                                                 ('poly', True)])
    def testPoly(self, reg, expected):
        assert expected == bool(reg.regs[0].poly)

    @pytest.mark.parametrize('method,idx,inversed,expected',
                             [('lr', 0, False, 43), ('svr', 0, False, 0.0),
                              ('svr', 1, False, -6.5065991),
                              ('svr', 0, True, 164.50353647)])
    def testScale(self, reg, idx, inversed, expected):
        scaled = reg.regs[0].scale(reg.xtrain, idx=idx, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('method,inversed,expected',
                             [('svr', False, 0),
                              ('svr', True, 575439538784.8147)])
    def testScaleY(self, reg, inversed, expected):
        scaled = reg.regs[0].scaleY(reg.ytrain, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('method,expected', [('lr', 0.9779215),
                                                 ('svr', 0.951383945),
                                                 ('poly', 0.70548068772),
                                                 ('dt', 0.5857438),
                                                 ('rfr', 0.99068337)])
    def testScore(self, reg, expected):
        score = reg.regs[0].score(reg.xtest, reg.ytest)
        np.testing.assert_almost_equal(score, expected)

    @pytest.mark.parametrize('method,expected', [('lr', 335474.5596868884),
                                                 ('svr', 193234.81951553142),
                                                 ('poly', 196528.16459152475),
                                                 ('dt', 150000.0),
                                                 ('rfr', 162500.0)])
    def testPredict(self, reg, expected):
        np.testing.assert_almost_equal(reg.regs[0].predict([[6.5]]), expected)


@conftest.require_src
class TestRegression:

    @pytest.fixture
    def reg(self, args, logger):
        options = driver.ArgumentParser().parse_args(args)
        return driver.Regression(options=options, logger=logger)

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'
                            ], 'Figure saved as name.svg'),
                          ([SEL_CSV, '-seed', '0'], 'r2 score (lr): 0.9325')])
    def testRun(self, reg, expected, tmp_dir):
        reg.run()
        reg.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected', [([POS_CSV], (10, 3)),
                                               ([SEL_CSV], (9568, 5))])
    def testRead(self, reg, expected):
        reg.read()
        assert expected == reg.data.shape

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV], (8, 2, 1, 1)),
                              ([SEL_CSV], (7654, 1914, 4, 1))])
    def testSplit(self, reg, expected):
        reg.read()
        reg.split()
        assert expected == (reg.xtrain.shape[0], *reg.xtest.shape,
                            reg.ytrain.shape[1])

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-method', 'lr', 'svr', 'poly', 'dt', 'rfr'], 5)])
    def testSetRegs(self, reg, expected):
        reg.read()
        reg.split()
        reg.setRegs()
        assert expected == len(reg.regs)
        for reg in reg.regs:
            validation.check_is_fitted(reg.reg)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'lr'], None),
                              ([POS_CSV, '-method', 'svr', 'poly'], 2)])
    def testScs(self, reg, expected):
        reg.read()
        reg.split()
        assert expected == (reg.scs and len(reg.scs))

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'svr'], 2)])
    def testGetScaler(self, reg, expected):
        reg.read()
        reg.split()
        assert expected == len([x for x in reg.getScaler()])

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0'], 'r2 score (lr): 0.9779')])
    def testScore(self, reg, expected):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.score()

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-seed', '0'], (90, 1)),
                              ([SEL_CSV, '-seed', '0'], None)])
    def testGrid(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.grid()
        assert (expected == reg.gridded.shape) if expected else not reg.gridded

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'], True),
         ([SEL_CSV, '-seed', '0', '-JOBNAME', 'name'], False)])
    def testPlot(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.grid()
        reg.plot()
        assert expected == os.path.isfile('name.csv')


@conftest.require_src
class TestArgumentParser:

    @pytest.fixture
    def parser(self):
        return driver.ArgumentParser()

    @pytest.mark.parametrize('args,expected', [
        ([POS_CSV], [1, 2, 100]),
        ([POS_CSV, '-method', 'svr', 'rfr', '-degree', '3', '-tree_num', '10'
          ], [2, 3, 10])
    ])
    def testParseArgs(self, parser, args, expected):
        options = parser.parse_args(args)
        assert expected == [
            len(options.method), options.degree, options.tree_num
        ]
