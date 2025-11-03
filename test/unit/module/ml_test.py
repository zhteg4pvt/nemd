import os

import conftest
import numpy as np
import pytest
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.utils import validation

from nemd import envutils
from nemd import ml
from nemd import parserutils
from nemd import test

SRC = envutils.Src()
POS_CSV = SRC.test('ml', 'position_salaries.csv')
SEL_CSV = SRC.test('ml', 'model_selection.csv')
SOC_CSV = SRC.test('ml', 'social_network_ads.csv')


@conftest.require_src
class TestReg:

    @pytest.fixture
    def reg(self, file, method, logger):
        args = [file, '-method', method, '-seed', '0']
        options = parserutils.Reg().parse_args(args)
        reg = ml.Ml(options=options, logger=logger)
        reg.read()
        reg.split()
        reg.setRegs()
        return reg

    @pytest.mark.parametrize('file', [SOC_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('lr', (linear_model.LinearRegression, 0)),
                              ('poly', (linear_model.LinearRegression, 0)),
                              ('svr', (svm.SVR, 2)),
                              ('dt', (tree.DecisionTreeRegressor, 0)),
                              ('rfr', (ensemble.RandomForestRegressor, 0)),
                              ('logit', (linear_model.LogisticRegression, 1)),
                              ('knn', (neighbors.KNeighborsClassifier, 1))])
    def testSetUp(self, reg, expected):
        assert isinstance(reg.models[0].reg, expected[0])
        assert expected[1] == len([x for x in reg.scs if x])

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method', ['lr', 'svr', 'poly', 'dt', 'rfr'])
    def testFit(self, reg):
        reg.models[0].fit(reg.xtrain, reg.ytrain)
        validation.check_is_fitted(reg.models[0].reg)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', 43), ('svr', 0.0),
                                                 ('poly', 346.0)])
    def testOpr(self, reg, expected):
        assert expected == reg.models[0].opr(reg.xtrain).sum()

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', False),
                                                 ('poly', True)])
    def testPoly(self, reg, expected):
        assert expected == bool(reg.models[0].poly)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,idx,inversed,expected',
                             [('lr', 0, False, 43), ('svr', 0, False, 0.0),
                              ('svr', 1, False, -6.5065991),
                              ('svr', 0, True, 164.50353647)])
    def testScale(self, reg, idx, inversed, expected):
        scaled = reg.models[0].scale(reg.xtrain, idx=idx, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,inversed,expected',
                             [('svr', False, 0),
                              ('svr', True, 575439538784.8147)])
    def testScaleY(self, reg, inversed, expected):
        scaled = reg.models[0].scaleY(reg.ytrain, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file,method,expected',
                             [(POS_CSV, 'lr', ['r2', 0.978]),
                              (POS_CSV, 'svr', ['r2', 0.951]),
                              (POS_CSV, 'poly', ['r2', 0.705]),
                              (POS_CSV, 'dt', ['r2', 0.586]),
                              (POS_CSV, 'rfr', ['r2', 0.991]),
                              (SOC_CSV, 'logit', ['accuracy', 0.925])])
    def testScore(self, reg, expected):
        name, score = reg.models[0].score(reg.xtest, reg.ytest)
        assert expected[0] == name
        np.testing.assert_almost_equal(score, expected[1], decimal=3)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', 335474),
                                                 ('svr', 193234),
                                                 ('poly', 196528),
                                                 ('dt', 150000),
                                                 ('rfr', 162500)])
    def testPredict(self, reg, expected):
        assert np.allclose(reg.models[0].predict([[6.5]]), expected, rtol=0.01)


@conftest.require_src
class TestMl:

    @pytest.fixture
    def reg(self, args, logger):
        options = parserutils.Reg().parse_args(args)
        return ml.Ml(options=options, logger=logger)

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
        [([POS_CSV, '-method', 'lr', 'svr', 'poly', 'dt', 'rfr'], 5),
         ([SOC_CSV, '-method', 'knn', 'logit'], 2),
         ([SOC_CSV, '-method', 'lr', 'logit', '-test_size', '0.997'], 1)])
    def testSetRegs(self, reg, expected):
        reg.read()
        reg.split()
        reg.setRegs()
        assert expected == len(reg.models)
        for reg in reg.models:
            validation.check_is_fitted(reg.reg)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'lr'], 0),
                              ([POS_CSV, '-method', 'svr', 'poly'], 2),
                              ([POS_CSV, '-method', 'knn', 'lr'], 1)])
    def testScs(self, reg, expected):
        reg.read()
        reg.split()
        assert expected == len([x for x in reg.scs if x])

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

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'], True),
         ([SEL_CSV, '-seed', '0', '-JOBNAME', 'name'], False)])
    def testScatter(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.scatter()
        assert expected == os.path.isfile('name.csv')

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'], [['lr', 101]]),
         ([SEL_CSV, '-seed', '0', '-JOBNAME', 'name'], [])])
    def testCols(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        assert expected == [[x, y.size] for x, y in reg.cols()]

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-seed', '0'], (101, 1)),
                              ([SEL_CSV, '-seed', '0'], (0, 0))])
    def testPred(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        assert expected == reg.pred.shape

    @pytest.mark.parametrize('args,expected', [([POS_CSV, '-seed', '0'], 1),
                                               ([SOC_CSV, '-seed', '0'], 2)])
    def testGrids(self, reg, expected):
        reg.read()
        reg.split()
        reg.setRegs()
        assert expected == len(reg.grids)

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'ml_test'], 'ml_test.svg')])
    def testSave(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.scatter()
        reg.save()
        assert os.path.isfile(expected)

    @pytest.mark.parametrize(
        'args,expected',
        [([SOC_CSV, '-method', 'logit', '-seed', '0', '-JOBNAME', 'ml_test'
           ], 'ml_test_logit.svg')])
    def testContourf(self, reg, expected, tmp_dir):
        reg.read()
        reg.split()
        reg.setRegs()
        reg.contourf()
        assert os.path.isfile(expected)
