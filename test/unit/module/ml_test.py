import os

import conftest
import numpy as np
import pytest
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
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
        reg = ml.Regression(options=options, logger=logger)
        reg.read()
        reg.split()
        return reg

    @pytest.mark.parametrize('file', [SOC_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('lr', (linear_model.LinearRegression, 0)),
                              ('poly', (linear_model.LinearRegression, 0)),
                              ('sv', (svm.SVR, 2)),
                              ('dt', (tree.DecisionTreeRegressor, 0)),
                              ('rf', (ensemble.RandomForestRegressor, 0))])
    def testSetUp(self, reg, expected):
        assert isinstance(reg.models[0].mdl, expected[0])
        assert expected[1] == len([x for x in reg.scs if x])

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method', ['lr', 'sv', 'poly', 'dt', 'rf'])
    def testFit(self, reg):
        reg.models[0].fit(reg.xtrain, reg.ytrain)
        validation.check_is_fitted(reg.models[0].mdl)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', 43), ('sv', 0.0),
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
                             [('lr', 0, False, 43), ('sv', 0, False, 0.0),
                              ('sv', 1, False, -6.5065991),
                              ('sv', 0, True, 164.50353647)])
    def testScale(self, reg, idx, inversed, expected):
        scaled = reg.models[0].scale(reg.xtrain, idx=idx, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,inversed,expected',
                             [('sv', False, 0),
                              ('sv', True, 575439538784.8147)])
    def testScaleY(self, reg, inversed, expected):
        scaled = reg.models[0].scaleY(reg.ytrain, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file,method,expected', [(POS_CSV, 'lr', 0.978),
                                                      (POS_CSV, 'sv', 0.951),
                                                      (POS_CSV, 'poly', 0.705),
                                                      (POS_CSV, 'dt', 0.586),
                                                      (POS_CSV, 'rf', 0.991)])
    def testScore(self, reg, expected):
        pred = reg.models[0].predict(reg.xtest)
        score = reg.models[0].score(reg.ytest, pred)
        np.testing.assert_almost_equal(score, expected, decimal=3)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('lr', 335474), ('sv', 193234), ('poly', 196528),
                              ('dt', 150000), ('rf', 162500)])
    def testPredict(self, reg, expected):
        assert np.allclose(reg.models[0].predict([[6.5]]), expected, rtol=0.01)


@conftest.require_src
class TestClf:

    @pytest.fixture
    def clf(self, file, method, logger):
        args = [file, '-method', method, '-seed', '0']
        options = parserutils.Clf().parse_args(args)
        clf = ml.Classification(options=options, logger=logger)
        clf.read()
        clf.split()
        return clf

    @pytest.mark.parametrize('file', [SOC_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('logit', (linear_model.LogisticRegression, 1)),
                              ('knn', (neighbors.KNeighborsClassifier, 1)),
                              ('sv', (svm.SVC, 1)),
                              ('gnb', (naive_bayes.GaussianNB, 1)),
                              ('dt', (tree.DecisionTreeClassifier, 1)),
                              ('rf', (ensemble.RandomForestClassifier, 1))])
    def testSetUp(self, clf, expected):
        assert isinstance(clf.models[0].mdl, expected[0])
        assert expected[1] == len([x for x in clf.scs if x])

    @pytest.mark.parametrize('file,method,expected',
                             [(SOC_CSV, 'logit', 0.925)])
    def testScore(self, clf, expected):
        pred = clf.models[0].predict(clf.xtest)
        score = clf.models[0].score(clf.ytest, pred)
        np.testing.assert_almost_equal(score, expected, decimal=3)


@conftest.require_src
class TestRegression:

    @pytest.fixture
    def reg(self, raw):
        raw.read()
        raw.split()
        return raw

    @pytest.fixture
    def raw(self, args, logger):
        options = parserutils.Reg().parse_args(args)
        return ml.Regression(options=options, logger=logger)

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'
                            ], 'Figure saved as name.svg'),
                          ([SEL_CSV, '-seed', '0'], 'r2 score (lr): 0.9325')])
    def testRun(self, raw, expected, tmp_dir):
        raw.run()
        raw.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected', [([POS_CSV], (10, 3)),
                                               ([SEL_CSV], (9568, 5))])
    def testRead(self, raw, expected):
        raw.read()
        assert expected == raw.data.shape

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV], (8, 2, 1, 1)),
                              ([SEL_CSV], (7654, 1914, 4, 1))])
    def testSplit(self, reg, expected):
        assert expected == (reg.xtrain.shape[0], *reg.xtest.shape,
                            reg.ytrain.shape[1])

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-method', 'lr', 'sv', 'poly', 'dt', 'rf'], 5)])
    def testModels(self, reg, expected):
        assert expected == len(reg.models)
        for reg in reg.models:
            validation.check_is_fitted(reg.mdl)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'lr'], 0),
                              ([POS_CSV, '-method', 'sv', 'poly'], 2)])
    def testScs(self, reg, expected):
        assert expected == len([x for x in reg.scs if x])

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'sv'], 2)])
    def testGetScaler(self, reg, expected):
        assert expected == len([x for x in reg.getScaler()])

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0'], 'r2 score (lr): 0.9779')])
    def testMeasure(self, reg, expected):
        reg.measure()
        reg.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'], True),
         ([SEL_CSV, '-seed', '0', '-JOBNAME', 'name'], False)])
    def testScatter(self, reg, expected, tmp_dir):
        reg.scatter()
        assert expected == os.path.isfile('name.csv')

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'name'], [['lr', 101]]),
         ([SEL_CSV, '-seed', '0', '-JOBNAME', 'name'], [])])
    def testCols(self, reg, expected, tmp_dir):
        assert expected == [[x, y.size] for x, y in reg.cols]

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-seed', '0'], (101, 1)),
                              ([SEL_CSV, '-seed', '0'], (0, 0))])
    def testPred(self, reg, expected, tmp_dir):
        assert expected == reg.pred.shape

    @pytest.mark.parametrize('args,expected', [([POS_CSV, '-seed', '0'], 1),
                                               ([SOC_CSV, '-seed', '0'], 2)])
    def testGrids(self, reg, expected):
        assert expected == len(reg.grids)

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-seed', '0', '-JOBNAME', 'ml_test'], 'ml_test.svg')])
    def testSave(self, reg, expected, tmp_dir):
        reg.scatter()
        reg.save()
        assert os.path.isfile(expected)


@conftest.require_src
class TestClassification:
    MALL_CSV = SRC.test('ml', 'mall_customers.csv')

    @pytest.fixture
    def clf(self, args, logger):
        options = parserutils.Clf().parse_args(args)
        clf = ml.Classification(options=options, logger=logger)
        clf.read()
        clf.split()
        return clf

    @pytest.mark.parametrize(
        'args,expected',
        [([SOC_CSV, '-method', 'knn', 'logit'], 2),
         ([SOC_CSV, '-method', 'logit', '-test_size', '0.997'], 0)])
    def testModels(self, clf, expected):
        assert expected == len(clf.models)
        for reg in clf.models:
            validation.check_is_fitted(reg.mdl)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'knn'], 1)])
    def testScs(self, clf, expected):
        assert expected == len([x for x in clf.scs if x])

    @pytest.mark.parametrize('args,expected', [
        ([POS_CSV, '-seed', '0', '-JOBNAME', 'ml_test'],
         ('accuracy score (logit): 0', 'ml_test_logit_cm.svg')),
        ([MALL_CSV, '-seed', '0', '-JOBNAME', 'ml_test'],
         ('accuracy score (logit): 0.025', 'ml_test_logit_cm.svg',
          'UserWarning: The number of unique classes is greater than 50% of the'
          ' number of samples. `y` could represent a regression problem, not a '
          'classification problem.'))
    ])
    def testMeasure(self, clf, expected, tmp_dir):
        clf.measure()
        clf.logger.log.assert_any_call(expected[0])
        assert os.path.isfile(expected[1])

    @pytest.mark.parametrize(
        'args,expected',
        [([SOC_CSV, '-method', 'logit', '-seed', '0', '-JOBNAME', 'ml_test'
           ], 'ml_test_logit.svg')])
    def testContourf(self, clf, expected, tmp_dir):
        clf.contourf()
        assert os.path.isfile(expected)
