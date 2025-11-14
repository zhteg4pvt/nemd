import os

import conftest
import numpy as np
import pytest
from sklearn import cluster
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
SOC_CSV = SRC.test('ml', 'social_network_ads.csv')
MALL_CSV = SRC.test('ml', 'mall_customers.csv')


class Base:
    Coll = None

    @pytest.fixture
    def coll(self, clus):
        clus.split()
        return clus

    @pytest.fixture
    def clus(self, raw):
        raw.read()
        return raw

    @pytest.fixture
    def raw(self, args, logger):
        Coll = self.Coll or getattr(
            ml, self.__class__.__name__.removeprefix('Test'))
        Parser = getattr(parserutils, Coll.Model.__name__)
        options = Parser().parse_args(args + ['-seed', '0', '-JOBNAME', 'ml'])
        return Coll(options=options, logger=logger)

    @pytest.fixture
    def args(self, file, method):
        return [file, '-method', method]


@conftest.require_src
class TestClus(Base):
    Coll = ml.Cluster

    @pytest.mark.parametrize('file', [MALL_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('k-means', (cluster.KMeans, 0)),
                              ('hca', (cluster.AgglomerativeClustering, 0))])
    def testMdl(self, clus, expected):
        assert isinstance(clus.models[0].mdl, expected[0])
        assert expected[1] == len([x for x in clus.scs if x])

    @pytest.mark.parametrize('file', [MALL_CSV])
    @pytest.mark.parametrize('method', ['k-means', 'hca'])
    def testFit(self, clus):
        clus.models[0].fit(clus.xtrain)
        validation.check_is_fitted(clus.models[0].mdl)


@conftest.require_src
class TestReg(Base):
    Coll = ml.Regression

    @pytest.mark.parametrize('file', [SOC_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('lr', (linear_model.LinearRegression, 0)),
                              ('poly', (linear_model.LinearRegression, 0)),
                              ('sv', (svm.SVR, 2)),
                              ('dt', (tree.DecisionTreeRegressor, 0)),
                              ('rf', (ensemble.RandomForestRegressor, 0))])
    def testMdl(self, coll, expected):
        assert isinstance(coll.models[0].mdl, expected[0])
        assert expected[1] == len([x for x in coll.scs if x])

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method', ['lr', 'sv', 'poly', 'dt', 'rf'])
    def testFit(self, coll):
        coll.models[0].fit(coll.xtrain, coll.ytrain)
        validation.check_is_fitted(coll.models[0].mdl)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', 43), ('sv', 0.0),
                                                 ('poly', 346.0)])
    def testOpr(self, coll, expected):
        assert expected == coll.models[0].opr(coll.xtrain).sum()

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected', [('lr', False),
                                                 ('poly', True)])
    def testPoly(self, coll, expected):
        assert expected == bool(coll.models[0].poly)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,idx,inversed,expected',
                             [('lr', 0, False, 43), ('sv', 0, False, 0.0),
                              ('sv', 1, False, -6.5065991),
                              ('sv', 0, True, 164.50353647)])
    def testScale(self, coll, idx, inversed, expected):
        scaled = coll.models[0].scale(coll.xtrain, idx=idx, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,inversed,expected',
                             [('sv', False, 0),
                              ('sv', True, 575439538784.8147)])
    def testScaleY(self, coll, inversed, expected):
        scaled = coll.models[0].scaleY(coll.ytrain, inversed=inversed)
        np.testing.assert_almost_equal(scaled.sum(), expected)

    @pytest.mark.parametrize('file,method,expected', [(POS_CSV, 'lr', 0.978),
                                                      (POS_CSV, 'sv', 0.951),
                                                      (POS_CSV, 'poly', 0.705),
                                                      (POS_CSV, 'dt', 0.586),
                                                      (POS_CSV, 'rf', 0.991)])
    def testScore(self, coll, expected):
        pred = coll.models[0].predict(coll.xtest)
        score = coll.models[0].score(coll.ytest, pred)
        np.testing.assert_almost_equal(score, expected, decimal=3)

    @pytest.mark.parametrize('file', [POS_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('lr', 335474), ('sv', 193234), ('poly', 196528),
                              ('dt', 150000), ('rf', 162500)])
    def testPredict(self, coll, expected):
        assert np.allclose(coll.models[0].predict([[6.5]]),
                           expected,
                           rtol=0.01)


@conftest.require_src
class TestClf(Base):
    Coll = ml.Classification

    @pytest.mark.parametrize('file', [SOC_CSV])
    @pytest.mark.parametrize('method,expected',
                             [('logit', (linear_model.LogisticRegression, 1)),
                              ('knn', (neighbors.KNeighborsClassifier, 1)),
                              ('sv', (svm.SVC, 1)),
                              ('gnb', (naive_bayes.GaussianNB, 1)),
                              ('dt', (tree.DecisionTreeClassifier, 1)),
                              ('rf', (ensemble.RandomForestClassifier, 1))])
    def testMdl(self, coll, expected):
        assert isinstance(coll.models[0].mdl, expected[0])
        assert expected[1] == len([x for x in coll.scs if x])

    @pytest.mark.parametrize('file,method,expected',
                             [(SOC_CSV, 'logit', 0.925)])
    def testScore(self, coll, expected):
        pred = coll.models[0].predict(coll.xtest)
        score = coll.models[0].score(coll.ytest, pred)
        np.testing.assert_almost_equal(score, expected, decimal=3)


@conftest.require_src
class TestMl(Base):

    @pytest.fixture
    def coll_fig(self, coll):
        with coll.subplots():
            pass
        return coll

    @pytest.fixture
    def coll(self, raw):
        raw.read()
        raw.xtrain = raw.xdata
        return raw

    @pytest.mark.parametrize('args,expected', [([POS_CSV], (10, 3, 2))])
    def testRead(self, raw, expected):
        raw.read()
        assert expected == (*raw.data.shape, raw.xdata.shape[1])

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV, '-method', 'k-means', 'hca'], 2)])
    def testModels(self, coll, expected):
        assert expected == len(coll.models)
        for coll in coll.models:
            validation.check_is_fitted(coll.mdl)

    @pytest.mark.parametrize('args,expected', [([POS_CSV], 0)])
    def testScs(self, raw, expected):
        assert expected == len(raw.scs)

    @pytest.mark.parametrize('args,expected', [([POS_CSV], 'ml.svg')])
    def testSave(self, coll_fig, expected, tmp_dir):
        coll_fig.save()
        assert os.path.isfile(expected)

    @pytest.mark.parametrize('args,expected', [([POS_CSV], 'Level')])
    def testSetLayout(self, coll_fig, expected, tmp_dir):
        coll_fig.setLayout()
        assert expected == coll_fig.ax.get_xlabel()

    @pytest.mark.parametrize('args', [[POS_CSV]])
    def testSubplots(self, coll_fig):
        assert coll_fig.fig
        assert coll_fig.ax


@conftest.require_src
class TestCluster(Base):

    @pytest.mark.parametrize('args,expected', [([MALL_CSV], 'ml_k-means.svg')])
    def testRun(self, raw, expected, tmp_dir):
        raw.run()
        assert os.path.exists(expected)

    @pytest.mark.parametrize('args,expected', [([MALL_CSV], (200, 2))])
    def testRead(self, clus, expected):
        assert clus.xtrain.shape

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV], 8),
                              ([MALL_CSV, '-method', 'hca'], 2),
                              ([MALL_CSV, '-cluster_num', 'auto'], 5)])
    def testSetKMeanNum(self, clus, expected, tmp_dir):
        clus.setKMeanNum()
        assert expected == clus.options.cluster_num[0]

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV, '-max_num', '7'], 8)])
    def testInertia(self, clus, expected):
        assert expected == clus.inertia.shape[0]

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV, '-max_num', '7'], 8)])
    def testGetInertia(self, clus, expected):
        assert expected == len(list(clus.getInertia()))

    @pytest.mark.parametrize('args,array,expected',
                             [([MALL_CSV], [10, 8, 6, 4, 3, 2, 1], 3)])
    def testGetIdx(self, clus, array, expected):
        assert expected == clus.getIdx(np.array(array))

    @pytest.mark.parametrize(
        'args,expected',
        [([MALL_CSV, '-method', 'k-means'], 8),
         ([MALL_CSV, '-method', 'hca'], 2),
         ([MALL_CSV, '-method', 'hca', '-cluster_num', 'auto'], 5)])
    def testSetHcaNum(self, clus, expected, tmp_dir):
        clus.setHcaNum()
        assert expected == clus.options.cluster_num[0]

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV, '-method', 'k-means', 'hca'], 2)])
    def testScatter(self, clus, expected, tmp_dir):
        clus.scatter()
        assert expected == len(clus.ax.collections)

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV, '-method', 'k-means', 'hca'],
                               (200, 2))])
    def testPred(self, clus, expected, tmp_dir):
        assert expected == clus.pred.shape

    @pytest.mark.parametrize('args,expected',
                             [([MALL_CSV], 'Spending Score (1-100)')])
    def testSetLayout(self, clus, expected, tmp_dir):
        with clus.subplots():
            clus.setLayout()
        assert expected == clus.ax.get_ylabel()


@conftest.require_src
class TestRegression(Base):
    SEL_CSV = SRC.test('ml', 'model_selection.csv')

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0'], 'Figure saved as ml.svg'),
                          ([SEL_CSV, '-seed', '0'], 'r2 score (lr): 0.9325')])
    def testRun(self, raw, expected, tmp_dir):
        raw.run()
        raw.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV], (8, 2, 1, 1)),
                              ([SEL_CSV], (7654, 1914, 4, 1))])
    def testSplit(self, coll, expected):
        assert expected == (coll.xtrain.shape[0], *coll.xtest.shape,
                            coll.ytrain.shape[1])

    @pytest.mark.parametrize(
        'args,expected',
        [([POS_CSV, '-method', 'lr', 'sv', 'poly', 'dt', 'rf'], 5)])
    def testModels(self, coll, expected):
        assert expected == len(coll.models)
        for coll in coll.models:
            validation.check_is_fitted(coll.mdl)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'lr'], 0),
                              ([POS_CSV, '-method', 'sv', 'poly'], 2)])
    def testScs(self, coll, expected):
        assert expected == len([x for x in coll.scs if x])

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'sv'], 2)])
    def testGetScaler(self, coll, expected):
        assert expected == len([x for x in coll.getScaler()])

    @pytest.mark.parametrize(
        'args,expected', [([POS_CSV, '-seed', '0'], 'r2 score (lr): 0.9779')])
    def testMeasure(self, coll, expected):
        coll.measure()
        coll.logger.log.assert_called_with(expected)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-seed', '0'], True),
                              ([SEL_CSV, '-seed', '0'], False)])
    def testScatter(self, coll, expected, tmp_dir):
        coll.scatter()
        assert expected == os.path.isfile('ml.csv')

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-seed', '0'], (101, 1)),
                              ([SEL_CSV, '-seed', '0'], (0, 0))])
    def testPred(self, coll, expected, tmp_dir):
        assert expected == coll.pred.shape

    @pytest.mark.parametrize('args,expected', [([POS_CSV, '-seed', '0'], 1),
                                               ([SOC_CSV, '-seed', '0'], 2)])
    def testGrids(self, coll, expected):
        assert expected == len(coll.grids)


@conftest.require_src
class TestClassification(Base):

    @pytest.mark.parametrize(
        'args,expected',
        [([SOC_CSV, '-method', 'knn', 'logit'], 2),
         ([SOC_CSV, '-method', 'logit', '-test_size', '0.997'], 0)])
    def testModels(self, coll, expected):
        assert expected == len(coll.models)
        for reg in coll.models:
            validation.check_is_fitted(reg.mdl)

    @pytest.mark.parametrize('args,expected',
                             [([POS_CSV, '-method', 'knn'], 1)])
    def testScs(self, coll, expected):
        assert expected == len([x for x in coll.scs if x])

    @pytest.mark.parametrize('args,expected', [
        ([POS_CSV, '-seed', '0'],
         ('accuracy score (logit): 0', 'ml_logit_cm.svg')),
        ([MALL_CSV, '-seed', '0'],
         ('accuracy score (logit): 0.025', 'ml_logit_cm.svg',
          'UserWarning: The number of unique classes is greater than 50% of the'
          ' number of samples. `y` could represent a regression problem, not a '
          'classification problem.'))
    ])
    def testMeasure(self, coll, expected, tmp_dir):
        coll.measure()
        coll.logger.log.assert_any_call(expected[0])
        assert os.path.isfile(expected[1])

    @pytest.mark.parametrize(
        'args,expected',
        [([SOC_CSV, '-method', 'logit', '-seed', '0'], 'ml_logit.svg')])
    def testContourf(self, coll, expected, tmp_dir):
        coll.contourf()
        assert os.path.isfile(expected)
