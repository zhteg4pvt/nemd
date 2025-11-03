# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Machine Learning A-Z: AI, Python & R + ChatGPT Prize 2025 by Kirill Eremenko,
# Hadelin de Ponteves, SuperDataScience Team, Ligency
# https://www.udemy.com/course/machinelearning
"""
Machine learning.
"""
import functools
import re

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from nemd import jobutils
from nemd import logutils
from nemd import plotutils


class Reg:
    """
    Regression & classification model wrapper.
    """
    LR = 'lr'
    SVR = 'svr'
    POLY = 'poly'
    DT = 'dt'
    RFR = 'rfr'
    LOGIT = 'logit'
    KNN = 'knn'
    NAMES = {
        LR: 'linear',
        SVR: 'support vector',
        POLY: 'polynomial',
        DT: 'decision tree',
        RFR: 'random forest',
        LOGIT: 'logistic',
        KNN: 'k-nearest neighbors'
    }
    XSCALES = {LOGIT, KNN}
    SCALES = XSCALES.union({SVR})
    CLF = {LOGIT, KNN}

    def __init__(self, method=LR, scs=None, options=None, **kwargs):
        """
        :param method: the regression method.
        :param scs list: the scalers.
        :param options 'argparse.Namespace': commandline arguments.
        """
        super().__init__(**kwargs)
        self.method = method
        self.scs = scs
        self.options = options
        self.reg = None
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        match self.method:
            case self.LR | self.POLY:
                self.reg = linear_model.LinearRegression()
            case self.SVR:
                self.reg = svm.SVR()
            case self.DT:
                self.reg = tree.DecisionTreeRegressor(
                    random_state=self.options.seed)
            case self.RFR:
                self.reg = ensemble.RandomForestRegressor(
                    n_estimators=self.options.tree_num,
                    random_state=self.options.seed)
            case self.LOGIT:
                self.reg = linear_model.LogisticRegression(
                    random_state=self.options.seed)
            case self.KNN:
                self.reg = neighbors.KNeighborsClassifier()
        if self.method not in self.SCALES:
            self.scs = []
            return
        if self.method in self.XSCALES:
            self.scs[1] = None

    def fit(self, xdata, ydata):
        """
        Fit the model.

        :param xdata ndarray: the xdata.
        :param ydata ndarray: the ydata.
        """
        self.reg.fit(self.opr(xdata), self.scaleY(ydata))

    def opr(self, data):
        """
        Perform operation on the data (e.g. polynomial, scale).

        :param data ndarray: the data to operate on.
        :return ndarray: the processed data.
        """
        return self.poly.fit_transform(data) if self.poly else self.scale(data)

    @functools.cached_property
    def poly(self):
        """
        Return the polynomial features.

        :return PolynomialFeatures: the polynomial features.
        """
        if self.method == self.POLY:
            return preprocessing.PolynomialFeatures(degree=self.options.degree)

    def scale(self, data, idx=0, inversed=False):
        """
        Scale the data.

        :param data ndarray: the data to scale.
        :param idx int: the scaler index.
        :param inversed bool: whether to inverse scale.
        """
        if not self.scs:
            return data
        sc = self.scs[idx]
        if sc is None:
            return data
        return sc.inverse_transform(data) if inversed else sc.transform(data)

    def scaleY(self, data, idx=1, **kwargs):
        """
        Scale the y data. (see scale)
        """
        return self.scale(data.reshape(-1, 1), idx=idx, **kwargs).ravel()

    def score(self, xdata, ydata):
        """
        Calculate the score.

        :param xdata ndarray: the xdata.
        :param ydata ndarray: the ydata.
        :return str, float: score name, the score.
        """
        predicted = self.predict(xdata)
        if self.method in self.CLF:
            return 'accuracy', metrics.accuracy_score(ydata, predicted)
        return 'r2', np.nan if xdata.shape[0] < 2 else metrics.r2_score(
            ydata, predicted)

    def predict(self, data):
        """
        Predict.

        :param data ndarray: the input xdata.
        :return ndarray: the prediction.
        """
        return self.scaleY(self.reg.predict(self.opr(data)), inversed=True)


class Ml(logutils.Base):
    """
    Main class to load, split, and analyze data.
    """
    FIG_EXT = '.svg'
    CSV_EXT = '.csv'
    ORIGINAL = 'original'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.xdata = None
        self.ydata = None
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.models = None
        self.fig = None
        self.ax = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.split()
        self.setRegs()
        self.score()
        self.scatter()
        self.contourf()

    def read(self):
        """
        Read data.
        """
        self.data = pd.read_csv(self.options.data)
        columns = self.data.select_dtypes(include=['number']).columns
        self.xdata = self.data[columns[:-1]]
        self.ydata = self.data[columns[-1:]]

    def split(self):
        """
        Split into train and test.
        """
        self.xtrain, self.xtest, self.ytrain, self.ytest = model_selection.train_test_split(
            self.xdata.values,
            self.ydata.values,
            test_size=self.options.test_size,
            random_state=self.options.seed)
        self.log(f"Train size: {self.xtrain.shape[0]}")
        self.log(f"Test size: {self.xtest.shape[0]}")

    def setRegs(self):
        """
        Set the regression models.
        """
        kwargs = dict(scs=self.scs, options=self.options)
        self.models = [Reg(method=x, **kwargs) for x in self.options.method]
        for reg in reversed(self.models):
            try:
                reg.fit(self.xtrain, self.ytrain)
            except ValueError as err:
                self.log(f'{reg.method} error: {err}')
                self.models.remove(reg)

    @functools.cached_property
    def scs(self):
        """
        The x & y scalers.

        :return list: the x & y scalers.
        """
        if Reg.SVR in self.options.method:
            return list(self.getScaler())
        if Reg.XSCALES.intersection(self.options.method):
            return [next(self.getScaler()), None]
        return []

    def getScaler(self):
        """
        Get the scalers.

        :return Scaler generator: the scaler.
        """
        for data in [self.xtrain, self.ytrain]:
            sc = preprocessing.StandardScaler()
            sc.fit(data)
            yield sc

    def score(self):
        """
        Calculate the scores.
        """
        for reg in self.models:
            name, score = reg.score(self.xtest, self.ytest)
            self.log(f'{name} score ({reg.method}): {score:.4g}')

    def scatter(self):
        """
        Create scatter plot.
        """
        if len(self.grids) != 1:
            return
        with plotutils.pyplot(inav=self.options.INTERAC,
                              name=self.options.method) as plt:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4.5))
            self.ax.scatter(self.xtrain,
                            self.ytrain,
                            color='k',
                            label=self.ORIGINAL)
            for method, pred in self.cols():
                self.ax.plot(self.grids[0], pred, label=Reg.NAMES[method])
            self.ax.legend()
            self.save()

    def cols(self, rex=re.compile(r'(.*) +\((.*)\)')):
        """
        Iterate over the predicted.

        :param rex: the regular expression to match words followed by brackets.
        :return tuple: label, data
        """
        for col in self.pred:
            yield rex.match(col).group(2), self.pred[col]

    @functools.cached_property
    def pred(self):
        """
        Return the gridded predictions.

        :return pd.DataFrame: the gridded xdata and predicted y.
        """
        if self.xdata.shape[-1] > 2 or not self.grids[0].size:
            return pd.DataFrame()
        grids = np.array([x.ravel() for x in self.grids]).T
        pred = {x.method: x.predict(grids) for x in self.models}
        pred = {f"{self.ydata.columns[0]} ({x})": y for x, y in pred.items()}
        index = pd.MultiIndex.from_arrays(grids.T, names=self.xdata.columns)
        pred = pd.DataFrame(pred, index=index)
        outfile = f"{self.options.JOBNAME}{self.CSV_EXT}"
        pred.to_csv(outfile)
        self.log(f'Gridded prediction saved as {outfile}')
        jobutils.Job.reg(outfile)
        return pred

    @functools.cached_property
    def grids(self, num=101):
        """
        Return the xdata grids.

        :param num int: the number of point in each dimension.
        :return tuple: the xdata grids.
        """
        lims = np.array([self.xtrain.min(axis=0), self.xtrain.max(axis=0)]).T
        args = [[*lims[x], num] for x in range(self.xdata.shape[1])]
        grids = np.array([np.linspace(*x) for x in args]).T
        return np.meshgrid(*grids.T)

    def save(self, label=''):
        """
        Save the figure.

        :param label str: the figure label extension.
        """
        self.ax.set_xlabel(self.xdata.columns[0])
        self.ax.set_ylabel(self.ydata.columns[0])
        self.ax.set_box_aspect(1)
        self.fig.tight_layout()
        outfile = f"{self.options.JOBNAME}{label and '_'}{label}{self.FIG_EXT}"
        self.fig.savefig(outfile)
        self.log(f'Figure saved as {outfile}')
        jobutils.Job.reg(outfile)

    def contourf(self, cmap='viridis'):
        """
        Create contourf plot.

        :param cmap str: the colormap
        """
        if len(self.grids) == 1:
            return
        for label, pred in self.cols():
            with plotutils.pyplot(inav=self.options.INTERAC,
                                  name=label) as plt:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
                pred = pred.values.reshape(self.grids[0].shape)
                ctrf = self.ax.contourf(*self.grids, pred, cmap=cmap)
                eles = ctrf.legend_elements(str_format=lambda x: f'{x:.4g}')
                self.ax.legend(*eles,
                               title="Level",
                               loc='center left',
                               bbox_to_anchor=(1, 0.5))
                plt.scatter(*self.xtrain.T,
                            c=self.ytrain.ravel(),
                            cmap=cmap,
                            label=self.ORIGINAL)
                self.save(label=label)
