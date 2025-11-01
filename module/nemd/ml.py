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
    NAMES = {
        LR: 'linear',
        SVR: 'support vector',
        POLY: 'polynomial',
        DT: 'decision tree',
        RFR: 'random forest',
        LOGIT: 'logistic'
    }
    SCALES = {SVR, LOGIT}

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
        if self.method not in self.SCALES:
            self.scs = None
        if self.method == self.LOGIT:
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
        if self.scs is None:
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
        if self.method == self.LOGIT:
            return 'accuracy', metrics.accuracy_score(ydata, predicted)
        return 'r2', metrics.r2_score(ydata, predicted)

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
        self.gridded = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.valid()
        self.split()
        self.setRegs()
        self.score()
        self.grid()
        self.plot()

    def read(self):
        """
        Read data.
        """
        self.data = pd.read_csv(self.options.data)
        columns = self.data.select_dtypes(include=['number']).columns
        self.xdata = self.data[columns[:-1]]
        self.ydata = self.data[columns[-1:]]

    def valid(self):
        """
        Valid the request models based on data.
        """
        if Reg.LOGIT in self.options.method and not self.ydata.isin(
            [0, 1]).all().all():
            self.options.method.remove(Reg.LOGIT)

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
        if Reg.LOGIT in self.options.method:
            return [next(self.getScaler()), None]

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

    def grid(self):
        """
        Grid the range and make predictions.
        """
        if self.xdata.shape[-1] != 1:
            return
        lim = (self.xtrain.min().item(), self.xtrain.max().item())
        index = pd.Index(np.arange(*lim, 0.1), name=self.xdata.columns[0])
        xdata = index.values.reshape(-1, 1)
        pred = {x.method: x.predict(xdata) for x in self.models}
        pred = {f"{self.ydata.columns[0]} ({x})": y for x, y in pred.items()}
        self.gridded = pd.DataFrame(pred, index=index)
        outfile = f"{self.options.JOBNAME}{self.CSV_EXT}"
        self.gridded.to_csv(outfile)
        self.log(f'Gridded prediction saved as {outfile}')
        jobutils.Job.reg(outfile)

    def plot(self, rex=re.compile(r'(.*) +\((.*)\)')):
        """
        Plot the gridded predictions.

        :param rex: the regular expression to match words followed by brackets.
        """
        if self.gridded is None:
            return
        with plotutils.pyplot(inav=self.options.INTERAC,
                              name=self.options.method) as plt:
            self.fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            ax.scatter(self.xtrain, self.ytrain, color='k', label='original')
            for col in self.gridded:
                label = Reg.NAMES[rex.match(col).group(2)]
                ax.plot(self.gridded.index, self.gridded[col], label=label)
            ax.set_xlabel(self.xdata.columns[0])
            ax.set_ylabel(self.ydata.columns[0])
            ax.legend()
            outfile = f"{self.options.JOBNAME}{self.FIG_EXT}"
            self.fig.savefig(outfile)
            self.log(f'Figure saved as {outfile}')
            jobutils.Job.reg(outfile)
