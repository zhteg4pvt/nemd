# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Machine Learning A-Z: AI, Python & R + ChatGPT Prize 2025 by Kirill Eremenko,
# Hadelin de Ponteves, SuperDataScience Team, Ligency
# https://www.udemy.com/course/machinelearning
"""
Machine learning regression.
"""
import functools
import re

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import plotutils


class Regressor(logutils.Base):
    """
    Regressor class wraps regression model.
    """
    LR = 'lr'
    SVR = 'svr'
    NAMES = {LR: 'linear', SVR: 'support vector'}

    def __init__(self, method=LR, scs=None, **kwargs):
        """
        :param method: the regression method.
        :param scs list: the scalers.
        """
        super().__init__(**kwargs)
        self.method = method
        self.scs = scs
        self.reg = None
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        match self.method:
            case self.LR:
                self.reg = linear_model.LinearRegression()
            case self.SVR:
                self.reg = svm.SVR(kernel='rbf')
        if self.method != self.SVR:
            self.scs = None

    def fit(self, xdata, ydata):
        """
        Fit the model.
        """
        self.reg.fit(self.scale(xdata), self.scaleY(ydata))

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
        """
        score = metrics.r2_score(ydata, self.predict(xdata))
        self.log(f'{self.method} score: {score}')

    def predict(self, data):
        """
        Predict.

        :param data ndarray: the input xdata.
        :return ndarray: the prediction.
        """
        return self.scaleY(self.reg.predict(self.scale(data)), inversed=True)


class Reg(logutils.Base):
    """
    Main class to run the driver.
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
        self.regs = None
        self.gridded = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
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

    def split(self):
        """
        Split into train and test.
        """
        self.xtrain, self.xtest, self.ytrain, self.ytest = model_selection.train_test_split(
            self.xdata.values,
            self.ydata.values,
            test_size=0.2,
            random_state=self.options.seed)

    def setRegs(self):
        """
        Set the regression models.
        """
        self.regs = [
            Regressor(method=x, scs=self.scs, logger=self.logger)
            for x in self.options.method
        ]
        for reg in self.regs:
            reg.fit(self.xtrain, self.ytrain)

    @functools.cached_property
    def scs(self):
        """
        The x & y scalers.

        :return list: the x & y scalers.
        """
        if Regressor.SVR in self.options.method:
            return list(self.getScaler())

    def getScaler(self):
        """
        Get the scalers.

        :return preprocessing.StandardScaler generator: the scaler.
        """
        for data in [self.xtrain, self.ytrain]:
            sc = preprocessing.StandardScaler()
            sc.fit(data)
            yield sc

    def score(self):
        """
        Calculate the scores.
        """
        for reg in self.regs:
            reg.score(self.xtest, self.ytest)

    def grid(self):
        """
        Grid the range and make predictions.
        """
        if self.xdata.shape[-1] != 1:
            return
        lim = (self.xtrain.min().item(), self.xtrain.max().item())
        gridded = np.arange(*lim, 0.1).reshape(-1, 1)
        pred = {
            f"{self.ydata.columns[0]} ({x.method})": x.predict(gridded)
            for x in self.regs
        }
        index = pd.Index(gridded.ravel(), name=self.xdata.columns[0])
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
            ax.scatter(self.xtrain, self.ytrain, color='k', label='Original')
            for col in self.gridded:
                ax.plot(self.gridded.index,
                        self.gridded[col],
                        label=rex.match(col).group(2))
            ax.set_xlabel(self.xdata.columns[0])
            ax.set_ylabel(self.ydata.columns[0])
            ax.legend()
            outfile = f"{self.options.JOBNAME}{self.FIG_EXT}"
            self.fig.savefig(outfile)
            self.log(f'Figure saved as {outfile}')
            jobutils.Job.reg(outfile)


class ArgumentParser(parserutils.Driver):
    """
    Parser with ml regression arguments.
    """
    FLAG_DATA = 'data'
    FLAG_METHOD = '-method'

    @classmethod
    def add(cls, parser, *args, **kwargs):
        """
        See parent.
        """
        parser.add_argument(cls.FLAG_DATA,
                            type=parserutils.Path.typeFile,
                            help='The csv file.')
        names = ", ".join([f"{y} ({x})" for x, y in Regressor.NAMES.items()])
        parser.add_argument(cls.FLAG_METHOD,
                            default=[Regressor.LR],
                            choices=Regressor.NAMES,
                            nargs='+',
                            help=f'Regression method: {names}')
        cls.addSeed(parser)


if __name__ == "__main__":
    logutils.Script.run(Reg, ArgumentParser(descr=__doc__), file=True)
