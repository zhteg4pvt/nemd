# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Machine Learning A-Z: AI, Python & R + ChatGPT Prize 2025 by Kirill Eremenko,
# Hadelin de Ponteves, SuperDataScience Team, Ligency
# https://www.udemy.com/course/machinelearning/?couponCode=25BBPMXINACTIVE
"""
Machine learning regression.
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import plotutils


class Reg(logutils.Base):
    """
    Machine learning regression.
    """
    FIG_EXT = '.svg'

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver': Parsed command-line options.
        """
        super().__init__(options=options, **kwargs)
        self.data = None
        self.xdata = None
        self.ydata = None
        self.x_sc = None
        self.y_sc = None
        self.reg = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.fit()
        self.plot()

    def read(self):
        """
        Read data file.
        """
        self.data = pd.read_csv(self.options.data)
        columns = self.data.select_dtypes(include=['number']).columns
        self.xdata = self.data[columns[:-1]]
        self.ydata = self.data[columns[-1:]]

    def fit(self):
        """
        Fit the model.
        """
        self.reg = svm.SVR(kernel='rbf')
        self.x_sc = preprocessing.StandardScaler()
        xdata = self.x_sc.fit_transform(self.xdata.values)
        self.y_sc = preprocessing.StandardScaler()
        ydata = self.y_sc.fit_transform(self.ydata.values)
        self.reg.fit(xdata, ydata.ravel())

    def plot(self):
        """
        Plot the prediction.
        """
        if self.xdata.shape[-1] != 1:
            return
        with plotutils.pyplot(inav=self.options.INTERAC,
                              name=self.name) as plt:
            self.fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            ax.scatter(self.xdata, self.ydata, color='red', label='data')
            xdata = self.xdata.values
            xdata = np.arange(xdata.min(), xdata.max(), 0.1).reshape(-1, 1)
            ax.plot(xdata,
                    self.predict(xdata),
                    color='blue',
                    label=self.options.method)
            ax.set_xlabel(self.xdata.columns[0])
            ax.set_ylabel(self.ydata.columns[0])
            ax.legend()
            outfile = f"{self.options.JOBNAME}{self.FIG_EXT}"
            self.fig.savefig(outfile)
            self.log(f'{self.name.capitalize()} figure saved as {outfile}')
            jobutils.Job.reg(outfile)

    def predict(self, xdata):
        """
        Predict.

        :param xdata ndarray: the input data.
        :return ndarray: the prediction.
        """
        predicted = self.reg.predict(self.x_sc.transform(xdata))
        if len(predicted.shape) == 1:
            predicted = predicted.reshape(-1, 1)
        return self.y_sc.inverse_transform(predicted)


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
        parser.add_argument(cls.FLAG_METHOD,
                            default='svr',
                            choices='svr',
                            help='svr: support vector regression')
        cls.addSeed(parser)


if __name__ == "__main__":
    logutils.Script.run(Reg, ArgumentParser(descr=__doc__), file=True)
