# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Machine Learning A-Z: AI, Python & R + ChatGPT Prize 2025 by Kirill Eremenko,
# Hadelin de Ponteves, SuperDataScience Team, Ligency
# https://www.udemy.com/course/machinelearning
"""
Machine learning.
"""
import contextlib
import functools
import types

import numpy as np
import pandas as pd
import scipy
from sklearn import cluster
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from nemd import jobutils
from nemd import logutils
from nemd import plotutils


class Reg:
    """
    Regression model wrapper.
    """
    LR = 'lr'
    POLY = 'poly'
    SV = 'sv'
    DT = 'dt'
    RF = 'rf'
    SHARED = {SV: 'support vector', DT: 'decision tree', RF: 'random forest'}
    NAMES = {LR: 'linear', POLY: 'polynomial', **SHARED}

    def __init__(self, method, options=None, scs=None):
        """
        :param method: the regression method.
        :param options 'argparse.Namespace': commandline arguments.
        :param scs list: the scalers.
        """
        self.method = method
        self.options = options
        self.scs = scs
        self.mdl = None
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        match self.method:
            case self.LR | self.POLY:
                self.mdl = linear_model.LinearRegression()
            case self.SV:
                self.mdl = svm.SVR()
            case self.DT:
                self.mdl = tree.DecisionTreeRegressor(
                    random_state=self.options.seed)
            case self.RF:
                self.mdl = ensemble.RandomForestRegressor(
                    n_estimators=self.options.tree_num,
                    random_state=self.options.seed)
        if self.method != self.SV:
            self.scs = []

    def fit(self, xdata, ydata):
        """
        Fit the model.

        :param xdata ndarray: the xdata.
        :param ydata ndarray: the ydata.
        """
        self.mdl.fit(self.opr(xdata), self.scaleY(ydata))

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
        if data is None:
            return
        return self.scale(data.reshape(-1, 1), idx=idx, **kwargs).ravel()

    def score(self, data, pred):
        """
        Calculate the score.

        :param data ndarray: the ydata.
        :param pred ndarray: the predicted.
        :return float: the score.
        """
        return np.nan if data.shape[0] < 2 else metrics.r2_score(data, pred)

    def predict(self, data):
        """
        Predict.

        :param data ndarray: the input xdata.
        :return ndarray: the prediction.
        """
        return self.scaleY(self.mdl.predict(self.opr(data)), inversed=True)


class Clus(Reg):
    """
    Cluster model wrapper.
    """
    K_MEANS = 'k-means'
    HCA = 'hca'
    NAMES = {K_MEANS: 'k-means cluster', HCA: 'hierarchical cluster analysis'}

    def setUp(self):
        """
        Set up.
        """
        n_clusters = self.options.cluster_num[self.options.method.index(
            self.method)]
        match self.method:
            case self.K_MEANS:
                self.mdl = cluster.KMeans(n_clusters=n_clusters,
                                          random_state=self.options.seed)
            case self.HCA:
                self.mdl = cluster.AgglomerativeClustering(
                    n_clusters=n_clusters)

    def predict(self, *args):
        """
        See parent.

        :raise AttributeError: when the model cannot predict.
        """
        if self.method == self.HCA:
            raise AttributeError(f"{self.method} focuses on grouping existing "
                                 "data and cannot predict.")
        return super().predict(*args)


class Clf(Reg):
    """
    Classification model wrapper.
    """
    LOGIT = 'logit'
    KNN = 'knn'
    GNB = 'gnb'
    NAMES = {
        LOGIT: 'logistic',
        KNN: 'k-nearest neighbors',
        GNB: 'gaussian naive bayes',
        **Reg.SHARED
    }

    def setUp(self):
        """
        Set up.
        """
        match self.method:
            case self.LOGIT:
                self.mdl = linear_model.LogisticRegression(
                    random_state=self.options.seed)
            case self.KNN:
                self.mdl = neighbors.KNeighborsClassifier()
            case self.SV:
                self.mdl = svm.SVC(random_state=self.options.seed)
            case self.GNB:
                self.mdl = naive_bayes.GaussianNB()
            case self.DT:
                self.mdl = tree.DecisionTreeClassifier(
                    random_state=self.options.seed)
            case self.RF:
                self.mdl = ensemble.RandomForestClassifier(
                    n_estimators=self.options.tree_num,
                    random_state=self.options.seed)

    def score(self, data, pred):
        """
        Calculate the score.

        :param data ndarray: the ydata.
        :param pred ndarray: the predicted.
        :return float: the score.
        """
        return metrics.accuracy_score(data, pred)


class Base(logutils.Base):
    """
    Base class to load data.
    """
    FIG_EXT = '.svg'
    CSV_EXT = '.csv'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.xdata = None
        self.ydata = None
        self.xtrain = None
        self.ytrain = None
        self.fig = None
        self.ax = None

    def read(self):
        """
        Read data.
        """
        self.data = pd.read_csv(self.options.data)
        self.xdata = self.data.select_dtypes(include=['number'])

    @functools.cached_property
    def models(self):
        """
        Return the models.

        :return list: the fitted models.
        """
        kwargs = dict(scs=self.scs, options=self.options)
        models = [self.Model(x, **kwargs) for x in self.options.method]
        for model in reversed(models):
            try:
                model.fit(self.xtrain, self.ytrain)
            except ValueError as err:
                self.log(f'{model.method} error: {err}')
                models.remove(model)
        return models

    @functools.cached_property
    def scs(self):
        """
        The x & y scalers.

        :return list: the x & y scalers.
        """
        return []

    def save(self, label='', **kwargs):
        """
        Save the figure.

        :param label str: the figure label extension.
        """
        self.setLayout(**kwargs)
        outfile = f"{self.options.JOBNAME}{label and '_'}{label}{self.FIG_EXT}"
        self.fig.savefig(outfile)
        self.log(f'Figure saved as {outfile}')
        jobutils.Job.reg(outfile)

    def setLayout(self, xlabel=None):
        """
        Set the layout.

        :param xlabel str: the xlabel.
        """
        self.ax.set_xlabel(xlabel or self.xdata.columns[0])
        self.ax.set_box_aspect(1)
        self.fig.tight_layout()

    @contextlib.contextmanager
    def subplots(self, figsize=(6, 4.5), **kwargs):
        """
        Create subplot.

        :param figure tuple: the figure size.
        """
        with plotutils.pyplot(inav=self.options.INTERAC, **kwargs) as plt:
            self.fig, self.ax = plt.subplots(figsize=figsize)
            yield


class Cluster(Base):
    """
    Main class to analyze data via clustering.
    """
    Model = Clus

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.setKMeanNum()
        self.setHcaNum()
        self.scatter()

    def read(self):
        """
        Read data.
        """
        super().read()
        self.xtrain = self.xdata

    def setKMeanNum(self, label='elbow'):
        """
        Set the cluster num for k-means method.

        :param label str: the label.
        """
        try:
            index = self.options.method.index(Clus.K_MEANS)
        except ValueError:
            return
        if self.options.cluster_num[index]:
            return
        idx = int(self.parse(self.inertia.index.name)[1])
        self.options.cluster_num[index] = self.inertia.index[idx]
        self.log(f'k-means cluster num: {self.options.cluster_num[index]}')
        with self.subplots(name=label):
            self.ax.scatter(self.inertia.index, self.inertia)
            elbow = self.inertia.iloc[idx]
            self.ax.scatter(elbow.name, elbow.values, label=label)
            self.ax.legend()
            self.save(label=label,
                      xlabel='Cluster Number (n)',
                      ylabel='Inertia')

    @functools.cached_property
    def inertia(self, name='cluster num'):
        """
        Return inertia vs cluster num with elbow point.

        :param name str: the index name.
        :return pd.DataFrame: inertia vs cluster num.
        """
        dtype = np.dtype([(name, int), ('inertia', float)])
        inertia = np.fromiter(self.getInertia(), dtype=dtype)
        inertia = pd.DataFrame(inertia)
        inertia.set_index(name, inplace=True)
        arr = inertia.values.flatten()
        inertia.index.name = f"{name} ({self.getIdx(arr[:-1] - arr[1:])})"
        return inertia

    def getIdx(self, gaps, wdw=10, idx=1):
        """
        Get the index of the largest gap in the small gap group.

        :param gaps np.ndarray: the gaps in descending orders.
        :param wdw int: the averaging window length of the gaps (slopes).
        :param idx int: the initial next gap id.
        """
        num = len(gaps)
        # a -- gap0 -- b -- gap1 -- c: gap1 ^ 2 > gap0 * gap1
        while idx < num and \
                gaps[idx]**2 > gaps[:idx][-wdw:].mean() * gaps[idx:][:wdw].mean():
            idx += 1
        return idx

    def getInertia(self):
        """
        Get the inertia vs cluster num.

        :return tuple of (int, float): cluster num, inertia
        """
        index = self.options.method.index(Clus.K_MEANS)
        max_num = min(self.options.max_num + 1, self.xtrain.shape[0])
        options = types.SimpleNamespace(**vars(self.options))
        for num in np.arange(max_num) + 1:
            options.cluster_num[index] = num
            mdl = self.Model(Clus.K_MEANS, options=options)
            with self.catchWarnings():
                mdl.fit(self.xtrain, self.ytrain)
            yield num, mdl.mdl.inertia_

    def setHcaNum(self, label='dendrogram'):
        """
        Set the cluster num for hca method.
        """
        try:
            index = self.options.method.index(Clus.HCA)
        except ValueError:
            return
        if self.options.cluster_num[index]:
            return
        linkage = scipy.cluster.hierarchy.linkage(self.xtrain, method='ward')
        with self.subplots(name=label):
            dendrogram = scipy.cluster.hierarchy.dendrogram(linkage,
                                                            ax=self.ax,
                                                            no_labels=True)
            self.save(label=label, xlabel='Data', ylabel='Euclidean Distances')
        ys = np.sort([x[1] for x in dendrogram['dcoord']])
        seps = np.flip(ys[1:] - ys[:-1])
        gaps = np.flip(np.sort(seps))
        gap = gaps[self.getIdx(gaps) - 1]
        self.options.cluster_num[index] = np.where(seps == gap)[0].max() + 2
        self.log(f'hca cluster num: {self.options.cluster_num[index]}')

    def scatter(self):
        """
        Create scatter plot.
        """
        if self.xtrain.shape[1] > 2:
            return
        for name, pred in self.pred.items():
            with self.subplots(name=name):
                xtrain = self.xtrain.values
                for label in np.unique(pred):
                    self.ax.scatter(*xtrain[pred.values == label, :].T)
                self.save(label=name)

    @functools.cached_property
    def pred(self):
        """
        Return the categories.

        :return pd.DataFrame: the gridded features and categories.
        """
        if self.xdata.shape[-1] > 2:
            return pd.DataFrame()
        pred = {x.method: x.mdl.labels_ for x in self.models}
        pred = pd.DataFrame(pred, index=pd.MultiIndex.from_frame(self.xtrain))
        outfile = f"{self.options.JOBNAME}{self.CSV_EXT}"
        pred.to_csv(outfile)
        self.log(f'Gridded prediction saved as {outfile}')
        jobutils.Job.reg(outfile)
        return pred

    def setLayout(self, ylabel=None, **kwargs):
        """
        See parent.

        :param ylabel str: the ylabel
        """
        self.ax.set_ylabel(ylabel or self.xdata.columns[1])
        super().setLayout(**kwargs)


class Regression(Base):
    """
    Main class to analyze data via regression.
    """
    ORIGINAL = 'original'
    Model = Reg

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xtest = None
        self.ytest = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.split()
        self.measure()
        self.scatter()
        self.contourf()

    def read(self):
        """
        See parent.
        """
        super().read()
        self.ydata = self.xdata[self.xdata.columns[-1:]]
        self.xdata = self.xdata[self.xdata.columns[:-1]]

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

    @functools.cached_property
    def scs(self):
        """
        The x & y scalers.

        :return list: the x & y scalers.
        """
        return list(self.getScaler()) if Reg.SV in self.options.method else []

    def getScaler(self):
        """
        Get the scalers.

        :return Scaler generator: the scaler.
        """
        for data in [self.xtrain, self.ytrain]:
            sc = preprocessing.StandardScaler()
            sc.fit(data)
            yield sc

    def measure(self):
        """
        Measure the model quality.
        """
        for reg in self.models:
            pred = reg.predict(self.xtest)
            self.log(f'r2 score ({reg.method}): '
                     f'{reg.score(self.ytest, pred):.4g}')

    def scatter(self):
        """
        Create scatter plot.
        """
        if len(self.grids) != 1:
            return
        with self.subplots(name=self.options.method):
            self.ax.scatter(self.xtrain,
                            self.ytrain,
                            color='k',
                            label=self.ORIGINAL)
            for col in self.pred:
                self.ax.plot(self.grids[0],
                             self.pred[col],
                             label=self.Model.NAMES[self.parse(col)[1]])
            self.ax.legend()
            self.save()

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
        if self.xtrain.shape[1] > 2:
            return ()
        lims = np.array([self.xtrain.min(axis=0), self.xtrain.max(axis=0)]).T
        args = [[*lims[x], num] for x in range(self.xdata.shape[1])]
        grids = np.array([np.linspace(*x) for x in args]).T
        return np.meshgrid(*grids.T)

    def setLayout(self):
        """
        See parent.
        """
        self.ax.set_ylabel(self.ydata.columns[0])
        super().setLayout()

    def contourf(self, cmap='viridis'):
        """
        Create contourf plot.

        :param cmap str: the colormap
        """
        if len(self.grids) != 2:
            return
        for col in self.pred:
            method = self.parse(col)[1]
            with self.subplots(name=method, figsize=(7, 4.5)):
                pred = self.pred[col]
                pred = pred.values.reshape(self.grids[0].shape)
                ctrf = self.ax.contourf(*self.grids, pred, cmap=cmap)
                eles = ctrf.legend_elements(str_format=lambda x: f'{x:.4g}')
                self.ax.legend(*eles,
                               title="Level",
                               loc='center left',
                               bbox_to_anchor=(1, 0.5))
                self.ax.scatter(*self.xtrain.T,
                                c=self.ytrain.ravel(),
                                cmap=cmap,
                                label=self.ORIGINAL)
                self.save(label=method)


class Classification(Regression):
    """
    Main class to analyze data via classification.
    """
    Model = Clf

    @functools.cached_property
    def scs(self):
        """
        See parent.
        """
        return [next(self.getScaler()), None]

    def measure(self):
        """
        See parent.
        """
        for model in self.models:
            pred = model.predict(self.xtest)
            with self.catchWarnings():
                score = model.score(self.ytest, pred)
            self.log(f'accuracy score ({model.method}): {score:.4g}')
            with self.subplots(name=model.method):
                with self.catchWarnings():
                    metrics.ConfusionMatrixDisplay.from_predictions(self.ytest,
                                                                    pred,
                                                                    ax=self.ax)
                self.save(label=f"{model.method}_cm")
