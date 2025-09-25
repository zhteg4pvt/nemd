# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Python functions called by the lammps script (see lmpfix.py).
"""
import math
import os
import warnings

import numpy as np
import pandas as pd
import scipy

from nemd import plotutils
from nemd import symbols


class Base:
    """
    Analyze file dumped by the LAMMPS.
    """
    DATA = 'Data'
    PNG_EXT = '.png'

    def __init__(self, filename):
        """
        :param filename str: the filename with path to load data from
        """
        self.filename = filename
        self.data = None
        self.ave = None

    def run(self):
        """
        Main method to run.
        """
        self.read()
        self.setAve()

    def read(self):
        """
        Load data from the file.
        """
        self.data = pd.read_csv(self.filename,
                                sep=r'\s+',
                                header=1,
                                na_filter=False,
                                escapechar='#',
                                index_col=0)

    def setAve(self):
        """
        Set the averaged data.
        """
        self.ave = self.data.mean()

    def getColumn(self, ending):
        """
        Get the column based the label ending str.

        :param ending str: select the column whose label ends with this string.
        :return 'pandas.core.series.Series': selected column
        """
        for label, column in self.data.items():
            if label.endswith(ending):
                return column

    @staticmethod
    def getLabel(name):
        """
        Shape the label for visualization.

        :param column str: one data column label
        :return str: to be displayed on the figure label.
        """
        column = name.removeprefix('c_').removeprefix('v_').split('_')
        return ' '.join([x.capitalize() for x in column])


class Length(Base):
    """
    Analyze xyzl (hi - lo) data.
    """
    XL = 'xl'
    YL = 'yl'
    ZL = 'zl'

    def __init__(self, filename, last_pct=0.2, ending=XL):
        """
        :param filename str: the filename with path to load data from
        :param last_pct float: the last this percentage of the data are used
        :param ending str: the data column ending with str is used.
        """
        super().__init__(filename)
        self.last_pct = last_pct
        self.ending = ending
        self.sidx = None

    def setAve(self):
        """
        Get the box length in one dimension.

        :return float: the averaged box length in one dimension.
        """
        data = self.getColumn(self.ending)
        self.sidx = math.floor(data.shape[0] * (1 - self.last_pct))
        self.ave = data[self.sidx:].mean()

    def plot(self):
        """
        To be overwritten.
        """
        with plotutils.pyplot(inav=False) as plt:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            column = self.getColumn(self.ending)
            ax.plot(self.data.index, column, label=self.DATA)
            ax.plot(self.data.index[self.sidx:],
                    column[self.sidx:],
                    'g',
                    label='Selected')
            ax.set_xlabel(self.data.index.name)
            ax.set_ylabel(self.getLabel(column.name))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            basename = os.path.basename(self.filename)
            name = symbols.PERIOD.join(basename.split(symbols.PERIOD)[:-1])
            fig.savefig(f"{name}_{self.ending}{self.PNG_EXT}")

    @classmethod
    def get(cls, filename, last_pct=0.2, ending=XL):
        """
        Get the box length in the one dimension.

        :param filename str: the filename with path to load data from
        :param last_pct float: the last this percentage of the data are used
        :param ending str: select the label ends with this string
        :return float: box length
        """
        box_length = cls(filename, last_pct=last_pct, ending=ending)
        box_length.run()
        return box_length.ave

    @classmethod
    def getX(cls, filename):
        """
        Get the box length in the x dimension.

        :param filename str: the filename with path to load data from
        :return float: box length
        """
        return cls.get(filename, ending=cls.XL)

    @classmethod
    def getY(cls, filename):
        """
        Get the box length in the y dimension.

        :param filename str: the filename with path to load data from
        :return float: box length
        """
        return cls.get(filename, ending=cls.YL)

    @classmethod
    def getZ(cls, filename):
        """
        Get the box length in the z dimension.

        :param filename str: the filename with path to load data from
        :return float: box length
        """
        return cls.get(filename, ending=cls.ZL)


class Press(Base):
    """
    Analyze pressure data dumped by the LAMMPS.
    """
    PRESS = 'press'

    def setAve(self):
        """
        Set the averaged data.
        """
        self.ave = self.getColumn(self.PRESS).mean()

    @classmethod
    def get(cls, filename):
        """
        Get the averaged pressure.

        :param filename str: the filename with path to load data from
        :return float: averaged pressure.
        """
        press = cls(filename)
        press.run()
        return press.ave


class Factor(Press):
    """
    Calculate the volume scale factor.
    """

    def __init__(self, press, *args, **kwargs):
        """
        :param press float: target pressure.
        """
        super().__init__(*args, **kwargs)
        self.press = press

    def run(self):
        """
        Main method to run.
        """
        super().run()
        delta = self.getColumn(self.PRESS).std()
        if self.press > self.ave + delta:
            return 0.995
        elif self.press < self.ave - delta:
            return 1.005
        else:
            return 1

    @classmethod
    def getVol(cls, press, filename):
        """
        Get the volume scale factor so that the pressure is expected to approach the
        target by scaling the volume.

        :param press float: the target pressure.
        :param filename str: the filename with path to load data from.
        :return float: the scale factor of the volume.
        """
        return cls(press, filename).run()

    @classmethod
    def getBdry(cls, press, filename):
        """
        Get the boundary scale factor so that the pressure is expected to approach
        the target by scaling the boundary length.

        :param press float: the target pressure.
        :param filename str: the filename with path to load data from.
        :return float: the scale factor of the volume.
        """
        return cls.getVol(press, filename)**(1 / 3)


class Modulus(Press):
    """
    Analyze press_vol (pressure & volume) data.
    """
    VOL = 'vol'
    MODULUS = 'modulus'
    STD_DEV = '_(Std_Dev)'
    SMOOTHED = '_(Smoothed)'

    def __init__(self, filename, rec_num):
        """
        :param filename str: the filename with path to load data from
        :param rec_num int: the recording number of each cycle.
        """
        super().__init__(filename)
        self.rec_num = rec_num
        self.ave = pd.DataFrame()
        self.modulus = None

    def run(self):
        """
        Main method to run.
        """
        super().run()
        self.setModulus()
        self.plot()

    def setAve(self):
        """
        Set the averaged data.
        """
        for column in self.data.columns:
            col = self.data[column].values
            mod = col.shape[0] % self.rec_num
            if mod:
                col = np.concatenate(([np.nan], col))
            data = col.reshape(-1, self.rec_num)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ave[column] = np.nanmean(data, axis=0)
                self.ave[column + self.STD_DEV] = np.nanstd(data, axis=0)
            smoothed_lb = column + self.SMOOTHED
            window = int(self.rec_num / 10)
            self.ave[smoothed_lb] = scipy.signal.savgol_filter(
                self.ave[column], window, 3)

    def setModulus(self, lower_bound=10):
        """
        Set the bulk modulus.

        :param lower_bound: the lower boundary of the modulus.
        :return float: the bulk modulus from cycles.
        """
        press_lb = self.getColumn(self.PRESS).name + self.SMOOTHED
        press_delta = self.ave[press_lb].max() - self.ave[press_lb].min()
        vol_lb = self.getColumn(self.VOL).name + self.SMOOTHED
        vol_delta = self.ave[vol_lb].max() - self.ave[vol_lb].min()
        modulus = press_delta / vol_delta * self.ave[vol_lb].mean()
        self.modulus = max([modulus, lower_bound])

    def plot(self):
        """
        Plot the data and save the figure.
        """
        with plotutils.pyplot(inav=False) as plt:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            for idx, (axis, column) in enumerate(zip(axes, self.data.columns)):
                self.subplot(axis, column)
                if not idx:
                    num = round(self.data.shape[0] / self.rec_num)
                    axis.set_title(f"Sinusoidal Deformation ({num} cycles)")
            basename = os.path.basename(self.filename)
            root, _ = os.path.splitext(basename)
            fig.savefig(f"{root}_{self.MODULUS}{self.PNG_EXT}")

    def subplot(self, ax, column):
        """
        Plot the data corresponding to column label on the axis.

        :param ax 'matplotlib.axes._axes.Axes':the axis to plot
        :param column str: the column of the data
        """
        ax.plot(self.ave.index, self.ave[column], label=self.DATA)
        smoothed_lb = column + self.SMOOTHED
        ax.plot(self.ave.index, self.ave[smoothed_lb], label="Smoothed")
        std_dev = self.ave[column + self.STD_DEV]
        if std_dev.any():
            lbndry = self.ave[column] - std_dev
            ubndry = self.ave[column] + std_dev
            ax.fill_between(self.ave.index,
                            lbndry,
                            ubndry,
                            alpha=0.5,
                            label="SD")
        ax.set_xlabel(self.data.index.name)
        ax.set_ylabel(self.getLabel(column))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    @classmethod
    def get(cls, filename, rec_num):
        """
        Get the bulk modulus.

        :param filename str: the filename with path to load data from
        :param rec_num int: the recording number of each cycle.
        :return float: the bulk modulus.
        """
        modulus = cls(filename, rec_num)
        modulus.run()
        return modulus.modulus
