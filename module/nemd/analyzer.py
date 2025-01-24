# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Lammps trajectory and log analyzers.
"""
import functools
import math
import os
import re
import types

import numpy as np
import pandas as pd
import scipy

from nemd import constants
from nemd import dist
from nemd import jobutils
from nemd import logutils
from nemd import molview
from nemd import plotutils
from nemd import symbols


class Base(logutils.Base):
    """
    The base class subclassed by analyzers.
    """

    NAME = 'base'
    DESCR = NAME
    UNIT = None
    LABEL = f'{NAME.capitalize()} ({UNIT})'
    LABEL_RE = re.compile('(.*) +\((.*)\)')
    ERR_LB = symbols.SD_PREFIX
    DATA_EXT = '.csv'
    FIG_EXT = '.png'
    FLOAT_FMT = '%.4g'

    def __init__(self, df_reader=None, options=None, logger=None):
        """
        :param df_reader: data file reader containing structural information
        :type df_reader: 'nemd.oplsua.Reader'
        :param options: the options from command line
        :type options: 'argparse.Namespace'
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.df_reader = df_reader
        self.options = options
        self.data = None
        self.sidx = 0
        self.eidx = None
        self.result = None
        self.outfile = self.getFilename(self.options)
        jobutils.add_outfile(self.outfile)

    @classmethod
    def getFilename(cls, options):
        """
        Get the filename based on the command line options.

        :param options 'argparse.Namespace' or str: command line options
        :return str: the filename of the data file.
        """
        if isinstance(options, str):
            return f"{options}_{cls.NAME}{cls.DATA_EXT}"
        if not hasattr(options, 'jobs'):
            return f"{options.jobname}_{cls.NAME}{cls.DATA_EXT}"
        filename = f"{options.jobname}_{cls.NAME}_{options.id}{cls.DATA_EXT}"
        return os.path.join(options.dir, filename)

    def run(self):
        """
        Main method to run the analyzer.
        """
        self.readData()
        self.setData()
        self.saveData()
        self.fit()
        self.plot()

    def readData(self):
        """
        Read the output files from independent runs to set the data.
        """
        if not hasattr(self.options, 'jobs'):
            return
        filename = self.getFilename(self.options.name)
        files = [x.fn(filename) for x in self.options.jobs]
        datas = [pd.read_csv(x, index_col=0) for x in files]
        # 'Time (ps)': None; 'Time (ps) (0)': '0'; 'Time (ps) (0, 1)': '0, 1'
        names = [self.parseIndex(x.index.name) for x in datas]
        label, unit, sidx, eidx = names[0]
        if len(datas) == 1:
            # One single run
            self.data = datas[0]
            self.sidx, self.eidx = sidx, eidx
            return
        # Runs with different randomized seeds are truncated backwards
        num = min([x.shape[0] for x in datas])
        datas = [x.iloc[-num:] for x in datas]
        # Averaged index
        indexes = [x.index.to_numpy().reshape(-1, 1) for x in datas]
        index_ave = np.concatenate(indexes, axis=1).mean(axis=1)
        _, _, self.sidx, self.eidx = pd.DataFrame(names).min()
        name = f"{label} ({unit}) ({self.sidx} {self.eidx})"
        index = pd.Index(index_ave, name=name)
        # Averaged value and standard deviation
        vals = [x.iloc[:, 0].to_numpy().reshape(-1, 1) for x in datas]
        vals = np.concatenate(vals, axis=1)
        name = f'{datas[0].columns[0]} (num={vals.shape[-1]})'
        data = {
            name: vals.mean(axis=1),
            f"{self.ERR_LB}{name}": vals.std(axis=1)
        }
        self.data = pd.DataFrame(data, index=index)

    def setData(self):
        """
        Set the data.
        """
        pass

    def saveData(self):
        """
        Save the data.
        """
        if self.data.empty:
            return
        self.data.to_csv(self.outfile, float_format=self.FLOAT_FMT)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')

    def fit(self):
        """
        Select the data and report average with std.

        :return int, int: the start and end index for the selected data
        """
        if self.data.empty:
            return
        sel = self.data.iloc[self.sidx:self.eidx]
        name, ave = sel.columns[0], sel.iloc[:, 0].mean()
        if sel.shape[1] == 1:
            err = sel.iloc[:, 0].std()
        else:
            err = sel.iloc[:, 1].mean()
        self.result = pd.Series({name: ave, f"{self.ERR_LB}{name}": err})
        self.result.index.name = sel.index.name
        label, unit, _ = self.parse(self.data.columns[0])
        stime, etime = sel.index[0], sel.index[-1]
        self.log(f'{label}: {ave:.4g} {symbols.PLUS_MIN} {err:.4g} {unit} '
                 f'{symbols.ELEMENT_OF} [{stime:.4f}, {etime:.4f}] ps')

    def parseIndex(self, name, sidx=0, eidx=None):
        """
        Parse the index name to get the label, unit, start index and end index.

        :param name: the column name
        :param sidx int: the start index
        :param eidx int: the end index
        return str, str, int, int: label, unit, start index, and end index.
        """
        label, unit, other = self.parse(name)
        if other is None:
            return label, unit, sidx, eidx
        splitted = list(map(int, other.split()))
        sidx = splitted[0]
        if len(splitted) >= 2:
            eidx = splitted[1]
        return label, unit, sidx, eidx

    @classmethod
    def parse(cls, name):
        """
        Parse the column label.

        :param name: the column name
        return str, str, str: the label, unit, and other information.
        """
        # 'Density (g/cm^3)
        (label, unit), other = cls.LABEL_RE.match(name).groups(), None
        match = cls.LABEL_RE.match(label)
        if match:
            # 'Density (g/cm^3) (num=4)' as data.columns[0]
            (label, unit), other = match.groups(), unit
        return label, unit, other

    def plot(self, marker_num=10):
        """
        Plot and save the data (interactively).

        :param marker_num: add markers when the number of points equals or is
            less than this value
        :type marker_num: int
        """
        if self.data.empty:
            return
        with plotutils.get_pyplot(inav=self.options.interactive,
                                  name=self.DESCR.upper()) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.13, 0.1, 0.8, 0.8])
            line_style = '--' if any([self.sidx, self.eidx]) else '-'
            if len(self.data) < marker_num:
                line_style += '*'
            ax.plot(self.data.index,
                    self.data.iloc[:, 0],
                    line_style,
                    label='average')
            if self.data.shape[-1] == 2 and self.data.iloc[:, 1].any():
                # Data has non-zero standard deviation column
                vals, errors = self.data.iloc[:, 0], self.data.iloc[:, 1]
                ax.fill_between(self.data.index,
                                vals - errors,
                                vals + errors,
                                color='y',
                                label='stdev',
                                alpha=0.3)
                ax.legend()
            if any([self.sidx, self.eidx]):
                gdata = self.data.iloc[self.sidx:self.eidx]
                ax.plot(gdata.index, gdata.iloc[:, 0], '.-g')
            label, unit, _ = self.parse(self.data.index.name)
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel(self.data.columns.values.tolist()[0])
            pathname = self.outfile[:-len(self.DATA_EXT)] + self.FIG_EXT
            fig.savefig(pathname)
            jobutils.add_outfile(pathname)
        self.log(f'{self.DESCR.capitalize()} figure saved as {pathname}')

    @classmethod
    def getName(cls, name=None, unit=None, label=None, names=None):
        """
        Get the property name.

        :param name str: build or search property name based on this name
        :param unit str: the unit of the property
        :param label str: get additional information from this label.
        :param names str: the available names to choose from.
        :return str: the property name
        :raise ValueError: if name cannot be determined.
        """
        if name is None:
            name = cls.NAME
        if unit is None:
            unit = cls.UNIT
        if names is not None:
            # Get name from names
            try:
                return next(x for x in names if re.match(name, x, re.I))
            except StopIteration:
                raise ValueError(f"{name} not in {names}.")
        # Build name from name, unit, and num
        name = f"{name} ({unit})" if unit else name
        num = cls.parse(label)[2] if label else None
        return name if num is None else f"{cls.PROP_NAME} ({num})"


class TrajBase(Base):
    """
    The base class for trajectory analyzers.
    """

    def __init__(self, traj=None, gids=None, **kwargs):
        """
        :param traj `traj.Traj`: traj frames
        :param gids list of int: global ids for the selected atom
        """
        super().__init__(**kwargs)
        self.traj = traj
        self.gids = gids
        if not self.traj:
            return
        self.sidx = self.traj.time.sidx


class XYZ(TrajBase):
    """
    xyz file writer
    """

    NAME = 'xyz'
    DESCR = NAME.upper()
    DATA_EXT = '.xyz'

    def run(self, wrapped=True, broken_bonds=False, glue=False):
        """
        Write the coordinates of the trajectory into XYZ format.

        :param wrapped bool: coordinates are wrapped into the PBC box.
        :param bond_across_pbc bool: allow bonds passing PBC boundaries.
        :param glue bool: circular mean to compact the molecules.

        NOTE: wrapped=False & glue=False is good for diffusion virtualization
        wrapped True & broken_bonds=False is good for box fully filled with molecules
        broken_bonds=False & glue=True is good for molecules droplets in vacuum
        Not all combination make physical senses.
        """
        with open(self.outfile, 'w') as self.out_fh:
            for frm in self.traj:
                if any([wrapped, glue]) and self.df_reader:
                    # wrapped and glue change the coordinates
                    frm = frm.copy(array=False)
                if wrapped:
                    frm.wrap(broken_bonds, dreader=self.df_reader)
                if glue:
                    frm.glue(dreader=self.df_reader)
                frm.write(self.out_fh, dreader=self.df_reader)
        self.log(f"{self.DESCR} coordinates are written into {self.outfile}")


class View(TrajBase):
    """
    Coordinates visualization
    """

    NAME = 'view'
    DESCR = 'trajectory visualization'
    DATA_EXT = '.html'

    def run(self):
        """
        Main method to run the visualization.
        """
        frm_vw = molview.FrameView(df_reader=self.df_reader)
        frm_vw.setData(self.traj[0])
        frm_vw.setElements()
        frm_vw.addTraces()
        frm_vw.setFrames(self.traj)
        frm_vw.updateLayout()
        frm_vw.show(outfile=self.outfile, inav=self.options.interactive)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')


class Density(TrajBase):
    """
    Structural density
    """

    NAME = 'density'
    DESCR = NAME
    UNIT = 'g/cm^3'
    LABEL = f'{NAME.capitalize()} ({UNIT})'

    def setData(self):
        """
        Set the time vs density data.
        """
        if self.data is not None:
            return
        mass = self.df_reader.molecular_weight / scipy.constants.Avogadro
        mass_scaled = mass / constants.ANG_TO_CM**3
        data = [mass_scaled / x.box.volume for x in self.traj]
        self.data = pd.DataFrame({self.LABEL: data}, index=self.traj.time)


class Clash(TrajBase):
    """
    Clashes between atoms
    """

    NAME = 'clash'
    DESCR = 'clash count'
    UNIT = 'count'
    LABEL = f'{NAME.capitalize()} ({UNIT})'

    def setData(self):
        """
        Set the time vs clash number.
        """
        if self.data is not None:
            return
        if not self.gids:
            self.log_warning("No atoms selected for clash counting.")
            self.data = pd.DataFrame({self.LABEL: []})
            return
        dcell = dist.Cell(gids=set(self.gids), struct=self.df_reader)
        data = []
        for frm in self.traj:
            dcell.setup(frm)
            data.append(len(dcell.getClashes()))
        self.data = pd.DataFrame(data={self.LABEL: data}, index=self.traj.time)


class RDF(Clash):
    """
    Radial distribution function
    """

    NAME = 'rdf'
    DESCR = 'radial distribution function'
    UNIT = 'r'
    LABEL = f'g ({UNIT})'
    INDEX_LB = f'r ({symbols.ANGSTROM})'
    PROP_NAME = f'{NAME} peak'
    POS_NAME = f"{PROP_NAME} position ({symbols.ANGSTROM})"

    def setData(self, res=0.02):
        """
        Set the radial distribution function.

        :param res float: the rdf minimum step size.
        """
        if self.data is not None:
            return
        if len(self.gids) < 2:
            self.log_warning("RDF requires least two atoms selected.")
            self.data = pd.DataFrame(data={self.LABEL: []})
            return

        span = np.array([x.box.span for x in self.traj.sel])
        vol = span.prod(axis=1)
        self.log(f'The volume fluctuates: [{vol.min():.2f} {vol.max():.2f}] '
                 f'{symbols.ANGSTROM}^3')

        dcell = dist.DistCell(span=span.min(), gids=self.gids)
        # the maximum distance for the RDF calculation
        max_dist = dcell.dist or span.min() * 0.5
        res = min(res, max_dist / 100)
        bins = round(max_dist / res)
        hist_range = [res / 2, res * bins + res / 2]
        rdf, num = np.zeros((bins)), len(self.gids)
        tenth, threshold, = len(self.traj.sel) / 10., 0
        for idx, frm in enumerate(self.traj.sel, start=1):
            self.log_debug(f"Analyzing frame {idx} for RDF..")
            dists = dcell.getDists(frm)
            hist, edge = np.histogram(dists, range=hist_range, bins=bins)
            mid = np.array([x for x in zip(edge[:-1], edge[1:])]).mean(axis=1)
            # 4pi*r^2*dr*rho from Radial distribution function - Wikipedia
            norm_factor = 4 * np.pi * mid**2 * res * num / frm.box.volume
            # Stands at every id but either (1->2) or (2->1) is computed
            rdf += (hist * 2 / num / norm_factor)
            if idx >= threshold:
                new_line = "" if idx == len(self.traj.sel) else ", [!n]"
                self.log(f"{int(idx / len(self.traj.sel) * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)
        rdf /= len(self.traj.sel)
        mid, rdf = np.concatenate(([0], mid)), np.concatenate(([0], rdf))
        index = pd.Index(data=mid, name=self.INDEX_LB)
        self.data = pd.DataFrame(data={self.LABEL: rdf}, index=index)

    def fit(self):
        """
        Smooth the rdf data and report peaks.
        """
        if self.data.empty:
            return
        raveled = np.ravel(self.data.iloc[:, 0])
        smoothed = scipy.signal.savgol_filter(raveled,
                                              window_length=31,
                                              polyorder=2)
        row = self.data.iloc[smoothed.argmax()]
        name = self.getName(label=self.data.columns[0])
        data = {name: row.values[0], self.POS_NAME: row.name}
        self.log('; '.join([f"{x}: {y}" for x, y in data.items()]))
        err = row.values[1] if len(row.values) > 1 else None
        self.result = pd.Series({
            name: row.values[0],
            f"{self.ERR_LB}{name}": err
        })

    @classmethod
    def getName(cls, name=None, **kwargs):
        """
        Get the property name and the error name.

        :param names str: the available names to choose from.
        :return str: the property name
        """
        return super().getName(name=cls.PROP_NAME, **kwargs)


class MSD(RDF):
    """
    Mean squared displacement & diffusion coefficient
    """

    NAME = 'msd'
    DESCR = 'mean squared displacement'
    UNIT = f'{symbols.ANGSTROM}^2'
    LABEL = f'{NAME.upper()} ({UNIT})'
    INDEX_LB = 'Tau (ps)'
    PROP_NAME = 'Diffusion Coefficient'
    ERR_LB = f'Standard Error of '

    def setData(self, spct=0.1, epct=0.2):
        """
        Set the mean squared displacement and diffusion coefficient.

        :param spct float: exclude the frames of this percentage at head
        :param epct float: exclude the frames of this percentage at tail
        """
        if self.data is not None:
            return
        if not self.gids:
            self.log_warning("No atoms selected for MSD.")
            self.data = pd.DataFrame({self.LABEL: []})
            return
        gids = list(self.gids)
        masses = self.df_reader.masses.mass[self.df_reader.atoms.type_id[gids]]
        msd, num = [0], len(self.traj.sel)
        for idx in range(1, num):
            disp = [
                x[gids, :] - y[gids, :]
                for x, y in zip(self.traj.sel[idx:], self.traj.sel[:-idx])
            ]
            data = np.array([np.linalg.norm(x, axis=1) for x in disp])
            sdata = np.square(data)
            msd.append(np.average(sdata.mean(axis=0), weights=masses))
        ps_time = self.traj.time[:num]
        self.sidx = math.floor(num * spct)
        self.eidx = math.ceil(num * (1 - epct))
        name = f"{self.INDEX_LB} ({self.sidx} {self.eidx})"
        tau_idx = pd.Index(data=ps_time - ps_time[0], name=name)
        self.data = pd.DataFrame({self.LABEL: msd}, index=tau_idx)

    def fit(self):
        """
        Select and fit the mean squared displacement to calculate the diffusion
        coefficient.
        """
        if self.data.empty:
            return
        sel = self.data.iloc[self.sidx:self.eidx]
        # Standard error of the slope, under the assumption of residual normality
        xvals = sel.index * scipy.constants.pico
        yvals = sel.iloc[:, 0] * constants.ANG_TO_CM**2
        slope, intcp, rval, pval, stderr = scipy.stats.linregress(xvals, yvals)
        # MSD=2nDt https://en.wikipedia.org/wiki/Mean_squared_displacement
        value, std_err = slope / 6, stderr / 6
        self.log(f'{value:.4g} {symbols.PLUS_MIN} {std_err:.4g} cm^2/s'
                 f' (R-squared: {rval**2:.4f}) linear fit of'
                 f' [{sel.index[0]:.4f} {sel.index[-1]:.4f}] ps')
        propname = self.getName()
        errname = f"{self.ERR_LB}{propname}"
        self.result = pd.Series({propname: value, errname: std_err})
        self.result.index.name = sel.index.name

    @classmethod
    def getName(cls, unit=None, **kwargs):
        """
        Get the property name and the error name.

        :param names str: the available names to choose from.
        :param unit str: the unit of the property
        :return str: the property name
        """
        return super().getName(unit='cm^2/s', **kwargs)


ALL_FRM = [x.NAME for x in [Density, Clash, View, XYZ]]
DATA_RQD = [x.NAME for x in [Density, MSD, RDF, Clash]]
TRAJ = {x.NAME: x for x in [Density, RDF, MSD, Clash, View, XYZ]}


class Thermo(Base):

    NAME = 'toteng'
    DESCR = 'Thermodynamic information'
    FLOAT_FMT = '%.8g'

    def __init__(self, thermo=None, **kwargs):
        """
        :param thermo: the thermodynamic data
        :type thermo: 'pandas.core.frame.DataFrame'
        """
        super().__init__(**kwargs)
        self.thermo = thermo
        if self.thermo is None:
            return
        self.sidx = int(self.LABEL_RE.match(self.thermo.index.name).group(2))

    def setData(self):
        """
        Select data by the thermo task name.
        """
        if self.data is not None:
            return
        column_re = re.compile(f"{self.NAME} +\((.*)\)", re.IGNORECASE)
        column = [x for x in self.thermo.columns if column_re.match(x)][0]
        self.data = self.thermo[column].to_frame()

    @classmethod
    @property
    @functools.cache
    def Analyzers(cls):
        """
        The list of thermo analyzer classes.

        :return subclass of 'Thermo': the therma analyzer classes
        """
        return [
            type(x, (cls, ), dict(NAME=x, DESCR=x.capitalize())) for x in
            ['temp', 'e_pair', 'e_mol', symbols.TOTENG, 'press', 'volume']
        ]


THERMO = {x.NAME: x for x in Thermo.Analyzers}
ANLZ = {**TRAJ, **THERMO}


class Agg(logutils.Base):

    DATA_EXT = '.csv'
    FIG_EXT = '.png'
    TO_SKIP = set([XYZ.NAME, View.NAME])

    def __init__(self, task, groups=None, options=None, logger=None):
        """
        :param task str: the task name to analyze
        :param groups list of tuples: each tuple contains parameters
            (pandas.Series), grouped jobs (signac.job.Job)
        :type jobs: list of (pandas.Series, 'signac.job.Job') tuples
        :param options 'argparse.Namespace': the options from command line
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.task = task
        self.groups = groups
        self.options = options
        self.yvals = None
        self.ydevs = None
        self.xvals = None
        self.result = pd.DataFrame()
        self.outfile = f"{self.options.jobname}_{self.task}{self.DATA_EXT}"
        self.Anlz = None if self.task in self.TO_SKIP else ANLZ[self.task]

    def run(self):
        """
        Main method to aggregate the analyzer output files over all parameters.
        """
        if self.Anlz is None:
            return
        self.setResult()
        self.save()
        self.setVals()
        self.fit()
        self.plot()

    def setResult(self):
        """
        Set the result for the given task over grouped jobs.
        """
        self.log(f"Aggregation Task: {self.task}")
        shared = vars(self.options)
        if self.options.interactive:
            shared = shared.copy()
            shared['interactive'] = len(self.jobs) > 1
        for parm, jobs in self.groups:
            if not parm.empty:
                pstr = parm.to_csv(lineterminator=' ', sep='=', header=False)
                self.log(f"Aggregation Parameters (num={len(jobs)}): {pstr}")
            options = types.SimpleNamespace(**shared,
                                            id=parm.index.name,
                                            jobs=jobs)
            anlz = self.Anlz(options=options, logger=self.logger)
            anlz.run()
            if anlz.result is None:
                continue
            result = [anlz.result] if parm.empty else [parm, anlz.result]
            result = pd.concat(result).to_frame().transpose()
            self.result = pd.concat([self.result, result])

    def save(self):
        """
        Save the results to a file.
        """
        if self.result.empty:
            return
        self.result.to_csv(self.outfile, index=False)
        task = self.task.capitalize()
        self.log(f"{task} of all parameters saved to {self.outfile}")
        jobutils.add_outfile(self.outfile)

    def setVals(self):
        """
        Set the x, y, and y standard deviation from the results.
        """
        name = self.Anlz.getName(names=self.result.columns)
        self.Anlz.getName(names=self.result.columns)
        self.yvals = self.result[name]
        ERR_LB_name = f"{self.Anlz.ERR_LB}{name}"
        self.ydevs = self.result[ERR_LB_name]
        x_lbs = list(set(self.result.columns).difference([name, ERR_LB_name]))
        self.xvals = self.result[x_lbs]
        rename = {
            x: ' '.join([y.capitalize() for y in x.split('_')])
            for x in self.xvals.columns
        }
        self.xvals = self.xvals.rename(columns=rename)

    def fit(self):
        """
        Fit the data and report.
        """
        if self.xvals.empty or self.xvals.size == 1:
            return
        index = self.yvals.argmin()
        self.log(f"The minimum {self.yvals.name} of {self.yvals.iloc[index]} "
                 f"is found with the {self.xvals.columns[0].replace('_',' ')} "
                 f"being {self.xvals.iloc[index, 0]}")

    def plot(self, xtick_num=12):
        """
        Plot the results.

        :param xtick_num int: the maximum number of xticks to show
        """
        if self.xvals.empty:
            return
        with plotutils.get_pyplot(inav=self.options.interactive,
                                  name=self.task.upper()) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.13, 0.1, 0.8, 0.8])
            ax.plot(self.xvals.iloc[:, 0], self.yvals, '--*', label='average')
            if not self.ydevs.isnull().any():
                # Data has non-zero standard deviation column

                ax.fill_between(self.xvals.values,
                                self.yvals - self.ydevs,
                                self.yvals + self.ydevs,
                                color='y',
                                label='stdev',
                                alpha=0.3)
                ax.legend()
            ax.set_xlabel(self.xvals.columns[0])
            if self.xvals.iloc[:, 0].size > xtick_num:
                intvl = round(self.xvals.iloc[:, 0].size / xtick_num)
                ax.set_xticks(self.xvals.iloc[:, 0].values[::intvl])
            ax.set_ylabel(self.yvals.name)
            pathname = self.outfile[:-len(self.DATA_EXT)] + self.FIG_EXT
            fig.savefig(pathname)
            jobutils.add_outfile(pathname)
        self.log(f'{self.task.upper()} figure saved as {pathname}')
