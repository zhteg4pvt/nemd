# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Lammps trajectory and log analyzers.
"""
import math
import os
import re

import numpy as np
import pandas as pd
import scipy

from nemd import constants
from nemd import dist
from nemd import frame
from nemd import jobutils
from nemd import logutils
from nemd import molview
from nemd import numbautils
from nemd import plotutils
from nemd import symbols


class Base(logutils.Base):
    """
    Base class for job and merger.
    """
    DATA_EXT = '.csv'
    FIG_EXT = '.svg'
    ST_DEV = 'St Dev'
    ST_ERR = 'St Err'
    FLOAT_FMT = '%.4g'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.fig = None
        self.fitted = None

    def run(self):
        """
        Main method.
        """
        self.merge()
        if self.data is None:
            return
        self.save()
        self.fit()
        self.plot()

    def merge(self):
        """
        Merge data.
        """
        pass

    def save(self):
        """
        Save the data.
        """
        self.data.to_csv(self.outfile, float_format=self.FLOAT_FMT)
        self.log(f'{self.full.capitalize()} data written into {self.outfile}')
        jobutils.Job.reg(self.outfile)

    @classmethod
    @property
    def name(cls):
        """
        The name of the analyzer.

        :return str: analyzer name
        """
        return cls.__name__.lower()

    @property
    def full(self):
        """
        The full name of the analyzer.

        :return str: analyzer full name.
        """
        return self.name

    @property
    def outfile(self):
        """
        The outfile to save data.

        :return str: the outfile.
        """
        return f"{self.options.JOBNAME}_{self.name}{self.DATA_EXT}"

    def fit(self):
        """
        Fit the data and report.
        """
        if self.data.index.name is None:
            return
        row = self.data.iloc[self.data.iloc[:, 0].argmin()]
        label, unit, _ = self.parse(self.data.columns[0])
        self.log(f"The minimum {label} of {row.iloc[0]:.4g} {unit} is found at"
                 f" {row.name} {self.parse(self.data.index.name)[1]}.")

    def plot(self, marker=None, selected=None, rotation=0):
        """
        Plot and save the data: index (time|param|default), data (original|ave),
        standard deviation (optional).

        :param marker str: the marker symbol
        :param selected pd.DataFrame: the selected data
        :param rotation float: The rotation angle of the xtick labels in degrees
        """
        with plotutils.pyplot(inav=self.options.INTERAC,
                              name=self.name) as plt:
            self.fig = plt.figure(figsize=(6, 4.5))
            ax = self.fig.add_subplot(1, 1, 1)
            if marker is None and len(self.data) < 20:
                marker = '*'
            ax.plot(self.data.index,
                    self.data.iloc[:, 0],
                    '-' if selected is None else '--',
                    marker=marker,
                    label='average' if self.with_err else 'data')
            if selected is not None:
                ax.plot(selected.index,
                        selected.iloc[:, 0],
                        '.-g',
                        label='selected')
            if self.with_err:
                vals = self.data.iloc[:, 0].astype(float)
                errs = self.data.iloc[:, 1].astype(float)
                ax.fill_between(self.data.index,
                                vals - errs,
                                vals + errs,
                                color='y',
                                label='stdev',
                                alpha=0.3)
            if self.fitted is not None:
                ax.plot(self.fitted[:, 0],
                        self.fitted[:, 1],
                        '--r',
                        label='fit')
            ax.legend()
            if self.data.index.name is not None:
                name, unit, _ = self.parse(self.data.index.name)
                ax.set_xlabel(f"{name} ({unit})" if unit else name)
            ax.set_ylabel(self.data.columns.values.tolist()[0])
            self.fig.autofmt_xdate(rotation=rotation)
            outfile = self.outfile[:-len(self.DATA_EXT)] + self.FIG_EXT
            self.fig.savefig(outfile)
            self.log(f'{self.name.capitalize()} figure saved as {outfile}')
            jobutils.Job.reg(outfile)

    @property
    def with_err(self):
        """
        Whether the data come with a non-null error column.

        :return bool: True if the data come with an error column.
        """
        return self.data.shape[1] == 2 and not self.data.iloc[:, 1].isnull(
        ).all()


class Job(Base):
    """
    The analyzer base class.
    """
    UNIT = 'a.u.'
    PROP = None

    def __init__(self, group=None, rdr=None, **kwargs):
        """
        :param group `task.Group`: job group.
        :param rdr `oplsua.Reader`: data file reader
        """
        super().__init__(**kwargs)
        self.group = group
        self.rdr = rdr
        self.sidx = 0
        self.eidx = None
        self._result = None

    def run(self):
        """
        See parent.
        """
        self.set()
        super().run()

    def set(self):
        """
        Set the data from new calculation.
        """
        if self.group is not None:
            return
        self.calculate()
        if self.data is not None:
            return
        self.warning(f"No data calculated for {self.name}")

    def calculate(self):
        """
        Calculate.
        """
        pass

    def merge(self):
        """
        Merge data from previous calculations.
        """
        if self.group is None:
            return
        if not self.group.parm.empty:
            self.log(self.group.to_str())
        name = f"{self.options.NAME}_{self.name}{self.DATA_EXT}"
        files = [x.fn(name) for x in self.group.jobs]
        files = [x for x in files if os.path.exists(x)]
        if not files:
            return

        datas = [pd.read_csv(x, index_col=0) for x in files]
        # 'Time (ps)': None; 'Time (ps) (0)': '0'; 'Time (ps) (0, 1)': '0, 1'
        names = [self.parseIndex(x.index.name) for x in datas]
        label, unit, sidx, eidx = names[0]
        if len(datas) == 1:
            # One single run
            self.data = datas[0]
            self.sidx, self.eidx = sidx, eidx
            return

        # Backwardly truncate the runs with different randomized seeds
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
        data = np.array((vals.mean(axis=1), vals.std(axis=1))).transpose()
        name = f'{datas[0].columns[0]} (num={vals.shape[-1]})'
        columns = [name, f"{self.ST_DEV} of {name}"]
        self.data = pd.DataFrame(data, columns=columns, index=index)

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
        split = list(map(int, other.split()))
        return label, unit, split[0], split[1] if len(split) >= 2 else eidx

    @property
    def label(self):
        """
        The label name with unit.

        :return str: name with unit
        """
        return f'{self.__class__.__name__} ({self.UNIT})'

    @property
    def outfile(self):
        """
        See parent.
        """
        if self.group is None:
            return super().outfile
        root, ext = os.path.splitext(super().outfile)
        root = os.path.join(jobutils.WORKSPACE, f"{root}")
        return f"{root}_{self.group.parm.index.name}{ext}"

    def fit(self):
        """
        Select the data and report average with std.
        """
        sel = self.data.iloc[self.sidx:self.eidx]
        name, ave = sel.columns[0], sel.iloc[:, 0].mean()
        self.fitted = np.array((sel.index, np.full(sel.index.shape, ave))).T
        err = sel.iloc[:, 1].mean() if self.data.shape[1] == 2 \
            else sel.iloc[:, 0].std()
        self._result = pd.Series({name: ave, f"{self.ST_DEV} of {name}": err})
        label, unit, _ = self.parse(self.data.columns[0])
        stime, etime = sel.index[0], sel.index[-1]
        self.log(f'{label}: {ave:.4g} {symbols.PLUS_MIN} {err:.4g} {unit} '
                 f'{symbols.ELEMENT_OF} [{stime:.4f}, {etime:.4f}] ps')

    @classmethod
    def getName(cls, name=None, unit=None, label=None, names=None, err=False):
        """
        Get the property name.

        :param name str: build or search property name based on this name
        :param unit str: the unit of the property
        :param label str: get additional information from this label.
        :param names str: the available names to choose from.
        :param err bool: search the error name if True
        :return str: the property name
        """
        if name is None:
            name = cls.PROP if cls.PROP else cls.name
        if unit is None:
            unit = cls.UNIT

        if names is None:
            # Build name from name, unit, and num
            name = f"{name} ({unit})"
            num = cls.parse(label)[2] if label else None
            return name if num is None else f"{name} ({num})"

        rex = f'St (Dev|Err) of {name}' if err else name
        return next((x for x in names if re.match(rex, x, re.I)), None)

    def plot(self, selected=None, **kwargs):
        """
        See parent.
        """
        if any([self.sidx, self.eidx]):
            selected = self.data.iloc[self.sidx:self.eidx]
        super().plot(selected=selected, **kwargs)

    @property
    def result(self):
        """
        Return the result (with parameter).

        :return pd.DataFrame: the result.
        """
        result = self._result if self.group is None or self.group.parm.empty \
            else pd.concat([self.group.parm, self._result])
        result.index.name = None
        return result.to_frame().transpose()


class Density(Job):
    """
    Structural density.
    """
    UNIT = 'g/cm^3'

    def __init__(self, trj=None, gids=None, **kwargs):
        """
        :param traj `traj.Traj`: traj frames
        :param gids np.ndarray: global ids for the selected atoms
        """
        super().__init__(**kwargs)
        self.trj = trj
        self.gids = gids
        if self.trj is None:
            return
        self.sidx = self.trj.time.sidx
        if self.gids is None:
            frm = next(x for x in self.trj if isinstance(x, frame.Frame))
            self.gids = np.array(list(range(frm.shape[0])))

    def calculate(self):
        """
        Calculate the density data.
        """
        vol = (x.box.volume for x in self.trj)
        data = np.fromiter(vol, dtype=float) * constants.ANG_TO_CM**3
        data = self.rdr.molecular_weight / scipy.constants.Avogadro / data
        self.data = pd.DataFrame({self.label: data}, index=self.trj.time)


class XYZ(Density):
    """
    xyz file writer.
    """
    DATA_EXT = '.xyz'

    def run(self, center=False, wrapped=True, broken_bonds=False):
        """
        Write the coordinates of the trajectory into XYZ format.

        :param center bool: align circular-mean center with the box center
        :param wrapped bool: wrap atoms or molecules into the first PBC image
        :param broken_bonds bool: allow bonds broken by PBC boundaries.

        glue=True & wrapped=True: one droplet in vacuum
        glue=False & wrapped=False: diffusion visualization
        wrapped=True & broken_bonds=False: condensed-phase molecules
        broken_bonds=True: infinite structures
        """
        with open(self.outfile, 'w') as self.out_fh:
            for frm in self.trj:
                if any([center, wrapped]) and self.rdr:
                    # wrapped and glue change the coordinates
                    frm = frm.getCopy()
                if center:
                    frm.center()
                if wrapped:
                    frm.wrap(broken_bonds, dreader=self.rdr)
                frm.write(self.out_fh, dreader=self.rdr)
        self.log(f"XYZ coordinates are written into {self.outfile}")


class View(Density):
    """
    Coordinates visualization.
    """
    DATA_EXT = '.html'

    def run(self):
        """
        Main method to run the visualization.
        """
        fig = molview.Figure(self.trj, rdr=self.rdr)
        fig.write_html(self.outfile)
        self.log(f'Trajectory visualization are written into {self.outfile}')


class Clash(Density):
    """
    Clashes between atoms.
    """
    UNIT = 'count'

    def __init__(self, *args, cut=None, srch=None, **kwargs):
        """
        :param cut float: the cut-off for neighbor searching
        :param srch bool: use distance cell to search neighbors if True
        """
        super().__init__(*args, **kwargs)
        self.cut = cut
        self.srch = srch
        self.grp = self.gids
        self.grps = None
        if self.cut is None:
            self.cut = dist.Radius(struct=self.rdr).max()
        if self.srch is None and self.trj is not None:
            self.srch = any(x.large(self.cut) for x in self.trj.sel)
        if self.srch and self.grp is not None:
            # The smallest gid included in grps (direct or from distance cell)
            self.gids.sort()
            self.grp = self.gids[1:]
            self.grps = [self.gids[:i] for i in range(1, len(self.gids))]

    def calculate(self):
        """
        Calculate the clash number.
        """
        if len(self.gids) < 2:
            self.warning("Clash requires least two atoms selected.")
            return
        data = []
        with self.logger.progress(len(self.trj)) as prog:
            for idx, frm in enumerate(self.trj, start=1):
                dfrm = dist.Frame(frm,
                                  gids=self.gids,
                                  struct=self.rdr,
                                  srch=self.srch)
                clashes = dfrm.getClashes(self.grp, grps=self.grps)
                data.append(len(clashes))
                prog(idx)
                del dfrm.cell
        self.data = pd.DataFrame(data={self.label: data}, index=self.trj.time)


class RDF(Clash):
    """
    Radial distribution function.
    """
    UNIT = 'r'
    PROP = f'peak'

    def __init__(self, *args, cut=symbols.DEFAULT_CUT, **kwargs):
        """
        See parent.
        """
        super().__init__(*args, cut=cut, **kwargs)

    def calculate(self, res=0.02):
        """
        Calculate radial distribution function.

        :param res float: the rdf minimum step size.
        """
        if len(self.gids) < 2:
            self.warning("RDF requires least two atoms selected.")
            return

        span = np.array([x.box.span for x in self.trj.sel])
        vol = span.prod(axis=1)
        self.log(f'The volume fluctuates: [{vol.min():.2f} {vol.max():.2f}] '
                 f'{symbols.ANGSTROM}^3')

        # edge / 2 sets the maximum distance in that direction
        max_dist = min(span.min() / 2, self.cut if self.srch else np.inf)
        res = min(res, max_dist / 100)
        bins = round(max_dist / res)
        hist_range = [res / 2, res * bins + res / 2]
        rdf, num = np.zeros((bins)), len(self.gids)
        sel_num = len(self.trj.sel)
        with self.logger.progress(len(self.trj.sel)) as prog:
            for idx, frm in enumerate(self.trj.sel, start=1):
                dfrm = dist.Frame(frm,
                                  gids=self.gids,
                                  cut=self.cut,
                                  srch=self.srch)
                dists = dfrm.getDists(grp=self.grp, grps=self.grps)
                hist, edge = np.histogram(dists, range=hist_range, bins=bins)
                mid = np.mean([x for x in zip(edge[:-1], edge[1:])], axis=1)
                # 4pi*r^2*dr*rho from Radial distribution function - Wikipedia
                norm_factor = 4 * np.pi * mid**2 * res * num / frm.box.volume
                # Stands at every id but either (1->2) or (2->1) is computed
                rdf += (hist * 2 / num / norm_factor)
                prog(idx)
                # The memory usage of CellNumba dfrm.cell is not released
                # by dfrm reassignment or deletion. (8GB memory stest 12)
                del dfrm.cell
        rdf /= sel_num
        mid, rdf = np.concatenate(([0], mid)), np.concatenate(([0], rdf))
        index = pd.Index(data=mid, name=f'r ({symbols.ANGSTROM})')
        self.data = pd.DataFrame(data={self.label: rdf}, index=index)

    @property
    def label(self):
        """
        See parent.
        """
        return f'g ({self.UNIT})'

    def fit(self, window_length=31):
        """
        Smooth the rdf data and report peaks.

        :param window_length int: the window length when smoothing.
        """
        raveled = np.ravel(self.data.iloc[:, 0])
        smoothed = scipy.signal.savgol_filter(raveled,
                                              window_length=window_length,
                                              polyorder=2)
        idx = smoothed.argmax()
        idxs = [y for x in range(window_length) for y in [idx - x, idx + x]]
        idx = next((x for x in idxs[1:] if self.data.iloc[x, 0]), idx)
        row = self.data.iloc[idx]
        peak = row.values[0]
        err = row.values[1] if len(row.values) > 1 else np.nan
        self.log(f'RDF peak {peak:.4g} {symbols.PLUS_MIN} {err:.4g} found '
                 f'at {row.name} {symbols.ANGSTROM}')
        name = self.getName(label=self.data.columns[0])
        self._result = pd.Series({name: peak, f"{self.ST_DEV} of {name}": err})
        self.fitted = np.array([self.data.index, smoothed]).T

    @property
    def full(self):
        """
        See parent.
        """
        return 'radial distribution function'


class MSD(RDF):
    """
    Mean squared displacement & diffusion coefficient.
    """
    UNIT = f'{symbols.ANGSTROM}^2'
    PROP = 'Diffusion Coefficient'

    def calculate(self, spct=0.1, epct=0.2):
        """
        Calculate the mean squared displacement and diffusion coefficient.

        :param spct float: exclude the frames of this percentage at head
        :param epct float: exclude the frames of this percentage at tail
        """
        if self.gids is None or not len(self.gids):
            self.warning("MSD requires least one atom selected.")
            return

        if len(self.trj.sel) == 1:
            self.warning("Only one trajectory frame selected for MSD.")
            return

        wts = self.rdr.weights if self.rdr else np.ones(len(self.gids))
        msd = numbautils.msd(np.array(self.trj.sel), self.gids, wts)
        num = len(msd)
        ps_time = self.trj.time[-num:]
        self.sidx = math.floor(num * spct)
        self.eidx = math.ceil(num * (1 - epct))
        name = f"Tau (ps) ({self.sidx} {self.eidx})"
        tau_idx = pd.Index(data=ps_time - ps_time[0], name=name)
        self.data = pd.DataFrame({self.label: msd}, index=tau_idx)

    @property
    def label(self):
        """
        See parent.
        """
        return f'{self.name.upper()} ({self.UNIT})'

    def fit(self):
        """
        Select and fit the mean squared displacement to calculate the diffusion
        coefficient.
        """
        sel = self.data.iloc[self.sidx:self.eidx]
        # Standard error of the slope, under the assumption of residual normality
        xvals = sel.index * scipy.constants.pico
        yvals = sel.iloc[:, 0] * constants.ANG_TO_CM**2
        slope, intcp, rval, pval, stderr = scipy.stats.linregress(xvals, yvals)
        # MSD=2nDt https://en.wikipedia.org/wiki/Mean_squared_displacement
        value, std_err = slope / 6, stderr / 6
        self.log(f'{self.PROP} {value:.4g} {symbols.PLUS_MIN} {std_err:.4g} '
                 f'cm^2/s calculated by fitting {self.name.upper()} '
                 f'{symbols.ELEMENT_OF} [{sel.index[0]:.3g} '
                 f'{sel.index[-1]:.3g}] ps. (R-squared: {rval**2:.3g})')
        name = self.getName(unit='cm^2/s', label=self.data.columns[0])
        index = [name, f"{self.ST_ERR} of {name}"]
        index = pd.Index(index, name=sel.index.name)
        self._result = pd.Series([value, std_err], index=index)
        fitted = (slope * xvals + intcp) / constants.ANG_TO_CM**2
        self.fitted = np.array([sel.index, fitted]).T

    @property
    def full(self):
        """
        See parent.
        """
        return 'mean squared displacement'


ALL_FRM = [Density, Clash, View, XYZ]
DATA_RQD = [Density]
TRAJ = [Density, RDF, MSD, Clash, View, XYZ]


class TotEng(Job):
    """
    Thermodynamic analyzer.
    """
    FLOAT_FMT = '%.8g'

    def __init__(self, thermo=None, **kwargs):
        """
        :param thermo 'pandas.core.frame.DataFrame': the thermodynamic data
        """
        super().__init__(**kwargs)
        self.thermo = thermo
        if self.thermo is None:
            return
        self.sidx = self.parseIndex(self.thermo.index.name)[2]

    def calculate(self):
        """
        Calculate thermo.
        """
        column_re = re.compile(rf"{self.name} +\((.*)\)", re.IGNORECASE)
        column = [x for x in self.thermo.columns if column_re.match(x)][0]
        self.data = self.thermo[column].to_frame()

    @property
    def full(self):
        """
        See parent.
        """
        return 'thermodynamic information'


THERMO = [
    type(x, (TotEng, ), {})
    for x in ['temp', 'e_pair', 'e_mol', 'toteng', 'press', 'volume']
]


class Merger(Base):
    """
    The analyzer aggregator.
    """

    def __init__(self, Anlz, groups=None, **kwargs):
        """
        :param task str: the task name to analyze
        :param groups list of tuples: each tuple contains parameters
            (pandas.Series), grouped jobs (signac.job.Job)
        :type jobs: list of (pandas.Series, 'signac.job.Job') tuples
        """
        super().__init__(**kwargs)
        self.Anlz = Anlz
        self.groups = groups

    def merge(self):
        """
        Merge data over parameter sets.
        """
        self.log(f"Task: {self.Anlz.name}")
        for group in self.groups:
            anlz = self.Anlz(options=self.options,
                             logger=self.logger,
                             group=group)
            anlz.run()
            if anlz._result is None:
                continue
            self.data = pd.concat([self.data, anlz.result])

        if self.data is None:
            self.warning(f"No data aggregated for {self.name}")
            return

        val = self.Anlz.getName(names=self.data.columns)
        err = self.Anlz.getName(names=self.data.columns, err=True)
        parm = self.data[list(set(self.data.columns).difference([val, err]))]
        if parm.empty:
            return
        self.data.set_index(parm.columns[0], inplace=True)
        words = [y.capitalize() for y in self.data.index.name.split('_')]
        self.data.index.name = ' '.join(words)

    @property
    def name(self):
        """
        See parent.
        """
        return self.Anlz.name

    @classmethod
    def main(cls, task, **kwargs):
        """
        Maim method to initialize and run.

        :param task str: the task name.
        """
        Anlz = cls.getAnlz(task)
        if Anlz is None:
            return
        anlz = cls(Anlz, **kwargs)
        anlz.run()

    @staticmethod
    def getAnlz(task, ANLZS=(Density, RDF, MSD, Clash, *THERMO)):
        """
        Get analyzer class.

        :param task: selected analyzer name.
        :param ANLZS tuple: analyzers that support aggregation.
        :return `type`: analyzer class.
        """
        return next((x for x in ANLZS if x.name == task), None)
