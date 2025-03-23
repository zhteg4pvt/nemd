# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Trajectory frame module to read, copy, wrap, blue, and writes coordinates as
well as computing pair distances.
"""
import types
import warnings

import numpy as np
import pandas as pd

from nemd import pbc
from nemd import symbols


class Base(np.ndarray):
    """
    Coordinates and box container.
    """

    def __new__(cls, data=None, shape=(0, ), **kwargs):
        """
        :param data 'np.ndarray': the xyz coordinates.
        :param shape tuple: the shape of the xyz coordinates.
        :return (sub-)class of the base: the base object of coordinates and box
        """
        return super().__new__(cls, shape=shape) if data is None \
            else np.asarray(data).view(cls)

    def __init__(self, data=None, box=None, **kwargs):
        """
        :param data nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param box `Box`: the pbc box
        """
        self.box = getattr(data, 'box', None) if box is None else box

    def pairDists(self, grp=None, grps=None):
        """
        Get the distances between atom pairs.

        :param grp list: atom global ids
        :param grps list of list: each sublist contains atom global ids to
            compute distances with each atom in grp.
        return numpy.ndarray: pair distances.
        """
        grp = list(range(self.shape[0])) if grp is None else sorted(grp)
        grps = [grp[i:] for i in range(1, len(grp))] if grps is None else grps
        vecs = [self[x, :] - self[y, :] for x, y in zip(grps, grp)]
        if not vecs:
            return np.array([])
        return np.concatenate([self.box.norm(x) for x in vecs])


class Frame(Base):
    """
    Coordinates manipulation.
    """

    def __init__(self, data=None, step=None, **kwargs):
        """
        :param data nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param step int: the number of simulation step that this frame is at
        """
        super().__init__(data=data, **kwargs)
        self.step = getattr(data, 'step', None) if step is None else step

    @classmethod
    def read(cls, fh, start=0):
        """
        Read a custom dumpy file with id, xu, yu, zu. Full coordinate
        information is available with step number >= start.

        :param fh '_io.TextIOWrapper': the file handle to read the frame from.
        :param start int: frames with step number < this value are fully read.
        :return 'SimpleNamespace' or 'Frame': 'Frame' has step, box and
            coordinates information; 'SimpleNamespace' only has step info.
        :raise EOFError: the frame block is incomplete.
        """
        header = [fh.readline() for _ in range(9)]
        if not header[-1]:
            raise EOFError('Empty Header')
        step, atom_num = int(header[1].rstrip()), int(header[3].rstrip())
        if step < start:
            with warnings.catch_warnings(record=True):
                np.loadtxt(fh, skiprows=atom_num, max_rows=0)
                return types.SimpleNamespace(step=step)
        data = np.loadtxt(fh, max_rows=atom_num, ndmin=2)
        if data.shape[0] != atom_num:
            raise EOFError('Incomplete Atom Coordinates')
        data = data[data[:, 0].argsort()]  # Sort the xyz by atom ids
        if int(data[-1, 0]) == atom_num:
            ndata = data[:, 1:]
        else:
            ndata = np.full([atom_num, 3], np.nan)
            ndata[data[:, 0].astype(int) - 1, :] = data[:, 1:]
        return cls(ndata, box=pbc.Box.fromLines(header[5:8]), step=step)

    def copy(self, array=True):
        """
        Copy the numpy array content if array else the whole object.
        The default copy behavior follows numpy.ndarray.copy() as pandas may
        use it. The array=False option is added to copy the object.

        :return 'np.ndarray' or 'Frame': the copied.
        """
        if array:
            return np.array(self)
        return Frame(data=self.copy(), box=self.box, step=self.step)

    def wrap(self, broken_bonds=False, molecules=None):
        """
        Wrap atoms or molecule centers into the PBC first image.

        :param broken_bonds bool: allow bonds broken by PBC boundaries.
        :param molecules 'dict': molecule ids -> global atom ids
        """
        if broken_bonds:
            self[:] = self % self.box.span
            return
        if molecules is None:
            return
        # The unwrapped xyz can directly perform molecule center operation
        for gids in molecules.values():
            center = self[gids, :].mean(axis=0)
            self[gids, :] += (center % self.box.span) - center

    def glue(self, molecules=None):
        """
        Circular mean to compact the molecules. (molecules droplets in vacuum)

        FIXME: support droplets or clustering in solution

        :param molecules 'dict': molecule ids -> global atom ids
        """
        if molecules is None:
            return
        centers = np.array([self[x].mean(axis=0) for x in molecules.values()])
        theta = centers / self.box.span * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        mtheta = np.arctan2(sin_theta.mean(axis=0), cos_theta.mean(axis=0))
        mcenters = mtheta * self.box.span / 2 / np.pi
        shifts = ((mcenters - centers) / self.box.span).round() * self.box.span
        for mol_id, gids in molecules.items():
            self[gids] += shifts[mol_id]

    def write(self, fh, dreader=None, visible=None, points=None):
        """
        Write XYZ to a file.

        :param fh '_io.TextIOWrapper': file handdle to write out xyz.
        :param dreader 'oplsua.Reader': datafile reader for element info
        :param visible list: visible atom gids.
        :param points list: additional point to visualize.
        """
        data = self[visible] if visible else self
        index = np.argwhere(~np.isnan(data[:, 0])).flatten()
        data = pd.DataFrame(data, columns=symbols.XYZU, index=index)
        if dreader is None:
            data.index = [symbols.UNKNOWN] * data.shape[0]
        else:
            type_ids = dreader.atoms.type_id.loc[data.index]
            data.index = dreader.masses.element[type_ids]
        if points:
            points = np.array(points)
            points = pd.DataFrame(points,
                                  index=[symbols.UNKNOWN] * points.shape[0],
                                  columns=symbols.XYZU)
            data = pd.concat((data, points), axis=0)
        fh.write(f'{data.shape[0]}\n')
        data.to_csv(fh,
                    mode='a',
                    index=True,
                    sep=' ',
                    header=True,
                    quotechar=' ',
                    float_format='%.4f')
