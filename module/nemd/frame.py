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

    def __new__(cls, frm, dtype=np.float32, **kwargs):
        """
        :param data 'np.ndarray': the xyz coordinates.
        :param dtype type: the data type.
        :return (sub-)class of the base: the base object of coordinates and box.
        """
        return np.asarray(frm, dtype=dtype).view(cls)

    def __init__(self, frm, box=None, **kwargs):
        """
        :param data nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param box `Box`: the pbc box
        """
        self.box = getattr(frm, 'box', None) if box is None else box

    def large(self, cut):
        """
        Whether the cell is considered as large with respect to the cut.

        :param cut float: the cut-off
        :return bool: whether to use the distance cell.
        """
        return np.prod(self.box.span) / np.power(cut, 3) >= 1000


class Frame(Base):
    """
    Coordinates manipulation.
    """

    def __init__(self, frm, step=None, **kwargs):
        """
        :param frm nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param step int: the number of simulation step that this frame is at
        """
        super().__init__(frm, **kwargs)
        self.step = getattr(frm, 'step', None) if step is None else step

    @classmethod
    def read(cls, fh, start=0):
        """
        Read a custom dumpy file with id, xu, yu, zu. Full coordinates
        information are available when step number >= start.

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
        # FIXME: triclinic support
        return cls(ndata, box=pbc.Box.fromLines(header[5:8]), step=step)

    def getCopy(self, **kwargs):
        """
        Get the frame with the array copied.

        :return 'Frame': the copied frame.
        """
        return Frame(super().copy(**kwargs), box=self.box, step=self.step)

    def center(self):
        """
        Align circular-mean center with the box center. (one droplet in vacuum)

        FIXME: support droplets or clustering in solution

        :param molecules 'dict': molecule ids -> global atom ids
        """
        theta = self / self.box.span * 2 * np.pi
        sin_ave = np.sin(theta).mean(axis=0)
        cos_ave = np.cos(theta).mean(axis=0)
        center = np.arctan2(sin_ave, cos_ave) * self.box.span / 2 / np.pi
        self[:] += center - self.box.center

    def wrap(self, broken_bonds=False, dreader=None):
        """
        Wrap atoms or molecule centers into the PBC first image.

        :param broken_bonds bool: allow bonds broken by PBC boundaries.
        :param dreader 'oplsua.Reader': datafile reader
        """
        if broken_bonds:
            self[:] = self % self.box.span
            return
        if dreader is None:
            return
        # The unwrapped xyz can directly perform molecule center operation
        for gids in dreader.mols.values():
            center = self[gids, :].mean(axis=0)
            self[gids, :] += center % self.box.span - center

    def write(self,
              fh,
              dreader=None,
              visible=None,
              points=None,
              fmt=('%s', '%.4f', '%.4f', '%.4f')):
        """
        Write XYZ to a file.

        :param fh '_io.TextIOWrapper': file handdle to write out xyz.
        :param dreader 'oplsua.Reader': datafile reader.
        :param visible list: visible atom gids.
        :param points list: additional point to visualize.
        :param fmt tuple: the format of each row.
        """
        data = self[visible] if visible else self
        sel = np.argwhere(~np.isnan(data[:, 0])).flatten()
        index = [symbols.UNKNOWN] * len(sel) if dreader is None else \
                dreader.masses.element[dreader.atoms.type_id.loc[sel]]
        data = pd.DataFrame(data, columns=symbols.XYZU, index=index)
        if points is not None:
            points = pd.DataFrame(points,
                                  index=[symbols.UNKNOWN] * len(points),
                                  columns=symbols.XYZU)
            data = pd.concat((data, points), axis=0)
        fh.write(f'{data.shape[0]}\n\n')
        np.savetxt(fh, data.reset_index().to_numpy(), fmt=fmt)
