# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module handles the Box, which represents the periodic boundary conditions.
"""
import collections
import io
import math

import methodtools
import numpy as np
import pandas as pd

from nemd import envutils
from nemd import numbautils
from nemd import symbols


class Base(pd.DataFrame):
    """
    Base class to handle a datafile block.
    """
    COLUMNS = ['label']  # Column labels
    NAME = 'Block'  # Block header
    LABEL = 'label'  # Counting suffix
    ID_COLS = None  # Atom (or molecule) ids
    TYPE_COL = None  # Atom type
    FMT = None  # when use np.savetxt to speed up

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        Initialize the Mass object.

        :param data: `pandas.DataFrame` or int: the data or the row number.
        :param index: `pandas.Index`: the index to initialize the object.
        :param columns: `list`: the column labels to initialize the object.
        """
        if not isinstance(data, pd.DataFrame) and columns is None:
            columns = self.COLUMNS
        if isinstance(data, int):
            data = np.ones((data, len(columns)), dtype=np.int32)
        super().__init__(data=data, index=index, columns=columns, **kwargs)

    @classmethod
    @property
    def _constructor(cls):
        """
        Return the constructor of the class.

        :return (sub-)class of 'Block': the constructor of the class
        """
        return cls

    @classmethod
    def fromLines(cls,
                  lines,
                  names=None,
                  index_col=None,
                  sep=r'\s+',
                  quotechar=symbols.POUND,
                  **kwargs):
        """
        Construct a new instance from a list of lines.

        :param lines list: list of lines to parse.
        :param names list: Sequence of column labels to apply.
        :param index_col int: Column(s) to use as row label(s)
        :param sep str: Character or regex pattern to treat as the delimiter.
        :param quotechar str: Character used to denote the start and end of a
            quoted item

        :return instance of Block (sub-)class: the parsed object.
        """
        if names is None:
            names = cls.COLUMNS
        df = pd.read_csv(io.StringIO(''.join(lines)),
                         names=names,
                         index_col=index_col,
                         sep=sep,
                         quotechar=quotechar,
                         **kwargs)
        if df.empty:
            return cls(df)
        cls.shift(df, delta=-1, index=index_col is not None)
        return cls(df)

    @classmethod
    def shift(cls, df, delta=1, index=True):
        """
        Shift the id, type, and index columns by delta.

        :param df `pd.DataFrame`: the dataframe to shift
        :param delta int: the delta to shift by
        :param index bool: shift the index if True
        """
        if cls.ID_COLS is not None:
            df[cls.ID_COLS] += delta
        if cls.TYPE_COL is not None:
            df[cls.TYPE_COL] += delta
        if index:
            df.index += delta

    def write(self,
              hdl,
              join=None,
              index_column=None,
              as_block=True,
              columns=None,
              sep=symbols.SPACE,
              header=False,
              float_format=symbols.FLOAT_FMT,
              mode='a',
              quotechar=symbols.POUND,
              **kwargs):
        """
        Write the data to a text stream.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler
        :param join `Block`: the data to join with.
        :param index_column: the column to use as the index.
        :param as_block `bool`: whether to write the data as a block.
        :param columns list: the labels of the columns to write out.
        :param sep `str`: the separator to use.
        :param header `bool`: whether to write the column names as the header.
        :param float_format `str`: the format to use for floating point numbers.
        :param mode `str`: the mode to use for writing.
        :param quotechar `str`: the quote character to use.
        """
        if not self.size:
            return
        # Columns
        if columns is None:
            columns = self.COLUMNS
        if index_column is not None and index_column in columns:
            columns = [x for x in columns if x != index_column]
        if join is not None:
            columns.extend(join.columns)
        # Join
        data = self if join is None else self.join(join)
        # Index
        if index_column is not None:
            data = data.set_index(index_column)
        self.shift(data)
        # Write
        if as_block and self.NAME:
            hdl.write(self.NAME + '\n\n')
        if self.FMT:
            np.savetxt(hdl, data.reset_index().values, fmt=self.FMT)
        else:
            data.to_csv(hdl,
                        columns=columns,
                        sep=sep,
                        header=header,
                        float_format=float_format,
                        mode=mode,
                        quotechar=quotechar,
                        **kwargs)
        if as_block:
            hdl.write('\n')
        self.shift(data, delta=-1)

    def allClose(self, other, **kwargs):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other float: the other data reader to compare against.
        :return bool: whether two data are close.
        """
        included = self.select_dtypes(include=['float'])
        others = other.select_dtypes(include=['float'])
        if included.shape != others.shape:
            return False
        if not np.allclose(included, others, **kwargs):
            return False
        excluded = self.select_dtypes(exclude=['float'])
        return excluded.equals(other.select_dtypes(exclude=['float']))


class BoxOrig(Base):
    """
    The simulation box (e.g., periodic boundary conditions).
    """

    NAME = None
    LABEL = 'box'
    LO, HI = 'lo', 'hi'
    LIMIT_CMT = '{limit}_cmt'
    LO_LABEL = LIMIT_CMT.format(limit=LO)
    HI_LABEL = LIMIT_CMT.format(limit=HI)
    COLUMNS = [LO, HI, LO_LABEL, HI_LABEL]

    # https://pandas.pydata.org/docs/development/extending.html
    _metadata = ['tilt']

    def __init__(self, *args, tilt=None, **kwargs):
        """
        :param tilt list: xy, xz, yz tilt factors
        """
        super().__init__(*args, **kwargs)
        self.tilt = tilt

    @property
    def volume(self):
        """
        Get the box volume.

        :param float: the box volume
        """
        return np.prod(self.span)

    @methodtools.lru_cache()
    @property
    def span(self):
        """
        Set and cache the span of the box.

        FIXME: triclinic support

        :return 'numpy.ndarray': the span of the box.
        """
        return (self.hi - self.lo).values

    @classmethod
    def fromParams(cls,
                   al,
                   bl=None,
                   cl=None,
                   alpha=90,
                   beta=90,
                   gamma=90,
                   tilted=True):
        """
        Construct a box instance from lattice vectors.

        Crystallographic general triclinic representation of a simulation box:
        https://docs.lammps.org/Howto_triclinic.html

        Lattice constant:
        https://en.wikipedia.org/wiki/Lattice_constant

        :param al float: lattice constant a
        :param bl float: lattice constant b
        :param cl float: lattice constant c
        :param alpha float: angles alpha
        :param beta float: angles beta
        :param gamma float: angles gamma
        :return `Box`: the Box build from lattice vectors
        """
        if not tilted:
            assert all(x == 90 for x in [alpha, beta, gamma])
        if bl is None:
            bl = al
        if cl is None:
            cl = bl
        xy = bl * math.cos(math.radians(gamma))
        ly = math.sqrt(bl**2 - xy**2)
        xz = cl * math.cos(math.radians(beta))
        yz = (bl * cl * math.cos(math.radians(alpha)) - xy * xz) / ly
        lz = math.sqrt(cl**2 - xz**2 - yz**2)
        data = {cls.LO: [0, 0, 0], cls.HI: [al, ly, lz], **cls.getLabels()}
        return cls(data=data, tilt=[xy, xz, yz] if tilted else None)

    @methodtools.lru_cache()
    @classmethod
    def getLabels(cls):
        """
        Get the tailing labels.

        :return dict: the tailing labels of different dimensions
        """
        lo_labels = [f'{d}{cls.LO}' for d in symbols.XYZ]
        hi_labels = [f'{d}{cls.HI}' for d in symbols.XYZ]
        return {cls.LO_LABEL: lo_labels, cls.HI_LABEL: hi_labels}

    def write(self, fh, index=False, as_block=False, **kwargs):
        """
        Write the box into the handler.

        :param fh `_io.TextIOWrapper` or `_io.StringIO`: write to this handler.
        :param index `bool`: whether to write the index.
        :param as_block `bool`: whether to write the data as a block.
        """
        super().write(fh, index=index, as_block=as_block, **kwargs)
        if self.tilt:
            tilt = [symbols.FLOAT_FMT % x for x in self.tilt]
            fh.write(' '.join(tilt + ['xy', 'xz', 'yz']) + "\n")
        fh.write("\n")

    @property
    def edges(self):
        """
        Get the edges from point list of low and high points.

        :return 12x2x3 numpy.ndarray: 12 edges of the box, and each edge
            contains two points.
        """
        # FIXME: triclinic support
        # Three edges starting from the [xlo, ylo, zlo]
        lo_xyzs = np.array([self.lo.values] * 3, dtype=float)
        lo_points = lo_xyzs.copy()
        np.fill_diagonal(lo_points, self.hi.values)
        lo_edges = np.stack((lo_xyzs, lo_points), axis=1)
        # Three edges starting from the [xhi, yhi, zhi]
        hi_xyzs = np.array([self.hi.values] * 3, dtype=float)
        hi_points = hi_xyzs.copy()
        np.fill_diagonal(hi_points, self.lo)
        hi_edges = np.stack((hi_xyzs, hi_points), axis=1)
        # Six edges connecting the open ends of the known edges
        spnts = collections.deque([x[1] for x in lo_edges])
        epnts = collections.deque([x[1] for x in hi_edges])
        epnts.rotate(1)
        oedges = [[x, y] for x, y in zip(spnts, epnts)]
        epnts.rotate(1)
        oedges += [[x, y] for x, y in zip(spnts, epnts)]
        return np.concatenate((lo_edges, hi_edges, np.array(oedges)))

    def norm(self, vecs):
        """
        Calculate the PBC distance of the vectors.

        FIXME: triclinic support

        :param vecs `np.array`: the vectors
        :return `np.ndarray`: the PBC distances
        """
        for idx in range(3):
            func = lambda x: math.remainder(x, self.span[idx])
            vecs[:, idx] = np.frompyfunc(func, 1, 1)(vecs[:, idx])
        return np.linalg.norm(vecs, axis=1)


class BoxNumba(BoxOrig):
    """
    Base class sped up with numba.
    """

    def norm(self, vecs):
        """
        Calculate the PBC distance of the vectors.

        FIXME: triclinic support

        :param vecs `np.array`: the vectors
        :return `np.ndarray`: the PBC distances
        """
        return np.array(numbautils.norm(vecs, self.span))


Box = BoxOrig if envutils.is_original() else BoxNumba