# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module handles the Box, which represents the periodic boundary conditions.
"""
import collections
import io
import itertools
import math
import re

import numpy as np
import pandas as pd

from nemd import symbols


class Base(pd.DataFrame):
    """
    Base class to handle a datafile block.
    """

    NAME = 'Block'
    COLUMN_LABELS = ['column_labels']
    ID_COLS = None  # atom (or molecule) id columns
    TYPE_COL = None
    POUND = symbols.POUND
    SPACE = symbols.SPACE
    SPACE_PATTERN = symbols.SPACE_PATTERN
    LABEL = 'label'
    FLOAT_FMT = '%.4f'
    FMT = None

    def __init__(self, data=None, index=None, columns=None, **kwargs):
        """
        Initialize the Mass object.

        :param data: `pandas.DataFrame`: the data to initialize the object.
        :param index: `pandas.Index`: the index to initialize the object.
        :param columns: `list`: the column labels to initialize the object.
        """
        if not isinstance(data, pd.DataFrame) and columns is None:
            columns = self.COLUMN_LABELS
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
    def new(cls, *args, **kwargs):
        """
        Create a new instance of the (sub-)class.

        :return instance of Block (sub-)class: the new instance.
        """
        return cls(*args, **kwargs)

    @classmethod
    def fromLines(cls,
                  lines,
                  *args,
                  names=None,
                  index_col=None,
                  header=None,
                  sep=SPACE_PATTERN,
                  quotechar=POUND,
                  **kwargs):
        """
        Construct a new instance from a list of lines.

        :param names list: Sequence of column labels to apply.
        :param index_col int: Column(s) to use as row label(s)
        :param header int, ‘infer’ or None: the row number defining the header
        :param lines list: list of lines to parse.
        :param sep str: Character or regex pattern to treat as the delimiter.
        :param quotechar str: Character used to denote the start and end of a
            quoted item

        :return instance of Block (sub-)class: the parsed object.
        """
        if names is None:
            names = cls.COLUMN_LABELS
        df = pd.read_csv(io.StringIO(''.join(lines)),
                         *args,
                         names=names,
                         index_col=index_col,
                         header=header,
                         sep=sep,
                         quotechar=quotechar,
                         **kwargs)
        if df.empty:
            return cls(df)
        if cls.ID_COLS is not None:
            df[cls.ID_COLS] -= 1
        if cls.TYPE_COL is not None:
            df[cls.TYPE_COL] -= 1
        if index_col == 0:
            df.index -= 1
        return cls(df)

    def write(self,
              hdl,
              join=None,
              index_column=None,
              as_block=True,
              columns=None,
              sep=SPACE,
              header=False,
              float_format=FLOAT_FMT,
              mode='a',
              quotechar=POUND,
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
        if columns is None:
            columns = self.COLUMN_LABELS
        if index_column is not None and index_column in columns:
            columns = [x for x in columns if x != index_column]
        if join is not None:
            columns.extend(join.columns)
        if as_block and self.NAME:
            hdl.write(self.NAME + '\n\n')
        if self.TYPE_COL is not None:
            self[self.TYPE_COL] += 1
        if self.ID_COLS is not None:
            self[self.ID_COLS] += 1
        data = self if join is None else self.join(join)
        if index_column is not None:
            data = data.set_index(index_column)
        data.index += 1
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
        data.index -= 1
        if self.TYPE_COL is not None:
            self[self.TYPE_COL] -= 1
        if self.ID_COLS is not None:
            self[self.ID_COLS] -= 1
        if as_block:
            hdl.write('\n')

    def allClose(self, other, atol=1e-08, rtol=1e-05, equal_nan=False):
        """
        Returns a boolean where two arrays are equal within a tolerance

        :param other: the other data reader to compare against.
        :type other: float
        :param atol: The relative tolerance parameter (see Notes).
        :type atol: float
        :param rtol: The absolute tolerance parameter (see Notes).
        :type rtol: float
        :param equal_nan: If True, NaNs are considered close.
        :type equal_nan: bool
        :return: whether two data are close.
        :rtype: bool
        """
        sf = self.select_dtypes(include=['float'])
        of = other.select_dtypes(include=['float'])
        if sf.shape != of.shape:
            return False
        if not np.allclose(sf, of, atol=atol, rtol=rtol, equal_nan=equal_nan):
            return False
        nsf = self.select_dtypes(exclude=['float'])
        nof = other.select_dtypes(exclude=['float'])
        return nsf.equals(nof)


class Box(Base):
    """
    The Box block representing the periodic boundary conditions.
    """

    NAME = ''
    LABEL = 'box'
    LO, HI = 'lo', 'hi'
    INDEX = ['x', 'y', 'z']
    ORIGIN = [0, 0, 0]
    LIMIT_CMT = '{limit}_cmt'
    LO_LABEL, HI_LABEL = LIMIT_CMT.format(limit=LO), LIMIT_CMT.format(limit=HI)
    COLUMN_LABELS = [LO, HI, LO_LABEL, HI_LABEL]
    LO_CMT = [x + y for x, y in itertools.product(INDEX, [LO])]
    HI_CMT = [x + y for x, y in itertools.product(INDEX, [HI])]
    FLT_RE = "[+-]?[\d\.\d]+"
    LO_HI = [f'{x}{symbols.SPACE}{y}' for x, y in zip(LO_CMT, HI_CMT)]
    RE = re.compile(f"^{FLT_RE}\s+{FLT_RE}\s+({'|'.join(LO_HI)}).*$")

    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['_span', '_tilt']
    _internal_names_set = set(_internal_names)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._span = None
        self._tilt = None

    @property
    def volume(self):
        """
        Get the volume of the frame.

        :param float: the volume of the frame
        """
        return np.prod(self.span)

    @property
    def span(self):
        """
        Set and cache the span of the box.

        :return: the span of the box.
        :rtype: 'numpy.ndarray'
        """
        if self._span is not None:
            return self._span
        self._span = (self.hi - self.lo).values
        return self._span

    @classmethod
    def fromEdges(cls, edges):
        """
        Box built from these edges and origin.

        :param list: the box edges.
        """
        data = {
            cls.LO: cls.ORIGIN,
            cls.HI: edges,
            cls.LO_LABEL: cls.LO_CMT,
            cls.HI_LABEL: cls.HI_CMT
        }
        return cls(data=data)

    @classmethod
    def fromVecs(cls, vecs):
        """
        Construct a box instance from lattice vectors.

        Crystallographic general triclinic representation of a simulation box
        https://docs.lammps.org/Howto_triclinic.html

        """
        a_vec, b_vec, c_vec, alpha, beta, gamma = vecs
        lx = a_vec
        xy = b_vec * math.cos(math.radians(gamma))
        xz = c_vec * math.cos(math.radians(beta))
        ly = math.sqrt(b_vec**2 - xy**2)
        yz = (b_vec * c_vec * math.cos(math.radians(alpha)) - xy * xz) / ly
        lz = math.sqrt(c_vec**2 - xz**2 - yz**2)
        box = cls.fromEdges([lx, ly, lz])
        box._tilt = pd.DataFrame([[xy, xz, yz]])
        return box

    def write(self, fh, index=False, as_block=False, **kwargs):
        """
        Write the box into the handler.

        :param hdl `_io.TextIOWrapper` or `_io.StringIO`: write to this handler.
        :param as_block `bool`: whether to write the data as a block.
        :param index `bool`: whether to write the index.
        """
        super().write(fh, index=index, as_block=as_block, **kwargs)
        if self._tilt is not None:
            self._tilt.to_csv(fh,
                              header=False,
                              sep=' ',
                              lineterminator=' xy xz yz\n',
                              index=index,
                              float_format=self.FLOAT_FMT)
        fh.write("\n")

    @property
    def edges(self):
        """
        Get the edges from point list of low and high points.

        :return 12x2x3 numpy.ndarray: 12 edges of the box, and each edge
            contains two points.
        """
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
