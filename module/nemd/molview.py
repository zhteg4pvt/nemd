# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module visualizes the trajectory.
"""
import warnings

import methodtools
import numpy as np
import pandas as pd
import plotly

from nemd import envutils
from nemd import symbols
from nemd import table


class Frame(pd.DataFrame):
    """
    Frame with coordinates, elements, marker sizes, and color info.
    """
    ELEMENT = 'element'
    SIZE = 'size'
    COLOR = 'color'
    ELE_SIZE_CLR = [ELEMENT, SIZE, COLOR]
    COLUMNS = symbols.XYZU + ELE_SIZE_CLR
    # https://webmail.life.nthu.edu.tw/~fmhsu/rasframe/CPKCLRS.HTM
    X_LINE = [0., 0., 0.0, 'X', 20, '#FF1493']
    _metadata = ['trj', 'rdr', 'box', 'elements']

    def __init__(self, trj, rdr=None):
        """
        :param trj 'traj.Traj': the trajectory
        :param rdr `oplsua.Reader`: data file reader
        """
        super().__init__([self.X_LINE] * trj[0].shape[0], columns=self.COLUMNS)
        self.trj = trj
        self.rdr = rdr
        self.box = None
        self.step = None
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        self.update(self.trj[0])
        if self.rdr is None:
            return
        self[self.ELEMENT] = self.rdr.elements.element
        size_map = dict(self.rdr.pair_coeffs.dist.items())
        self[self.SIZE] = self.rdr.atoms.type_id.map(size_map)
        elements = set(self.rdr.masses.element)
        color_map = {x: table.TABLE.loc[x].cpk_color for x in elements}
        self[self.COLOR] = self.rdr.elements.element.map(color_map)

    def update(self, other):
        """
        Update the frame.

        :param other traj.Frame: DataFrame, or object coercible into a DataFrame
        :return Frame: the updated frame
        """
        self[symbols.XYZU] = other
        self.box = other.box
        self.step = other.step
        return self

    def getCoords(self):
        """
        Get the coordinates element wisely.

        :return iterator of (list, Dataframe): information, coordinates
        """
        for element in self.elements:
            sel = self[self[self.ELEMENT] == element]
            yield sel[self.ELE_SIZE_CLR].values[0], sel[symbols.XYZU]

    @methodtools.lru_cache()
    @property
    def elements(self):
        """
        Get the element.

        :return list of str: elements sorted by size
        """
        to_sort = self[[self.ELEMENT, self.SIZE]].drop_duplicates()
        return to_sort.sort_values(by=self.SIZE, ascending=False).element

    def getBonds(self):
        """
        Get the bonds.

        :return DataFrame, dict: bond points, bond style
        """
        if self.rdr is None:
            return
        for aids in self.rdr.bonds.getPairs():
            pnts = self.loc[list(aids)][symbols.XYZU]
            pnts = pd.concat([pnts, pnts.mean().to_frame().transpose()])
            for pnts, idx in zip([pnts[::2], pnts[1::]], aids):
                yield pnts, dict(width=8, color=self.xs(idx).color)

    def getRanges(self):
        """
        Get the ranges.

        :return ndarray: xyz ranges.
        """
        dmin = np.min([x.min(axis=0) for x in self.trj], axis=0)
        dmax = np.max([x.max(axis=0) for x in self.trj], axis=0)
        center = (dmin + dmax / 2)
        span = (dmax - dmin).max()
        return np.array([(center - span), (center + span)]).transpose()

    def iter(self):
        """
        Update self to every trajectory frame during the iteration.

        :return generator: the updated frames
        """
        return (self.update(x) for x in self.trj)


class Figure(plotly.graph_objects.Figure):
    """
    Figure of trajectory frame(s).
    """
    LINE = dict(opacity=0.8, mode='lines', showlegend=False, hoverinfo='skip')
    EDGE = dict(opacity=0.5,
                mode='lines',
                showlegend=False,
                hoverinfo='skip',
                line=dict(width=8, color='#b300ff'))
    PLAY = dict(label="Play",
                method="animate",
                args=[None, dict(fromcurrent=True)])
    PAUSE = dict(label='Pause',
                 method="animate",
                 args=[[None], dict(mode='immediate')])
    MENU = dict(type="buttons",
                showactive=False,
                font={'color': '#000000'},
                direction="left",
                pad=dict(r=10, t=87),
                xanchor="right",
                yanchor="top",
                x=0.1,
                y=0)
    SLIDER = dict(active=0,
                  yanchor="top",
                  xanchor="left",
                  x=0.1,
                  y=0,
                  pad=dict(b=10, t=50),
                  len=0.9,
                  transition=dict(duration=300, easing='cubic-in-out'),
                  currentvalue=dict(prefix='Frame:',
                                    visible=True,
                                    xanchor='right'))
    AXES = ['xaxis', 'yaxis', 'zaxis']
    LAYOUT = dict(template='plotly_dark', overwrite=True, uirevision=True)
    _metadata = Frame._metadata + ['outfile', 'fig']

    def __init__(self, *arg, delay=False, **kwargs):
        """
        :param trj `traj.Traj`: the trajectory
        :param rdr `oplsua.Reader`: data file reader
        :param delay bool: delay the setup if True
        """
        super().__init__()
        self._frm = Frame(*arg, **kwargs)
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Main method.
        """
        self.add_traces(self.traces)
        self.update(frames=self.getFrames())
        with warnings.catch_warnings(record=True):
            # *scattermapbox* is deprecated! Use *scattermap* due to plotly_dark
            self.update_layout(**self.getLayout())
        if envutils.is_interac():
            self.show()

    @property
    def traces(self):
        """
        Set scattered markers for atoms.

        :return list of `Scatter3d`: the traces
        """
        return list(self.scatters) + list(self.lines) + list(self.edges)

    @property
    def scatters(self):
        """
        Set scattered markers for atoms.

        :return Scatter3d iterator: the scattered markers to represent atoms.
        """
        for (element, size, color), coords in self._frm.getCoords():
            yield plotly.graph_objects.Scatter3d(
                x=coords.xu,
                y=coords.yu,
                z=coords.zu,
                opacity=0.9,
                mode='markers',
                name=element,
                marker=dict(size=size, color=color),
                hovertemplate='%{customdata}',
                customdata=coords.index.values)

    @property
    def lines(self):
        """
        Get bond lines.

        :return Scatter3d iterator: the line traces to represent bonds.
        """
        for points, line in self._frm.getBonds():
            yield plotly.graph_objects.Scatter3d(x=points.xu,
                                                 y=points.yu,
                                                 z=points.zu,
                                                 line=line,
                                                 **self.LINE)

    @property
    def edges(self):
        """
        Set box edges.

        :return Scatter3d iterator: the box edges markers.
        """
        for edges in self._frm.box.edges:
            yield plotly.graph_objects.Scatter3d(x=edges[:, 0],
                                                 y=edges[:, 1],
                                                 z=edges[:, 2],
                                                 **self.EDGE)

    def getFrames(self):
        """
        Get animation frames.

        :return list of Frame: the frames
        """
        return [
            plotly.graph_objects.Frame(data=self.traces, name=x.step)
            for x in self._frm.iter()
        ]

    def getLayout(self):
        """
        Update the figure layout.

        :return dict: keyword arguments for layout.
        """
        names = [x['name'] for x in self.frames]
        steps = [dict(label=x, method='animate', args=[[x]]) for x in names]
        sliders = [{**self.SLIDER, **dict(steps=steps)}]
        updatemenus = [{**self.MENU, **dict(buttons=[self.PLAY, self.PAUSE])}]
        return dict(scene=self.scene,
                    sliders=sliders,
                    updatemenus=updatemenus,
                    **self.LAYOUT)

    @property
    def scene(self):
        """
        Return the scene with axis range and styles.

        :return dict: keyword arguments for scene.
        """
        rngs = [dict(range=x, autorange=False) for x in self._frm.getRanges()]
        return dict(**dict(zip(self.AXES, rngs)), aspectmode='cube')
