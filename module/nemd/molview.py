# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module visualizes the trajectory and the LAMMPS data file.
"""
import numpy as np
import pandas as pd
import plotly

from nemd import envutils
from nemd import symbols
from nemd import table


class Frame(pd.DataFrame):
    """
    Frame data with coordinates, elements, marker sizes, and color info.
    """

    _metadata = ['box']

    def __init__(self, data, box=None):
        """
        :param data nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        """
        super().__init__(data)
        self.box = box

    def update(self, other):
        """
        Update the frame.

        :param other traj.Frame: DataFrame, or object coercible into a DataFrame
        """
        self.loc[:, symbols.XYZU] = other
        self.box = other.box


class View:
    """
    Viewer datafile and trajectory frame
    """

    ELEMENT = 'element'
    SIZE = 'size'
    COLOR = 'color'
    X_ELE = 'X'
    X_SIZE = 20
    # https://webmail.life.nthu.edu.tw/~fmhsu/rasframe/CPKCLRS.HTM
    X_COLOR = '#FF1493'
    ELE_SZ_CLR = {ELEMENT: X_ELE, SIZE: X_SIZE, COLOR: X_COLOR}
    LINE_PROP = dict(opacity=0.8,
                     mode='lines',
                     showlegend=False,
                     hoverinfo='skip')
    EPROP = dict(opacity=0.5,
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
    Scatter3d = plotly.graph_objects.Scatter3d
    Frame = plotly.graph_objects.Frame

    def __init__(self, trj=None, rdf=None, outfile=None):
        """
        :param rdf `oplsua.Reader`: datafile reader with atom, bond, element and
            other info.
        """
        self.trj = trj
        self.rdf = rdf
        self.outfile = outfile
        self.fig = plotly.graph_objects.Figure()
        self.data = None
        self.elements = None

    def run(self):
        """
        Main method.
        """
        self.setData()
        self.setElements()
        self.fig.add_traces(self.traces)
        self.fig.update(frames=list(self.frames))
        self.fig.update_layout(**self.layout)
        self.fig.write_html(self.outfile)
        if envutils.is_interac():
            self.fig.show()

    def setData(self):
        """
        Set data frame with coordinates, elements, marker sizes, and color info.

        :param frm 'nemd.traj.Frame': the frame to create the data from.
        """
        if not self.rdf:
            data = pd.DataFrame(self.trj[0], columns=symbols.XYZU)
            self.data = Frame(data.assign(**self.ELE_SZ_CLR),
                              box=self.trj[0].box)
            return

        elements = self.rdf.elements
        smap = {i: x for i, x in self.rdf.pair_coeffs.dist.items()}
        sizes = self.rdf.atoms.type_id.map(smap).to_frame(name=self.SIZE)
        uq_elements = set(self.rdf.masses.element)
        color_map = {x: table.TABLE.loc[x].cpk_color for x in uq_elements}
        colors = elements.element.map(color_map).to_frame(name=self.COLOR)
        data = [self.rdf.xyz, elements, sizes, colors]
        self.data = Frame(pd.concat(data, axis=1), box=self.rdf.box)

    def setElements(self):
        """
        Set elements and sizes.
        """
        elements = self.data[[self.ELEMENT, self.SIZE]].drop_duplicates()
        elements.sort_values(by=self.SIZE, ascending=False, inplace=True)
        self.elements = elements.element

    @property
    def traces(self):
        traces = list(self.scatters) + list(self.edges)
        return traces + list(self.lines) if self.rdf else traces

    @property
    def scatters(self,
                 opacity=0.9,
                 mode='markers',
                 hovertemplate='%{customdata}'):
        """
        Set scattered markers for atoms.

        :return Scatter3d iterator: the scattered markers to represent atoms.
        """
        for element in self.elements:
            selected = self.data[self.data[self.ELEMENT] == element]
            sizes = selected[self.SIZE]
            colors = selected[self.COLOR]
            yield plotly.graph_objects.Scatter3d(
                x=selected.xu,
                y=selected.yu,
                z=selected.zu,
                opacity=opacity,
                mode=mode,
                name=element,
                marker=dict(size=sizes.values[0], color=colors.values[0]),
                hovertemplate=hovertemplate,
                customdata=selected.index.values)

    @property
    def lines(self, width=8):
        """
        Set lines for bonds.

        :return Scatter3d iterator: the line traces to represent bonds.
        """
        for _, _, atom_id1, atom_id2 in self.rdf.bonds.itertuples():
            pnts = self.data.loc[[atom_id1, atom_id2]][symbols.XYZU]
            pnts = pd.concat([pnts, pnts.mean().to_frame().transpose()])
            for pnts, idx in zip([pnts[::2], pnts[1::]], [atom_id1, atom_id2]):
                line = dict(width=width, color=self.data.xs(idx).color)
                yield self.Scatter3d(x=pnts.xu,
                                     y=pnts.yu,
                                     z=pnts.zu,
                                     line=line,
                                     **self.LINE_PROP)

    @property
    def edges(self):
        """
        Set box edges.

        :return Scatter3d iterator: the box edges markers.
        """
        for edges in self.data.box.edges:
            yield self.Scatter3d(x=edges[:, 0],
                                 y=edges[:, 1],
                                 z=edges[:, 2],
                                 **self.EPROP)

    @property
    def frames(self):
        """
        Set animation from trajectory frames.

        :return Frame iterator: the frames
        """
        for frm in self.trj:
            self.data.update(frm)
            yield self.Frame(data=self.traces, name=str(frm.step))

    @property
    def layout(self, method='animate'):
        """
        Update the figure layout.
        """
        menu, sliders = self.MENU, []
        if self.trj:
            menu = {**menu, **dict(buttons=[self.PLAY, self.PAUSE])}
            steps = [str(x.step) for x in self.trj]
            steps = [dict(label=x, method=method, args=[[x]]) for x in steps]
            sliders = [{**self.SLIDER, **dict(steps=steps)}]
        return dict(scene=self.scene,
                    sliders=sliders,
                    updatemenus=[menu],
                    **self.LAYOUT)

    @property
    def scene(self, autorange=False, aspectmode='cube'):
        """
        Return the scene with axis range and styles.

        :param autorange bool: whether to let the axis range be adjusted
            automatically.
        :param aspectmode str: how the 3D scene's axes are scaled
        :return dict: keyword arguments for preference.
        """
        dmin = np.min([x.min(axis=0) for x in self.trj], axis=0)
        dmax = np.max([x.max(axis=0) for x in self.trj], axis=0)
        center = (dmin + dmax / 2)
        span = (dmax - dmin).max()
        ranges = [[x, y] for x, y in zip((center - span), (center + span))]
        ranges = [dict(range=x, autorange=autorange) for x in ranges]
        return dict(**dict(zip(self.AXES, ranges)), aspectmode=aspectmode)
