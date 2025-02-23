# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver calculates and visualizes hexagonal 2D lattice in the real and
reciprocal spaces.
"""
import collections
import functools
import math
import sys

from nemd import jobutils
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import pd
from nemd import plotutils

FLAG_MILLER_INDICES = '-miller_indices'


class Reciprocal:
    """
    This class is used to plot the reciprocal space on axis.
    """

    TITLE = 'Reciprocal Space'
    PLANE_PROPS = dict(color='0.8', linestyle='--', alpha=0.5)
    GRID_PROPS = dict(marker='o', alpha=0.5)
    EQUAL = 'equal'
    QV_PROPS = dict(angles='xy',
                    scale_units='xy',
                    scale=1,
                    units='dots',
                    width=3)
    AW_PROPS = dict(linestyle="-",
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color='r')
    VEC_TXT = r'$\vec {sym}^*$'
    R_SYM = 'r'

    def __init__(self, ax, vecs=None, miller=None):
        """
        :param ax 'matplotlib.axes._axes.Axes': axis to plot
        :param vecs 'pandas.DataFrame': a, b vectors
        :param miller: the Miller Indexes
        """
        self.ax = ax
        self.vecs = vecs
        self.miller = miller
        self.m_vecs = None
        self.vec = None
        self.xlim = None
        self.ylim = None
        self.grids = []
        self.quivers = []
        self.origin = np.array([0., 0.])

    def run(self):
        """
        Plot the grids and vectors.
        """
        self.setMiller()
        self.setVec()
        self.setGridsAndLim()
        self.plotGrids()
        self.quiver(self.vecs.a1)
        self.quiver(self.vecs.a2)
        self.annotate(self.m_vecs.a1)
        self.annotate(self.m_vecs.a2)
        self.quiver(self.vec, color='g')
        self.legend()

    def setMiller(self):
        """
        Set the vectors for Miller Plane by converting real space Miller indexes
        to the reciprocal ones.
        """
        self.m_vecs = self.vecs * self.miller

    def setVec(self):
        """
        Plot the vector summation.
        """
        self.vec = self.m_vecs.a1 + self.m_vecs.a2
        self.vec.rename(self.R_SYM, inplace=True)

    def setGridsAndLim(self, num=6):
        """
        Calculate the grids based on the lattice vectors, set rectangular limits
        based on the grids, and crop the grids by the rectangular.

        :param num int: the minimum number of duplicates along each lattice vec.
        """
        mill = math.ceil(max(self.miller + [1. / x for x in self.miller if x]))
        num = max(mill, num) + 1
        # Uniform grids in a_vec and b_vec directions
        idxs = np.meshgrid(range(-num, num + 1), range(-num, num + 1))
        idx = np.stack(idxs, axis=-1)
        xs = np.dot(idx, self.vecs.iloc[0])
        ys = np.dot(idx, self.vecs.iloc[1])
        # Four points of a parallelogram starting from bottom counter-clockwise
        arg_idxs = [ys.argmin(), xs.argmax(), ys.argmax(), xs.argmin()]
        shapes = [ys.shape, xs.shape, ys.shape, xs.shape]
        idxs = [np.unravel_index(x, y) for x, y in zip(arg_idxs, shapes)]
        parlgrm_pnts = [np.array([xs[x], ys[x]]) for x in idxs]
        rotated = collections.deque(parlgrm_pnts)
        rotated.rotate(1)
        rect_pnts = np.stack((parlgrm_pnts, rotated)).mean(axis=0)
        # Set the rectangular x, y limits
        self.xlim = rect_pnts[:, 0].min(), rect_pnts[:, 0].max()
        self.ylim = rect_pnts[:, 1].min(), rect_pnts[:, 1].max()
        x_buf = (self.xlim[1] - self.xlim[0]) / (num * 2 + 1)
        sel_x = (xs >= self.xlim[0] - x_buf) & (xs <= self.xlim[1] + x_buf)
        y_buf = (self.ylim[1] - self.ylim[0]) / (num * 2 + 1)
        sel_y = (ys >= self.ylim[0] - y_buf) & (ys <= self.ylim[1] + y_buf)
        sel = np.logical_and(sel_x, sel_y)
        self.grids.append(xs[sel] + self.origin[0])
        self.grids.append(ys[sel] + self.origin[1])

    def plotGrids(self):
        """
        Plot the selected grids.
        """
        self.ax.scatter(*[x.tolist() for x in self.grids], **self.GRID_PROPS)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect(self.EQUAL)
        self.ax.set_title(self.TITLE)

    def quiver(self, vec, color='b'):
        """
        Plot a quiver for the vector.

        :param vec 'pandas.core.series.Series': two points as a vector
        :param color str: the color of the annotate
        """
        if not any(vec):
            return
        qv = self.ax.quiver(*self.origin, *vec, **self.QV_PROPS, color=color)
        text = self.VEC_TXT.format(sym=vec.name)
        self.ax.annotate(text, vec, color=color)
        label = f"{text} ({vec.iloc[0]:.4g}, {vec.iloc[1]:.4g})"
        self.quivers.append([qv, label])

    def annotate(self, vec):
        """
        Annotate arrow for the vector.

        :param vec list: list of two points
        """
        if not any(vec):
            return
        self.ax.annotate('', vec, xytext=self.origin, arrowprops=self.AW_PROPS)

    def legend(self):
        """
        Set the legend.
        """
        quivers = [x[0] for x in self.quivers]
        labels = [x[1] for x in self.quivers]
        self.ax.legend(quivers, labels, loc='upper right')


class Real(Reciprocal):
    """
    This class is used to plot the real space on axis.
    """

    TITLE = 'Real Space'
    VEC_TXT = r'$\vec {sym}$'

    def setVec(self):
        """
        Plot the normal to a plane.
        """
        vec = self.getNormal(factor=1) - self.getNormal(factor=0)
        self.vec = pd.Series(vec, name=self.R_SYM)

    def getNormal(self, factor=1):
        """
        Get the intersection between the plane and its normal passing origin.

        :param factor int: by this factor the Miller plane is moved.
        :return 1x3 'numpy.ndarray': the intersection point
        """
        pnt1, pnt2 = self.getPlane(factor=factor)
        vec = pnt2 - pnt1  # vector along the plane
        normal = np.dot([[0, 1], [-1, 0]], vec)  # normal to the plane
        # Interaction is on the normal: factor * normal
        # Interaction is on the plane: fac2 * vec + pnt1
        # Equation: normal * factor - vec * fac2 = pnt1
        factor, _ = np.linalg.solve(np.transpose([normal, -vec]), pnt1)
        return normal * factor

    @functools.cache
    def getPlane(self, factor=1):
        """
        Get the intersection points between the Miller plane and the lines of
        the x,y limits.

        :param factor int: the Miller index plane is moved by this factor.
        :return 'np.ndarray': two points (rows) on the Miller
        """
        nonzero = self.m_vecs.any()
        if factor and nonzero.all():
            return self.m_vecs.T.values * factor

        index = nonzero.to_list().index(True)
        pnt = self.m_vecs.iloc[:, index] if factor else self.origin
        if nonzero.all():
            # Vectors are available, and the subtraction defines the direction
            vec = self.m_vecs.a2 - self.m_vecs.a1
        else:
            # If one vector, the other's lattice vector defines the direction
            vec = self.vecs.iloc[:, nonzero.to_list().index(False)]
        pnts = np.array([pnt, pnt + vec])
        return pnts * factor if factor else pnts

    def run(self):
        """
        Main method to run.
        """
        super().run()
        self.plotPlane(factor=-1)
        self.plotPlane(factor=0)
        self.plotPlane(factor=1)

    def plotPlane(self, factor=1, num=1000):
        """
        Plot the Miller plane moved by the index factor.

        :param factor int: by this factor the Miller plane is moved.
        :param num int: the span over this number is the buffer
        """
        # factor * (b_pnt - a_pnt) + a_pnt = lim
        a_pnt, b_pnt = self.getPlane(factor=factor)
        vec, pnts = b_pnt - a_pnt, []
        if vec[0]:
            factors = [(x - a_pnt[0]) / vec[0] for x in self.xlim]
            pnts += [f * (b_pnt - a_pnt) + a_pnt for f in factors]
        if vec[1]:
            factors = [(y - a_pnt[1]) / vec[1] for y in self.ylim]
            pnts += [f * (b_pnt - a_pnt) + a_pnt for f in factors]
        pnts = np.array(pnts)
        if pnts.shape[0] == 4:
            # The intersection points are on x low, x high, y low, y high
            buf = (self.xlim[1] - self.xlim[0]) / num
            sel = pnts[:, 0] >= self.xlim[0] - buf
            sel &= pnts[:, 0] <= self.xlim[1] + buf
            buf = (self.ylim[1] - self.ylim[0]) / num
            sel &= pnts[:, 1] >= self.ylim[0] - buf
            sel &= pnts[:, 1] <= self.ylim[1] + buf
            pnts = pnts[sel][:2, :]
        x_vals, y_vals = pnts.T
        if np.isclose(*x_vals):
            # The plane is vertical
            self.ax.vlines(x_vals.mean(), *y_vals, **self.PLANE_PROPS)
            return
        self.ax.plot(x_vals, y_vals, **self.PLANE_PROPS)


class RecipSp(logutils.Base):
    """
    Class to set and plot the reciprocal space lattice vectors for 2D graphene.

    References:
    https://www.youtube.com/watch?v=cdN6OgwH8Bg
    https://en.wikipedia.org/wiki/Reciprocal_lattice
    """

    PNG_EXT = '.png'
    AX = ['a1', 'a2']
    XY = ['x', 'y']

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver':  Parsed command-line options
        """
        super().__init__(**kwargs)
        self.options = options
        self.origin = np.array([0., 0.])
        self.real = None
        self.recip = None

    def run(self):
        """
        Main method to run.
        """
        self.setReal()
        self.setReciprocal()
        self.plot()

    def setReal(self):
        """
        Define real space lattice vector with respect to the origin.

        Characteristic length of a hexagon is sqrt(3) x the edge length
        https://physics.stackexchange.com/questions/664945/integration-over-first-brillouin-zone
        """
        # Primitive lattice vector of graphene
        data = np.array([[math.sqrt(3) / 2., 0.5], [math.sqrt(3) / 2., -0.5]])
        data *= math.sqrt(3)
        self.real = pd.DataFrame(data.T, columns=self.AX, index=self.XY)

    def setReciprocal(self):
        """
        Set the reciprocal lattice vectors based on the real ones.

        https://en.wikipedia.org/wiki/Reciprocal_lattice (Two dimensions)
        """
        ab_norm = np.dot([[0, -1], [1, 0]], self.real)
        ba_norm = ab_norm[:, ::-1]
        column_dot = np.multiply(self.real, ba_norm).sum(axis=0).tolist()
        recip = 2 * np.pi * ba_norm / column_dot
        self.recip = pd.DataFrame(recip, columns=self.AX, index=self.XY)

    def plot(self):
        """
        Plot the real and reciprocal paces.
        """
        with plotutils.get_pyplot(inav=self.options.INTERAC) as plt:
            fig = plt.figure(figsize=(15, 9))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            plane = [1. / x if x else 0 for x in self.options.miller_indices]
            ltp = Real(ax1, vecs=self.real, miller=plane)
            ltp.run()
            ltp_vec = ', '.join(map('{:.4g}'.format, ltp.vec))
            self.log(f"The vector in real space is ({ltp_vec}) with "
                     f"{np.linalg.norm(ltp.vec):.4g} being the norm.")
            rltp = Reciprocal(ax2,
                              vecs=self.recip,
                              miller=self.options.miller_indices)
            rltp.run()
            rltp_vec = ', '.join(map('{:.4g}'.format, rltp.vec))
            self.log(f"The vector in reciprocal space is ({rltp_vec}) with "
                     f"{np.linalg.norm(rltp.vec):.4g} being the norm.")
            self.log(
                f"The cross product is {np.cross(ltp.vec, rltp.vec): .4g}")
            self.log(
                f"The product is {np.dot(ltp.vec, rltp.vec) / np.pi: .4g} * pi"
            )
            idxs = ' '.join(map('{:.4g}'.format, self.options.miller_indices))
            fig.suptitle(f'Miller indices ({idxs})')
            fig.tight_layout()
            fname = self.options.JOBNAME + self.PNG_EXT
            fig.savefig(fname)
            jobutils.add_outfile(fname, file=True)
            self.log(f'Figure saved as {fname}')


class Parser(parserutils.Driver):
    """
    The argument parser with additional validations.
    """

    def setUp(self):
        """
        The user-friendly command-line parser.
        """
        self.add_argument(FLAG_MILLER_INDICES,
                          metavar=FLAG_MILLER_INDICES[1:].upper(),
                          default=[0.5, 2],
                          type=parserutils.type_float,
                          nargs='+',
                          help='Plot the planes of this Miller indices.')

    def parse_args(self, *args, **kwargs):
        """
        See parent class for details.
        """
        options = super().parse_args(*args, **kwargs)
        if len(options.miller_indices) != 2:
            self.error('Please provide two floats as the Miller indices.')
        if not any(options.miller_indices):
            self.error(f'Miller indices cannot be all zeros.')
        return options


if __name__ == "__main__":
    parser = Parser(__file__, descr=__doc__)
    options = parser.parse_args(sys.argv[1:])
    with logutils.Script(options) as logger:
        recip_sp = RecipSp(options, logger=logger)
        recip_sp.run()
