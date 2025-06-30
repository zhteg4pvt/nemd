# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Calculate and visualize 2D lattice in the real and reciprocal spaces.
"""
import functools
import math
import sys

from nemd import jobutils
from nemd import logutils
from nemd import np
from nemd import parserutils
from nemd import pd
from nemd import plotutils


class Recip(logutils.Base):
    """
    Calculate and plot the reciprocal space on axis.
    """
    NAME = 'Reciprocal Space'

    def __init__(self, lat, ax=None, **kwargs):
        """
        :param lat 'pandas.DataFrame': lattice vectors
        :param ax 'matplotlib.axes._axes.Axes': axis to plot
        """
        super().__init__(**kwargs)
        self.lat = pd.DataFrame(lat, index=('x', 'y'), columns=('a1', 'a2'))
        self.ax = ax
        self.qvs = {}
        self.scaled = None
        self.origin = np.array([0., 0.])
        self.vec = pd.Series([0.0, 0.0], name='r')
        self.setUp()
        self.logNorm()

    def setUp(self):
        """
        Set up.
        """
        self.scaled = self.lat * self.options.miller_indices
        self.vec[:] = self.scaled.sum(axis=1)

    def logNorm(self):
        """
        Log the vector norm.
        """
        self.log(f"The {self.NAME.lower()} vector {self.vec.values} has a "
                 f"norm of {np.linalg.norm(self.vec):.4g}")

    def plot(self, ax):
        """
        Plot.

        :param 'matplotlib.axes._axes.Axes': the plot on this axis.
        """
        self.ax = ax
        self.grid()
        self.quiver(self.lat.a1)
        self.quiver(self.lat.a2)
        self.arrow(self.scaled.a1)
        self.arrow(self.scaled.a2)
        self.quiver(self.vec, color='g')
        self.legend()

    def grid(self):
        """
        Plot the grids.
        """
        self.ax.scatter(self.grids[:, 0],
                        self.grids[:, 1],
                        marker='o',
                        alpha=0.5)
        self.ax.set_xlim(self.lim[0])
        self.ax.set_ylim(self.lim[1])
        self.ax.set_aspect('equal')
        self.ax.set_title(self.NAME)

    @functools.cached_property
    def grids(self):
        """
        Return the meshed points cropped by a rectangular.

        :return np.ndarray: grid points.
        """
        return self.crop(self.meshed.reshape(-1, 2))

    def crop(self, pnts):
        """
        Crop the points with a rectangular.

        :param pnts np.ndarray: the input points.
        :return np.ndarray: the cropped points.
        """
        selected = (pnts > self.lim[:, 0]) | np.isclose(pnts, self.lim[:, 0])
        selected |= (pnts < self.lim[:, 1]) | np.isclose(pnts, self.lim[:, 1])
        return pnts[selected.all(axis=1)]

    @functools.cached_property
    def lim(self):
        """
        Return the rectangular limit.

        :return np.ndarray: left-bottom & right-upper rectangular vertices.
        """
        line = self.meshed[-1, :]
        dist = np.linalg.norm(line, axis=1).min()
        if dist > np.linalg.norm(self.meshed[:, -1], axis=1).min():
            line = self.meshed[:, -1]
        point = np.abs(line[np.abs(line.prod(axis=1)).argmax()])
        return np.array([-point, point]).T

    @functools.cached_property
    def meshed(self, num=6):
        """
        Return the meshed scaled lattice vectors.

        :param num int: the minimum number of duplicates along each lattice vec.
        :return np.ndarray: meshed points.
        """
        recip = [1. / x for x in self.options.miller_indices if x]
        num = math.ceil(max(*recip, *self.options.miller_indices, num))
        xi = range(-num, num + 1)
        meshed = np.stack(np.meshgrid(xi, xi, indexing='ij'), axis=-1)
        # The last dimension of meshed (coefficients) and dots with self.lat.T
        # [[x1, y1], [x2, y2]] as https://xaktly.com/DotProduct.html
        return np.dot(meshed, self.lat.T)

    def quiver(self, vec, fmt=r'$\vec {sym}^*$', color='b'):
        """
        Plot a quiver for the vector.

        :param vec 'pandas.core.series.Series': 2D vector.
        :param fmt 'str': the label text format.
        :param color 'r': the color of the quiver and annotation.
        """
        if not any(vec):
            return
        qv = self.ax.quiver(*self.origin,
                            *vec,
                            color=color,
                            angles='xy',
                            scale_units='xy',
                            scale=1,
                            units='dots',
                            width=3)
        self.qvs[qv] = self.ax.annotate(fmt.format(sym=vec.name),
                                        vec,
                                        color=color)

    def arrow(self, vec):
        """
        Annotate arrow for the vector.

        :param vec 'pandas.core.series.Series': 2D vector.
        """
        if not any(vec):
            return
        self.ax.annotate('',
                         vec,
                         xytext=self.origin,
                         arrowprops=dict(linestyle="-",
                                         arrowstyle="-|>",
                                         mutation_scale=10,
                                         color='r'))

    def legend(self):
        """
        Set the legend.
        """
        text_xys = [[x.get_text(), x.xy] for x in self.qvs.values()]
        texts = [f"{x} ({y[0]:.4g}, {y[1]:.4g})" for x, y in text_xys]
        self.ax.legend(self.qvs.keys(), texts, loc='upper right')


class Real(Recip):
    """
    Customized for the real space.
    """
    NAME = 'Real Space'

    def setUp(self):
        """
        See parent.
        """
        factors = [1. / x if x else 0 for x in self.options.miller_indices]
        self.scaled = factors * self.lat
        self.vec[:] = self.getNormal()

    def getNormal(self):
        """
        Get the plane normal. (intersection between the plane and plane normal
        that passes the origin)

        :return 1x3 'numpy.ndarray': the intersection point.
        """
        pnt, vec = self.getPlane()
        normal = np.dot([[0, 1], [-1, 0]], vec)  # normal to the plane
        # Interaction is on the normal: factor * normal
        # Interaction is on the plane: fac2 * vec + pnt1
        # Equation: normal * factor - vec * fac2 = pnt1
        return normal * np.linalg.solve(np.transpose([normal, -vec]), pnt)[0]

    @functools.cache
    def getPlane(self, factor=1):
        """
        Return the Miller plane. (two points in the plane)

        :param factor int: the Miller index plane is moved by this factor.
        :return 'np.ndarray': two points (rows) on the Miller plane.
        """
        vecs = self.scaled.values.T
        idx = (~vecs.any(axis=1)).nonzero()[0]
        point = vecs[vecs.any(axis=1).nonzero()[0][0]] if idx.size else vecs[0]
        vec = self.lat.values.T[idx[0]] if idx.size else (vecs[1] - vecs[0])
        return factor * point, vec

    def plot(self, *args):
        """
        See parent.
        """
        super().plot(*args)
        self.plane(-1)
        self.plane(0)
        self.plane(1)

    def quiver(self, *args, fmt=r'$\vec {sym}$', **kwargs):
        """
        See parent.
        """
        super().quiver(*args, fmt=fmt, **kwargs)

    def plane(self, factor):
        """
        Plot the Miller plane moved by the index factor.

        :param factor int: by this factor the Miller plane is moved.
        """
        # factor * (b_pnt - a_pnt) + a_pnt = lim
        pnt, vec = self.getPlane(factor=factor)
        facs = [(x - y) / z for x, y, z in zip(self.lim, pnt, vec) if z]
        # The intersection points are on x low, x high, y low, y high
        pnts = [x * vec + pnt for x in np.array(facs).flatten()]
        pnts = self.crop(np.array(pnts))[:2, :]
        self.ax.plot(pnts[:, 0],
                     pnts[:, 1],
                     color='lightgray',
                     linestyle='--',
                     alpha=0.5)


class RecipSp(logutils.Base):
    """
    Set and plot 2D lattice vectors.

    https://www.youtube.com/watch?v=cdN6OgwH8Bg
    https://en.wikipedia.org/wiki/Recip_lattice
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver':  Parsed command-line options
        """
        super().__init__(options=options, **kwargs)
        self.real = None
        self.recip = None
        self.outfile = f'{self.options.JOBNAME}.png'
        jobutils.Job.reg(self.outfile, file=True)

    def run(self):
        """
        Main method to run.
        """
        self.setReal()
        self.setRecip()
        self.product()
        self.plot()

    def setReal(self):
        """
        Set real space lattice vector.

        Characteristic length of a hexagon is sqrt(3) x the edge length
        https://physics.stackexchange.com/questions/664945/integration-over-first-brillouin-zone
        """
        # Primitive lattice vector of graphene
        vecs = [[3 / 2., 0.5 * math.sqrt(3)], [3 / 2., -0.5 * math.sqrt(3)]]
        vecs = np.transpose(vecs)
        self.real = Real(vecs, options=self.options, logger=self.logger)

    def setRecip(self):
        """
        Set the reciprocal lattice vectors.

        https://en.wikipedia.org/wiki/Recip_lattice (Two dimensions)
        """
        normal = np.dot([[0, -1], [1, 0]], self.real.lat)[:, ::-1]
        recip = 2 * np.pi * normal
        recip /= np.multiply(self.real.lat, normal).sum(axis=0).values
        self.recip = Recip(recip, options=self.options, logger=self.logger)

    def product(self):
        """
        Log cross and doc product.
        """
        cross = np.cross(self.real.vec, self.recip.vec)
        dot = np.dot(self.real.vec, self.recip.vec) / np.pi
        if np.isclose(cross, 0):
            self.log(f"The real and reciprocal vectors are parallel to each "
                     f"other with {dot:.4g}π being the dot product.")
            return
        self.log(f"The cross product: {cross}; The dot product: {dot}π")

    def plot(self):
        """
        Plot the real and reciprocal paces.
        """
        with plotutils.pyplot(inav=self.options.INTERAC) as plt:
            fig = plt.figure(figsize=(15, 9))
            self.real.plot(fig.add_subplot(1, 2, 1))
            self.recip.plot(fig.add_subplot(1, 2, 2))
            fig.suptitle(f'Miller indices {self.options.miller_indices}')
            fig.tight_layout()
            fig.savefig(self.outfile)
            self.log(f'Figure saved as {self.outfile}')


class MillerAction(parserutils.Action):
    """
    Action on miller indices.
    """

    def doTyping(self, *args):
        """
        See parent.
        """
        if not any(args):
            self.error(f'Miller indices cannot be all zeros.')
        return args


class Parser(parserutils.Driver):
    """
    The argument parser.
    """
    FLAG_MILLER_INDICES = '-miller_indices'

    def setUp(self):
        """
        The user-friendly command-line parser.
        """
        self.add_argument(self.FLAG_MILLER_INDICES,
                          metavar=self.FLAG_MILLER_INDICES[1:].upper(),
                          type=parserutils.type_float,
                          nargs=2,
                          action=MillerAction,
                          default=(0.5, 2),
                          help='The Miller indices.')


if __name__ == "__main__":
    parser = Parser(descr=__doc__)
    options = parser.parse_args(sys.argv[1:])
    with logutils.Script(options) as logger:
        recip_sp = RecipSp(options, logger=logger)
        recip_sp.run()
