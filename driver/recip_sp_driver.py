# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Calculates and visualizes 2D lattice in the real and reciprocal spaces.
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
    This class is used to plot the reciprocal space on axis.
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
        self.origin = np.array([0., 0.])
        self.scaled = self.lat * self.options.miller_indices
        self.vec = pd.Series(self.scaled.sum(axis=1), name='r')
        self.setUp()

    def setUp(self):
        """
        Set up.
        """
        self.log(f"The {self.NAME.lower()} vector {self.vec.values} has a "
                 f"norm of {np.linalg.norm(self.vec):.4g}")

    def plot(self, ax):
        """
        Plot the grids and vectors.
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
        Plot the selected grids.
        """
        self.ax.scatter(self.grids[:, 0],
                        self.grids[:, 1],
                        marker='o',
                        alpha=0.5)
        self.ax.set_xlim([-self.lim[0], self.lim[0]])
        self.ax.set_ylim([-self.lim[1], self.lim[1]])
        self.ax.set_aspect('equal')
        self.ax.set_title(self.NAME)

    @functools.cached_property
    def grids(self):
        """
        Crop the meshed points with a rectangular.

        :param np.ndarray: grid points.
        """
        return self.crop(self.meshed.reshape(-1, 2))

    def crop(self, points):
        """
        Crop the points with a rectangular.

        :param np.ndarray: cropped points.
        """
        abs = np.abs(points)
        close_or_within = (abs < self.lim) | np.isclose(abs, self.lim)
        return points[close_or_within.all(axis=1)]

    @functools.cached_property
    def lim(self):
        """
        Return the rectangular vertice in quadrant I.

        :param np.ndarray: the rectangular vertice.
        """
        line = self.meshed[-1, :]
        dist = np.linalg.norm(line, axis=1).min()
        if dist > np.linalg.norm(self.meshed[:, -1], axis=1).min():
            line = self.meshed[:, -1]
        return np.abs(line[np.abs(line.prod(axis=1)).argmax()])

    @functools.cached_property
    def meshed(self, num=6):
        """
        Scale the mesh lattice vectors.

        :param num int: the minimum number of duplicates along each lattice vec.
        :param np.ndarray: meshed points.
        """
        recip = [1. / x for x in self.options.miller_indices if x]
        num = math.ceil(max(*recip, *self.options.miller_indices, num))
        xi = range(-num, num + 1)
        meshed = np.stack(np.meshgrid(xi, xi, indexing='ij'), axis=-1)
        # The last dimension of meshed (coefficients) and dots with self.lat.T
        # [[x1, y1], [x2, y2]] as https://xaktly.com/DotProduct.html
        return np.dot(meshed, self.lat.T)

    def quiver(self,
               vec,
               fmt=r'$\vec {sym}^*$',
               color='b',
               angles='xy',
               scale_units='xy',
               scale=1,
               units='dots',
               width=3):
        """
        Plot a quiver for the vector. (see matplotlib.pyplot.quiver)

        :param vec 'pandas.core.series.Series': two points as a vector
        """
        if not any(vec):
            return
        qv = self.ax.quiver(*self.origin,
                            *vec,
                            color=color,
                            angles=angles,
                            scale_units=scale_units,
                            scale=scale,
                            units=units,
                            width=width)
        text = fmt.format(sym=vec.name)
        self.ax.annotate(text, vec, color=color)
        self.qvs[qv] = f"{text} ({vec.iloc[0]:.4g}, {vec.iloc[1]:.4g})"

    def arrow(self, vec, arrowprops=None):
        """
        Annotate arrow for the vector. (see matplotlib.pyplot.annotate)

        :param vec list: list of two points
        """
        if arrowprops is None:
            arrowprops = dict(linestyle="-",
                              arrowstyle="-|>",
                              mutation_scale=10,
                              color='r')
        if not any(vec):
            return
        self.ax.annotate('', vec, xytext=self.origin, arrowprops=arrowprops)

    def legend(self):
        """
        Set the legend.
        """
        self.ax.legend(self.qvs.keys(), self.qvs.values(), loc='upper right')


class Real(Recip):
    """
    This class is used to plot the real space on axis.
    """
    NAME = 'Real Space'

    def setUp(self):
        miller = [1. / x if x else 0 for x in self.options.miller_indices]
        self.scaled = self.lat * miller
        self.vec[:] = self.getNormal(factor=1)
        super().setUp()

    def quiver(self, *args, fmt=r'$\vec {sym}$', **kwargs):
        """
        See parent.
        """
        super().quiver(*args, fmt=fmt, **kwargs)

    def getNormal(self, factor=1):
        """
        Get the intersection between the plane and its normal passing origin.

        :param factor int: by this factor the Miller plane is moved.
        :return 1x3 'numpy.ndarray': the intersection point
        """
        pnt1, vec = self.getPlane(factor=factor)
        normal = np.dot([[0, 1], [-1, 0]], vec)  # normal to the plane
        # Interaction is on the normal: factor * normal
        # Interaction is on the plane: fac2 * vec + pnt1
        # Equation: normal * factor - vec * fac2 = pnt1
        factor, _ = np.linalg.solve(np.transpose([normal, -vec]), pnt1)
        return normal * factor

    @functools.cache
    def getPlane(self, factor=1):
        """
        Return two points in the Miller plane.

        :param factor int: the Miller index plane is moved by this factor.
        :return 'np.ndarray': two points (rows) on the Miller
        """
        nonzero = self.scaled.any()
        if factor and nonzero.all():
            a_pnt, b_pnt = self.scaled.T.values * factor
            return a_pnt, b_pnt - a_pnt

        index = nonzero.to_list().index(True)
        pnt = self.scaled.iloc[:, index] if factor else self.origin
        if nonzero.all():
            # Vectors are available, and the subtraction defines the direction
            vec = self.scaled.a2 - self.scaled.a1
        else:
            # If one vector, the other's lattice vector defines the direction
            vec = self.lat.iloc[:, nonzero.to_list().index(False)]
        if factor:
            pnt *= factor
            vec *= factor
        return pnt, vec

    def plot(self, *args):
        """
        See parent.
        """
        super().plot(*args)
        self.plotPlane(factor=-1)
        self.plotPlane(factor=0)
        self.plotPlane(factor=1)

    def plotPlane(self, factor=1, color='0.8', linestyle='--', alpha=0.5):
        """
        Plot the Miller plane moved by the index factor.

        :param factor int: by this factor the Miller plane is moved.
        """
        # factor * (b_pnt - a_pnt) + a_pnt = lim
        lim = [np.array([-x, x]) for x in self.lim]
        a_pnt, vec = self.getPlane(factor=factor)
        facs = np.array([(x - y) / z for x, y, z in zip(lim, a_pnt, vec) if z])
        pnts = np.array([x * vec + a_pnt for x in facs.flatten()])
        if pnts.shape[0] == 4:
            # The intersection points are on x low, x high, y low, y high
            pnts = self.crop(pnts)[:2, :]
        x_vals, y_vals = pnts.T
        if np.isclose(*x_vals):
            # The plane is vertical
            self.ax.vlines(x_vals.mean(),
                           *y_vals,
                           color=color,
                           linestyle=linestyle,
                           alpha=alpha)
            return
        self.ax.plot(x_vals,
                     y_vals,
                     color=color,
                     linestyle=linestyle,
                     alpha=alpha)


class RecipSp(logutils.Base):
    """
    Class to set and plot the reciprocal space lattice vectors for 2D graphene.

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
        Define real space lattice vector with respect to the origin.

        Characteristic length of a hexagon is sqrt(3) x the edge length
        https://physics.stackexchange.com/questions/664945/integration-over-first-brillouin-zone
        """
        # Primitive lattice vector of graphene
        data = np.array([[math.sqrt(3) / 2., 0.5], [math.sqrt(3) / 2., -0.5]])
        data *= math.sqrt(3)
        self.real = Real(data.T, options=self.options, logger=self.logger)

    def setRecip(self):
        """
        Set the reciprocal lattice vectors based on the real ones.

        https://en.wikipedia.org/wiki/Recip_lattice (Two dimensions)
        """
        ab_norm = np.dot([[0, -1], [1, 0]], self.real.lat)
        ba_norm = ab_norm[:, ::-1]
        column_dot = np.multiply(self.real.lat, ba_norm).sum(axis=0).tolist()
        recip = 2 * np.pi * ba_norm / column_dot
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
        else:
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
            jobutils.Job.reg(self.outfile, file=True)


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
                          help='Plot the planes of this Miller indices.')


if __name__ == "__main__":
    parser = Parser(descr=__doc__)
    options = parser.parse_args(sys.argv[1:])
    with logutils.Script(options) as logger:
        recip_sp = RecipSp(options, logger=logger)
        recip_sp.run()
