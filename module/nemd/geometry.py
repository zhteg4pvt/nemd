# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Geometry calculation of 3D points, such as distance, angle, and dihedral angle.
"""
import numpy as np


class Point(np.ndarray):
    """
    Class to perform geometry calculations on points.
    """

    def __new__(cls, array):
        """
        :param array np.ndarray: each row is a point.
        """
        return np.asarray(array).view(cls)

    def distance(self):
        """
        Calculate distance between two points: The coordinates of two points.

        :return float: The distance between two points.
        """
        return np.linalg.norm(self[0] - self[1])

    def angle(self):
        """
        Calculate angle formed by three points: point 0-1-2 forms the angle.

        :return float: The angle formed by the three points in degrees.
        """
        end_points = self[::2, :]
        return self.angle_vs(end_points - self[1])

    @staticmethod
    def angle_vs(vecs):
        """
        Calculate angle between two vectors.

        :param vecs list of two ndarrays: The two vectors forms the angle.
        :return float: The angle between two vectors in degrees.
        """
        product = np.dot(vecs[0], vecs[1])
        cos = product / np.linalg.norm(vecs[0]) / np.linalg.norm(vecs[1])
        return np.arccos(cos) / np.pi * 180.

    def dihedral(self):
        """
        Calculate dihedral angle formed by four points: points 0-1-2-3 forms the
        dihedral angle.

        :return float: The dihedral angle formed by the four points in degrees.
        """
        bonds = self[:3, :] - self[1:, :]
        return self.angle_vs(
            [np.cross(*bonds[:2, :]),
             np.cross(*bonds[1:, :])])
