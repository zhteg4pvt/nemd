# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Geometry calculation of 3D points, such as distance, angle, and dihedral angle.
"""
import numpy as np


def distance(pnts):
    """
    Calculate distance between two points.

    :param pnts list of two ndarrays: The coordinates of two points.
    :return float: The distance between two points.
    """
    return np.linalg.norm(pnts[0] - pnts[1])


def angle_vs(vecs):
    """
    Calculate angle between two vectors.

    :param vecs list of two ndarrays: The two vectors forms the angle.
    :return float: The angle between two vectors in degrees.
    """
    product = np.dot(vecs[0], vecs[1])
    cos = product / np.linalg.norm(vecs[0]) / np.linalg.norm(vecs[1])
    return np.arccos(cos) / np.pi * 180.


def angle(pnts):
    """
    Calculate angle formed by three points.

    :param pnts list of three ndarrays: point 0-1-2 forms the angle.
    :return float: The angle formed by the three points in degrees.
    """
    end_points = pnts[::2, :]
    return angle_vs(end_points - pnts[1])


def dihedral(pnts):
    """
    Calculate dihedral angle formed by four points.

    :param pnts list of four ndarrays: points 0-1-2-3 forms the dihedral angle.
    :return float: The dihedral angle formed by the four points in degrees.
    """
    bonds = pnts[:3, :] - pnts[1:, :]
    return angle_vs([np.cross(*bonds[:2, :]), np.cross(*bonds[1:, :])])
