# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg

from pysages.colvars.core import TwoPointCV
from pysages.colvars.coordinates import barycenter


class DistanceComponents(TwoPointCV):
    """
    Use the distance components along x, y, and z of atom groups 
    selected via the indices as collective variable.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices. (2 Groups required)
    """

    @property
    def function(self):
        if len(self.groups) == 0:
            return distance
        return lambda r1, r2: distance(barycenter(r1), barycenter(r2))


def distancecomponents(r1, r2):
    """
    Returns the distance between two points in space or
    between the barycenters of two groups of points in space.

    Parameters
    ----------
    r1: DeviceArray
        Array containing the position in space of the first point or group of points.
    r2: DeviceArray
        Array containing the position in space of the second point or group of points.

    Returns
    -------
    distance: float
        Distance components along each axis between the two points.
    """
    
    return r1 - r2
