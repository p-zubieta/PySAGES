# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from jax import scipy
from pysages.ssages.grids import build_indexer
from pysages.approxfun import (
    VandermondeGradientFit,
    build_fitter,
    build_interpolator,
)

from .core import GriddedSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np


# ===================== #
#   Spectral Sampling   #
# ===================== #

class ForceSpectrumState(namedtuple(
    "ForceSpectrumState",
    (
        "fun",
        "bias",
        "hist",
        "Fsum",
        "F",
        "Wp",
        "Wp_",
    ),
)):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ForceSpectrum(GriddedSamplingMethod):
    def __call__(self, snapshot, helpers):
        N = np.asarray(self.kwargs.get('N', 200))
        return _spectral(snapshot, self.cv, self.grid, N, helpers)


def _spectral(snapshot, cv, grid, N, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    indices, momenta = helpers
    model = VandermondeGradientFit(grid)
    fit = build_fitter(model)
    interpolate = build_interpolator(model, grid)

    def average_forces(state):
        Fsum = state.Fsum.reshape(-1, dims)
        hist = np.maximum(state.hist.reshape(-1, 1), 1)
        return Fsum / hist

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        return ForceSpectrumState(fun, bias, hist, Fsum, F, Wp, Wp_)

    def update(state, rs, vms, ids):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(rs, indices(ids))
        #
        p = momenta(vms)
        Wp = scipy.linalg.solve(Jξ @ Jξ.T, Jξ @ p, sym_pos="sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_ξ = get_grid_index(ξ)
        N_ξ = state.hist[I_ξ] + 1
        # Add previous force to remove bias
        F_ξ = state.Fsum[I_ξ] + dWp_dt + state.F
        hist = state.hist.at[I_ξ].set(N_ξ)
        Fsum = state.Fsum.at[I_ξ].set(F_ξ)
        fun = fit(average_forces(state))
        F = interpolate(fun, ξ)  # F_ξ / np.maximum(N_ξ, N)
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return ForceSpectrumState(fun, bias, hist, Fsum, F, Wp, state.Wp)

    return snapshot, initialize, generalize(update)
