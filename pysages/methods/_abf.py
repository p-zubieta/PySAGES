# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax.lax import cond
from jax.scipy.linalg import solve
from plum import dispatch
from typing import NamedTuple

from pysages.grids import Grid, build_indexer
from pysages.methods.core import GriddedSamplingMethod, generalize
from pysages.utils import JaxArray

import jax.numpy as np


# ======= #
#   ABF   #
# ======= #


class ABFState(NamedTuple):
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    F:    JaxArray
    Wp:   JaxArray
    Wp_:  JaxArray
    xi:   JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ABF(GriddedSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get("N", 200))
        self.k = self.kwargs.get("k", None)
        return _abf(self, snapshot, helpers)


def _abf(method, snapshot, helpers):
    grid = method.grid
    cv = method.cv
    k = method.k

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    estimate_force = build_force_estimator(method, grid, k)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        x, _ = cv(helpers.query(snapshot))
        return ABFState(bias, hist, Fsum, F, Wp, Wp_, x)

    def update(state, data):
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        # The following could equivalently be computed as `linalg.pinv(Jξ.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        Wp = solve(Jxi @ Jxi.T, Jxi @ p, sym_pos = "sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.F)
        F_xi = Fsum[I_xi]
        N_xi = hist[I_xi]
        # F = F_xi / np.maximum(N_xi, N)
        #
        F = estimate_force(xi, I_xi, F_xi, N_xi).reshape(dims)
        bias = np.reshape(-Jxi.T @ F, state.bias.shape)
        #
        return ABFState(bias, hist, Fsum, F, Wp, state.Wp, xi)

    return snapshot, initialize, generalize(update, helpers)


def _apply_restraints(lo, up, klo, kup, xi):
    return np.where(xi < lo, klo * (xi - lo), np.where(xi > up, kup * (xi - up), 0))


@dispatch
def build_force_estimator(method: ABF, grid: Grid, k):
    N = method.N

    def _estimate_force(xi, I_xi, F_xi, N_xi):
        return F_xi / np.maximum(N, N_xi)

    if k is None:
        print("No restraints!")
        estimate_force = _estimate_force
    else:
        lo = grid.lower  # - grid.size * 0.01
        up = grid.upper  # + grid.size * 0.01
        klo = kup = k

        def apply_restraints(data):
            xi, *_ = data
            xi = xi.reshape(grid.shape.size)
            return _apply_restraints(lo, up, klo, kup, xi)

        def estimate_force(xi, I_xi, F_xi, N_xi):
            return cond(
                np.any(np.array(I_xi) == grid.shape),
                apply_restraints,
                lambda args: _estimate_force(*args),
                (xi, I_xi, F_xi, N_xi),
            )

    return estimate_force
