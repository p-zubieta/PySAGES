# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from typing import NamedTuple, Tuple

from jax import numpy as np
from jax.lax import cond
from jax.scipy.linalg import solve
from plum import dispatch

from pysages.approxfun import (
    Fun,
    SpectralGradientFit,
    build_fitter,
    build_grad_evaluator,
)
from pysages.grids import Chebyshev, Grid, build_indexer, convert
from pysages.methods.core import GriddedSamplingMethod, generalize
from pysages.utils import Bool, Int, JaxArray


class ForceSpectrumState(NamedTuple):
    bias:  JaxArray
    hist:  JaxArray
    Fsum:  JaxArray
    F:     JaxArray
    Wp:    JaxArray
    Wp_:   JaxArray
    xi:    JaxArray
    fun:   Fun
    nstep: Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialState(NamedTuple):
    hist: JaxArray
    Fsum: JaxArray
    xi:   JaxArray
    I:    Tuple
    fun:  Fun
    pred: Bool


class ForceSpectrum(GriddedSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get('N', 200))
        self.k = self.kwargs.get('k', None)
        self.fit_freq = self.kwargs.get("fit_freq", 100)
        self.fit_threshold = self.kwargs.get("fit_threshold", 500)
        self.grid = (
            self.grid if self.grid.is_periodic else
            convert(self.grid, Grid[Chebyshev])
        )
        self.model = SpectralGradientFit(self.grid)
        self.external_force = self.kwargs.get("external_force", lambda rs: 0)
        return _fspectral(self, snapshot, helpers)


def _fspectral(method, snapshot, helpers):
    grid = method.grid
    cv = method.cv
    N = method.N
    model = method.model
    fit_freq = method.fit_freq
    fit_threshold = method.fit_threshold
    external_force = method.external_force

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    average_forces = partial(_average_forces, N)
    fit = build_fitter(model)
    estimate_force = build_force_estimator(method)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        xi, _ = cv(helpers.query(snapshot))
        return ForceSpectrumState(bias, hist, Fsum, F, Wp, Wp_, xi, fun, 1)

    def update(state, data):
        # During the intial stage use ABF
        nstep = state.hist.sum()
        use_abf = nstep <= fit_threshold
        #
        # Fit forces
        fun = cond(
            (nstep % fit_freq) == 1 & ~use_abf,
            lambda state: fit(average_forces(state)),
            lambda state: state.fun,
            state
        )
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = solve(Jxi @ Jxi.T, Jxi @ p, sym_pos = "sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.F)
        #
        F = estimate_force(PartialState(hist, Fsum, xi, I_xi, fun, use_abf))
        bias = np.reshape(-Jxi.T @ F, state.bias.shape)
        bias = bias + external_force(data)
        #
        return ForceSpectrumState(
            bias, hist, Fsum, F, Wp, state.Wp, xi, fun, state.nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers)


def _average_forces(N, state):
    Fsum = state.Fsum
    shape = (*drop_last(Fsum.shape), 1)
    return Fsum / np.maximum(state.hist.reshape(shape), 1)


def _apply_restraints(lo, up, klo, kup, xi):
    return np.where(xi < lo, klo * (xi - lo), np.where(xi > up, kup * (xi - up), 0))


def drop_last(collection):
    return collection[:-1]


def last(collection, n = 0):
    return collection[-1 + n]


@dispatch
def build_force_estimator(method: ForceSpectrum):
    k = method.k
    N = method.N
    grid = method.grid
    model = method.model
    get_grad = build_grad_evaluator(model)

    def interpolate_force(fun, xi, F):
        return get_grad(fun, xi).reshape(F.shape)

    def _estimate_force(state):
        i = state.I
        F = state.Fsum[i] / np.maximum(N, state.hist[i])
        return cond(
            state.pred,
            lambda args: args[-1],
            lambda args: interpolate_force(*args),
            (state.fun, state.xi, F)
        )

    if k is None:
        estimate_force = _estimate_force
    else:
        lo = grid.lower
        up = grid.upper
        klo = kup = k

        def apply_restraints(state):
            xi = state.xi.reshape(grid.shape.size)
            return _apply_restraints(lo, up, klo, kup, xi)

        def estimate_force(state):
            return cond(
                np.any(np.array(state.I) == grid.shape),
                apply_restraints,
                _estimate_force,
                state
            )

    return estimate_force
