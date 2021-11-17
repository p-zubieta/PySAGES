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
    SpectralSobolev1Fit,
    build_fitter,
    build_evaluator,
    build_grad_evaluator,
    compute_mesh,
)
from pysages.grids import Chebyshev, Grid, build_indexer, convert
from pysages.methods.core import GriddedSamplingMethod, generalize
from pysages.utils import Bool, Int, JaxArray


class CFFSpectralState(NamedTuple):
    bias:  JaxArray
    hist:  JaxArray
    histp: JaxArray
    A:     JaxArray
    prob:  JaxArray
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
    ind:  Tuple
    fun:  Fun
    pred: Bool


class CFFSpectral(GriddedSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        if "kT" not in self.kwargs:
            raise ValueError("The value of kT must be provided")

        self.kT = self.kwargs["kT"]
        self.N = np.asarray(self.kwargs.get('N', 100))
        self.k = self.kwargs.get("k", None)
        self.fit_freq = self.kwargs.get("fit_freq", 500)
        self.grid = (
            self.grid if self.grid.is_periodic else
            convert(self.grid, Grid[Chebyshev])
        )
        self.external_force = self.kwargs.get("external_force", lambda rs: 0)

        return _fspectral(self, snapshot, helpers)


def _fspectral(method, snapshot, helpers):
    # N = method.N
    kT = method.kT
    cv = method.cv
    grid = method.grid
    fit_freq = method.fit_freq
    external_force = method.external_force

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    gshape = grid.shape if dims > 1 else (*grid.shape, 1)
    get_grid_index = build_indexer(grid)
    model = SpectralSobolev1Fit(grid)
    method.model = model
    # fmodel = SpectralGradientFit(grid)
    fit = build_fitter(model)
    # ffit = build_fitter(fmodel)
    evaluate = build_evaluator(model)
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    fit_force_and_potential = partial(
        _fit_force_and_potential, kT, fit, evaluate, inputs
    )
    estimate_force = build_force_estimator(method)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(gshape, dtype=np.uint32)
        histp = np.zeros(gshape, dtype = np.uint32)
        A = np.zeros(gshape)
        prob = np.zeros(gshape)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(A, Fsum)
        xi, _ = cv(helpers.query(snapshot))
        return CFFSpectralState(bias, hist, histp, A, prob, Fsum, F, Wp, Wp_, xi, fun, 1)

    def update(state, data):
        # During the intial stage use ABF
        nstep = state.nstep
        use_abf = nstep <= fit_freq
        #
        histp, A, prob, fun = cond(
            ~use_abf & (nstep % fit_freq == 1),
            fit_force_and_potential,
            lambda state: (state.histp, state.A, state.prob, state.fun),
            state
        )
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        p = data.momenta
        Wp = solve(Jx @ Jx.T, Jx @ p, sym_pos = "sym")
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_x = get_grid_index(x)
        hist = state.hist.at[I_x].add(1)
        Fsum = state.Fsum.at[I_x].add(dWp_dt + state.F)
        histp = histp.at[I_x].add(1)
        #
        F = estimate_force(PartialState(hist, Fsum, x, I_x, fun, use_abf))
        bias = (-Jx.T @ F).reshape(state.bias.shape)
        bias = bias + external_force(data)
        #
        return CFFSpectralState(
            bias, hist, histp, A, prob, Fsum, F, Wp, state.Wp, x, fun, nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers)


def _fit_force_and_potential(kT, fit, evaluate, inputs, state):
    shape = (*drop_last(state.Fsum.shape), 1)
    prob = state.prob + state.histp * np.exp(state.A / kT)
    A = kT * np.log(np.maximum(1, prob))
    F = state.Fsum / np.maximum(1, state.hist.reshape(shape))
    #
    # Reset the network parameters before training
    fun = fit(A, F)
    #
    A = evaluate(fun, inputs)
    A = (A - A.min()).reshape(state.A.shape)
    #
    histp = np.zeros_like(state.histp)
    #
    return histp, A, state.prob, fun


@dispatch
def build_force_estimator(method: CFFSpectral):
    k = method.k
    N = method.N
    grid = method.grid
    model = method.model
    get_grad = build_grad_evaluator(model)

    def interpolate_force(fun, x, F):
        return get_grad(fun, x).reshape(F.shape)

    def _estimate_force(state):
        i = state.ind
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


def _apply_restraints(lo, up, klo, kup, xi):
    return np.where(xi < lo, klo * (xi - lo), np.where(xi > up, kup * (xi - up), 0))


def drop_last(collection):
    return collection[:-1]
