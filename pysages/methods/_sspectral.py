# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from jax.lax import cond
from jax.scipy.linalg import solve
from pysages.approxfun import (
    Fun,
    # SpectralGradientFit,
    SpectralSobolev1Fit,
    build_fitter,
    build_evaluator,
    build_grad_evaluator,
    compute_mesh,
)
from pysages.ssages.grids import build_indexer
from pysages.ssages.methods._funn import _estimate_abf
from pysages.utils import Int, JaxArray
from typing import NamedTuple

from pysages.methods.core import GriddedSamplingMethod, generalize

import jax.numpy as np


# ===================== #
#   Spectral Sampling   #
# ===================== #

class CFFSpectralState(NamedTuple):
    bias:   JaxArray
    hist:   JaxArray
    phist:  JaxArray
    phi:    JaxArray
    prob:   JaxArray
    Fsum:   JaxArray
    F:      JaxArray
    Wp:     JaxArray
    Wp_:    JaxArray
    fun:    Fun
    nstep:  Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class CFFSpectral(GriddedSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def __call__(self, snapshot, helpers):
        kT = self.kwargs.get("kT", 1.0)
        N = np.asarray(self.kwargs.get('N', 100))
        fit_freq = self.kwargs.get("fit_freq", 500)
        return _fspectral(snapshot, self.cv, self.grid, N, kT, fit_freq, helpers)


def _fspectral(snapshot, cv, grid, N, kT, fit_freq, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    gshape = grid.shape if dims > 1 else (*grid.shape, 1)
    get_grid_index = build_indexer(grid)
    model = SpectralSobolev1Fit(grid)
    # fmodel = SpectralGradientFit(grid)
    fit = build_fitter(model)
    # ffit = build_fitter(fmodel)
    evaluate = build_evaluator(model, grid)
    get_grad = build_grad_evaluator(model, grid)
    fit_force_and_potential = partial(
        _fit_force_and_potential, kT, fit, evaluate, compute_mesh(grid)
    )
    estimate_abf = partial(_estimate_abf, N)
    estimate_force = partial(_estimate_force, get_grad)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.ones(gshape, dtype=np.uint32)
        phist = np.ones(gshape, dtype = np.uint32)
        phi = np.zeros(gshape)
        prob = np.zeros(gshape)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(phi, Fsum)
        return CFFSpectralState(bias, hist, phist, phi, prob, Fsum, F, Wp, Wp_, fun, 1)

    def update(state, data):
        # During the intial stage use ABF
        nstep = state.nstep
        use_abf = nstep <= fit_freq
        # use_cff = nstep > fit_freq
        #
        # Fit forces
        # fun = cond(
        #     (nstep % N == 1) & ~use_abf & ~use_cff,
        #     lambda state: ffit(average_forces(state)),
        #     lambda state: state.fun,
        #     state
        # )
        phist, phi, prob, fun = cond(
            ~use_abf & (nstep % fit_freq == 1),
            fit_force_and_potential,
            lambda state: (state.phist, state.phi, state.prob, state.fun),
            state
        )
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        p = data.momenta
        Wp = solve(Jx @ Jx.T, Jx @ p, sym_pos = "sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_x = get_grid_index(x)
        N_x = state.hist[I_x] + 1
        F_x = state.Fsum[I_x] + dWp_dt + state.F
        hist = state.hist.at[I_x].set(N_x)
        Fsum = state.Fsum.at[I_x].set(F_x)
        phist = phist.at[I_x].add(1)
        #
        F = cond(use_abf, estimate_abf, estimate_force, (fun, x, F_x, N_x))
        bias = np.reshape(-Jx.T @ F, state.bias.shape)
        #
        return CFFSpectralState(
            bias, hist, phist, phi, prob, Fsum, F, Wp, state.Wp, fun, nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers)


def _fit_force_and_potential(kT, fit, evaluate, inputs, state):
    # phi = evaluate(state.fun, inputs)
    phi = state.phi
    prob = state.prob + state.phist * np.exp(phi / kT)
    phi = kT * np.log(prob)
    phi = phi - phi.min()
    #
    F = average_forces(state)
    #
    # Reset the network parameters before training
    fun = fit(phi, F)
    #
    phi = evaluate(fun, inputs)
    phi = (phi - phi.min()).reshape(state.phi.shape)
    #
    phist = np.zeros_like(state.phist)
    #
    return phist, phi, prob, fun


def average_forces(state):
    Fsum = state.Fsum
    shape = (*drop_last(Fsum.shape), 1)
    return Fsum / state.hist.reshape(shape)


def _estimate_force(get_grad, data_bundle):
    fun, x, *_ = data_bundle
    return get_grad(fun, x).flatten()


def drop_last(collection):
    return collection[:-1]
