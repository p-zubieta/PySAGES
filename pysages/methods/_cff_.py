# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from jax import grad, vmap
from jax.lax import cond
from jax.numpy import linalg
# from jax.scipy.signal import convolve
from typing import NamedTuple
from pysages.ml.models import MLP, Siren
from pysages.ml.objectives import Sobolev1SSE
from pysages.ml.optimizers import (
    LevenbergMarquardt,
    # LevenbergMarquardtBR,
    # update_hyperparams,
)
from pysages.ml.training import NNData, build_fitting_function, normalize, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.approxfun import compute_mesh
from pysages.grids import build_indexer
# from pysages.methods._funn import _estimate_abf
from pysages.utils import Int, JaxArray

from .core import NNSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np


# ======== #
#   FUNN   #
# ======== #

class CFFState(NamedTuple):
    bias:   JaxArray
    hist:   JaxArray
    phist:  JaxArray
    phi:    JaxArray
    prob:   JaxArray
    Fsum:   JaxArray
    F:      JaxArray
    Wp:     JaxArray
    Wp_:    JaxArray
    nn:     NNData
    fnn:    NNData
    nstep:  Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class CFF(NNSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        if "kT" not in self.kwargs:
            raise ValueError("The value of kT must be provided")

        self.kT = self.kwargs["kT"]
        self.N = np.asarray(self.kwargs.get("N", 100))
        self.k = self.kwargs.get("k", None)
        self.train_freq = self.kwargs.get("train_freq", 5000)
        model = self.kwargs.get("model", MLP)
        model_kwargs = self.kwargs.get("model_kwargs", dict())

        def build_model(ins, outs, topology, **kwargs):
            return model(ins, outs, topology, **model_kwargs, **kwargs)

        self.build_model = build_model
        # reg = VarRegularization() if model is Siren else L2Regularization(0.0)
        max_iters = 250 if model is Siren else 500
        self.optimizer = LevenbergMarquardt(
            loss = Sobolev1SSE(), max_iters = max_iters,  # reg = reg
        )

        return _cff(self, snapshot, helpers)


def _cff(method: CFF, snapshot, helpers):
    N = method.N
    cv = method.cv
    kT = method.kT
    grid = method.grid
    train_freq = method.train_freq

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    gshape = grid.shape if dims > 1 else (*grid.shape, 1)
    # Neural network and optimizer
    model = method.build_model(dims, 1, method.topology)
    ps, layout = unpack(model.parameters)
    # foptimizer = LevenbergMarquardt(loss = GradientsSSE())
    fit = build_fitting_function(model, method.optimizer)
    # ffit = build_fitting_function(model, foptimizer)
    # Training data
    inputs = compute_mesh(grid)
    # Helper methods
    get_grid_index = build_indexer(grid)
    # estimate_abf = jit(partial(_estimate_abf, N))
    smooth = partial(
        convolve,
        kernel = blackman_kernel(dims, 7),
        boundary = "wrap" if grid.is_periodic else "edge"
    )
    shift_and_scale = normalize
    train_e = partial(
        _train,
        fit, shift_and_scale,
        smooth if dims > 1 else lambda y: vmap(smooth)(y.T).T,
    )
    train_f = partial(
        _train,
        fit, shift_and_scale,
        lambda y: vmap(smooth)(y.T).T
    )
    learn_potential = partial(
        _learn_potential, kT, (*grid.shape, 1), train_e, model.apply, layout, inputs
    )
    learn_forces = partial(
        _learn_forces, N, (*grid.shape, 1), train_f, model.apply, layout, inputs
    )
    estimate_force = partial(
        _estimate_force, N, lambda params, x: model.apply(params, x).sum(), layout
    )

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(gshape, dtype = np.uint32)
        phist = np.zeros(gshape, dtype = np.uint32)
        phi = np.zeros(gshape)
        prob = np.zeros(gshape)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, np.array(0.0), np.array(1.0))
        fnn = NNData(ps, np.zeros(dims), np.ones(dims))
        return CFFState(bias, hist, phist, phi, prob, Fsum, F, Wp, Wp_, nn, fnn, 1)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        nstep = state.nstep
        use_abf = nstep <= 2 * train_freq
        #
        # Estimate free energy / train NN
        phist, phi, prob, nn = cond(
            (nstep % train_freq == 1) & ~use_abf,
            learn_potential,
            lambda state: (state.phist, state.phi, state.prob, state.nn),
            state
        )
        fnn = cond(
            (state.nstep % train_freq == 1) & ~use_abf,
            learn_forces,
            lambda state: state.fnn,
            state
        )
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        p = data.momenta
        Wp = linalg.tensorsolve(Jx @ Jx.T, Jx @ p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_x = get_grid_index(x)
        N_x = state.hist[I_x] + 1
        F_x = state.Fsum[I_x] + dWp_dt + state.F
        hist = state.hist.at[I_x].set(N_x)
        Fsum = state.Fsum.at[I_x].set(F_x)
        phist = phist.at[I_x].add(1)
        #
        F = cond(
            use_abf,
            # estimate_abf,
            lambda t: t[3] / np.maximum(N, t[4]),
            estimate_force,
            (nn, x, fnn, F_x, N_x, nstep // train_freq)
        )
        # F = np.where(nstep % train_freq == 0, state.F, F)
        #
        bias = np.reshape(-Jx.T @ F, state.bias.shape)
        #
        return CFFState(
            bias, hist, phist, phi, prob, Fsum, F, Wp, state.Wp, nn, fnn, nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers)


def _learn_potential(kT, shape, train, apply, layout, inputs, state):
    prob = state.prob + state.phist * np.exp(state.phi / kT)
    phi = kT * np.log(np.where(prob == 0, 1.0, prob))
    #
    # Should we reset the network parameters before training?
    nn = train(inputs, (state.nn, phi))
    #
    params = pack(nn.params, layout)
    phi = nn.std * apply(params, inputs).reshape(phi.shape)
    phi = phi - phi.min()
    #
    phist = np.zeros_like(state.phist)
    #
    return phist, phi, prob, nn


def _learn_forces(N, shape, train, apply, layout, inputs, state):
    hist = state.hist.reshape(shape)
    F = state.Fsum / np.maximum(hist, N)
    #
    # Should we reset the network parameters before training?
    fnn = train(inputs, (state.fnn, F), axes = tuple(range(F.ndim - 1)))
    #
    return fnn


def _train(fit, shift_and_scale, smooth, inputs, data, axes = None):
    nn, y = data
    y, mu, sigma = shift_and_scale(y, axes = axes)
    y = smooth(y)
    params = fit(nn.params, inputs, y).params
    return NNData(params, mu, sigma / y.std(axis = axes))


def _estimate_force(N, apply, layout, data):
    nn, x, fnn, F, _, sweep = data
    params = pack(nn.params, layout)
    fparams = pack(fnn.params, layout)
    dA = nn.std * np.float64(grad(apply, argnums = 1)(params, x).flatten())
    F = fnn.std * apply(fparams, x).reshape(F.shape) + fnn.mean
    omega = np.minimum(sweep, 4) / 8
    return omega * dA + (1 - omega) * F
