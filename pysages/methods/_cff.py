# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from jax import grad, vmap
from jax.lax import cond
from jax.scipy import linalg
from plum import dispatch
from typing import NamedTuple, Tuple

from pysages.ml.models import MLP, Siren
from pysages.ml.objectives import (
    Sobolev1SSE,
    # L2Regularization,
    # VarRegularization,
)
from pysages.ml.optimizers import (
    LevenbergMarquardt,
    # LevenbergMarquardtBR,
    # update_hyperparams,
)
from pysages.ml.training import NNData, build_fitting_function, normalize, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.approxfun import compute_mesh, scale as _scale
from pysages.grids import Grid, build_indexer
from pysages.methods.core import NNSamplingMethod, generalize
from pysages.utils import Bool, Int, JaxArray
# from pysages.methods._funn import _estimate_abf

import jax.numpy as np


# ======== #
#   FUNN   #
# ======== #


class CFFState(NamedTuple):
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
    nn:    NNData
    nstep: Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialCFFState(NamedTuple):
    hist: JaxArray
    Fsum: JaxArray
    xi:   JaxArray
    ind:  Tuple
    nn:   NNData
    pred: Bool


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
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    gshape = grid.shape if dims > 1 else (*grid.shape, 1)
    # Neural network and optimizer
    scale = partial(_scale, grid = grid)
    model = method.build_model(dims, dims, method.topology, transform = scale)
    ps, _ = unpack(model.parameters)
    # Helper methods
    get_grid_index = build_indexer(grid)

    learn_pmf = build_pmf_learner(method, model)
    estimate_force = build_force_estimator(method, grid, model)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(gshape, dtype = np.uint32)
        histp = np.ones(gshape, dtype = np.uint32)
        A = np.zeros(gshape)
        prob = np.zeros(gshape)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, np.zeros(dims), np.array(1.0))
        xi, _ = cv(helpers.query(snapshot))
        return CFFState(bias, hist, histp, A, prob, Fsum, F, Wp, Wp_, xi, nn, 1)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        nstep = state.nstep
        use_abf = nstep <= 6 * train_freq
        #
        # Estimate free energy / train NN
        histp, A, prob, nn = cond(
            (nstep % train_freq == 1) & ~use_abf,
            learn_pmf,
            lambda state: (state.histp, state.A, state.prob, state.nn),
            state,
        )
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = linalg.solve(Jxi @ Jxi.T, Jxi @ p, sym_pos = "sym")
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.F)
        histp = histp.at[I_xi].add(1)
        #
        F = estimate_force(PartialCFFState(hist, Fsum, xi, I_xi, nn, use_abf))
        bias = (-Jxi.T @ F).reshape(state.bias.shape)
        #
        return CFFState(
            bias, hist, histp, A, prob, Fsum, F, Wp, state.Wp, xi, nn, nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers)


def build_pmf_learner(method: CFF, model):
    N = method.N
    kT = method.kT
    grid = method.grid
    optimizer = method.optimizer

    dims = grid.shape.size
    shape = (*grid.shape, 1)
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    _, layout = unpack(model.parameters)

    w = 5 if type(model) is Siren else 7
    smooth = partial(
        convolve,
        kernel = blackman_kernel(dims, w),
        boundary = "wrap" if grid.is_periodic else "edge",
    )

    def vsmooth(y):
        return vmap(smooth)(y.T).T

    preprocess = partial(_preprocess, smooth if dims > 1 else vsmooth, vsmooth)
    # preprocess = (lambda y, dy: (y, dy, 1.0))
    fit = build_fitting_function(model, optimizer)

    def train(nn, data):
        y, dy, s = preprocess(*data)
        params = fit(nn.params, inputs, (y, dy)).params
        return NNData(params, nn.mean, s)

    def learn_pmf(state):
        prob = state.prob + state.histp * np.exp(state.A / kT)
        A = kT * np.log(prob)
        #
        F = state.Fsum / np.maximum(N, state.hist.reshape(shape))
        #
        # Should we reset the model parameters before training?
        nn = train(state.nn, (A, F))
        #
        params = pack(nn.params, layout)
        A = nn.std * model.apply(params, inputs).reshape(A.shape)
        A = A - A.min()
        #
        histp = np.zeros_like(state.histp)
        #
        return histp, A, prob, nn

    return learn_pmf


def _preprocess(smooth, smooth_grad, A, F):
    axes = tuple(range(F.ndim - 1))
    normalize
    A, _, Astd = normalize(A)
    Fstd = F.std(axis = axes)
    s = np.maximum(Astd, Fstd.max())
    # s = np.maximum(A.std(), Fstd.max())
    A = smooth(A)
    F = smooth_grad(F / Fstd)
    A = A * (Astd / A.std() / s)
    F = F * (Fstd / F.std(axis = axes) / s)
    # A = A / s
    # F = F / s
    return A, F, s


def _apply_restraints(lo, up, klo, kup, xi):
    return np.where(xi < lo, klo * (xi - lo), np.where(xi > up, kup * (xi - up), 0))


@dispatch
def build_force_estimator(method: CFF, grid: Grid, model):
    k = method.k
    N = method.N
    _, layout = unpack(model.parameters)

    def apply(params, x):
        return model.apply(params, x).sum()

    get_grad = grad(apply, argnums = 1)

    def estimate_abf(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(N, state.hist[i])

    def predict_force(state):
        nn = state.nn
        xi = state.xi
        params = pack(nn.params, layout)
        return nn.std * np.float64(get_grad(params, xi).flatten())

    def _estimate_force(state):
        return cond(state.pred, estimate_abf, predict_force, state)

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
                np.any(np.array(state.ind) == grid.shape),
                apply_restraints,
                _estimate_force,
                state,
            )

    return estimate_force
