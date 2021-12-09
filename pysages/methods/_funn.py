# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from typing import NamedTuple, Tuple

from jax import grad, jit, numpy as np, vmap
from jax.lax import cond
from jax.scipy import linalg

from pysages.approxfun import compute_mesh, scale as _scale
from pysages.grids import build_indexer
from pysages.methods.core import NNSamplingMethod, generalize
from pysages.ml.models import MLP, Siren
from pysages.ml.objectives import GradientsSSE, L2Regularization
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, normalize, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.utils import Bool, Int, JaxArray


class FUNNState(NamedTuple):
    bias:  JaxArray
    hist:  JaxArray
    Fsum:  JaxArray
    F:     JaxArray
    Wp:    JaxArray
    Wp_:   JaxArray
    xi:    JaxArray
    nn:    NNData
    nstep: Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialState(NamedTuple):
    hist: JaxArray
    Fsum: JaxArray
    xi:   JaxArray
    ind:  Tuple
    nn:   NNData
    pred: Bool


class FUNN(NNSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get('N', 100))
        self.k = self.kwargs.get("k", None)
        self.train_freq = np.asarray(self.kwargs.get("train_freq", 5000))
        model = self.kwargs.get("model", MLP)
        model_kwargs = self.kwargs.get("model_kwargs", dict())

        def build_model(ins, outs, topology, **kwargs):
            return model(ins, outs, topology, **model_kwargs, **kwargs)

        self.build_model = build_model
        max_iters = self.kwargs.get("max_iters", 100 if model is Siren else 500)
        self.optimizer = self.kwargs.get("optimzer", LevenbergMarquardt(
            loss = GradientsSSE(), max_iters = max_iters, reg = L2Regularization(1e-4)
        ))
        self.external_force = self.kwargs.get("external_force", lambda rs: 0)

        return _funn(self, snapshot, helpers)


def _funn(method, snapshot, helpers):
    # N = method.N
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq
    external_force = method.external_force

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    # Neural network and optimizer
    scale = partial(_scale, grid = grid)
    model = method.build_model(dims, dims, method.topology, transform = scale)
    method.model = model

    fit = build_fitting_function(model, method.optimizer)
    ps, layout = unpack(model.parameters)
    # Training data
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    w = 3 if type(model) is Siren else 7
    smooth = partial(
        convolve,
        kernel = blackman_kernel(dims, w),
        boundary = "wrap" if grid.is_periodic else "edge"
    )
    # Helper methods
    get_grid_index = build_indexer(grid)
    train = jit(partial(_train, fit, lambda y: vmap(smooth)(y.T).T, inputs))
    learn_forces = jit(partial(_learn_forces, train, train_freq, ps))
    estimate_force = build_force_estimator(method)

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, F, F)
        xi, _ = cv(helpers.query(snapshot))
        return FUNNState(bias, hist, Fsum, F, Wp, Wp_, xi, nn, 1)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        nstep = state.nstep
        use_abf = nstep <= 2 * train_freq
        # NN training
        nn = cond(
            (nstep % train_freq == 1) & ~use_abf,
            learn_forces,
            lambda s: s.nn,
            state
        )
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        p = data.momenta
        Wp = linalg.solve(Jx @ Jx.T, Jx @ p, sym_pos = "sym")
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_x = get_grid_index(x)
        hist = state.hist.at[I_x].add(1)
        Fsum = state.Fsum.at[I_x].add(dWp_dt + state.F)
        #
        F = estimate_force(PartialState(hist, Fsum, x, I_x, nn, use_abf))
        bias = (-Jx.T @ F).reshape(state.bias.shape)
        bias = bias + external_force(data)
        #
        return FUNNState(bias, hist, Fsum, F, Wp, state.Wp, x, nn, state.nstep + 1)

    return snapshot, initialize, generalize(update, helpers)


def _learn_forces(train, train_freq, ps, state):
    # Reset the network parameters before the first training cycles
    # nn = state.nn
    # nn = cond(
    #     state.nstep <= 4 * train_freq,
    #     lambda nn: NNData(ps, nn.mean, nn.std),
    #     lambda nn: nn,
    #     nn
    # )
    hist = np.expand_dims(state.hist, state.hist.ndim)
    F = state.Fsum / np.maximum(hist, 1)
    return train(state.nn, F)


def _train(fit, smooth, inputs, nn, y):
    axes = tuple(range(y.ndim - 1))
    # y, mean, std = normalize(y, axes = axes)
    std = y.std(axis = axes)
    reference = smooth(y / std)
    params = fit(nn.params, inputs, reference).params
    return NNData(params, nn.mean, std / reference.std(axis = axes))


def _apply_restraints(lo, up, klo, kup, xi):
    return np.where(xi < lo, klo * (xi - lo), np.where(xi > up, kup * (xi - up), 0))


def build_force_estimator(method: FUNN):
    k = method.k
    N = method.N
    grid = method.grid
    model = method.model
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
        klo = k
        kup = k

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
