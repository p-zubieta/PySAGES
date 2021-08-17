# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from jax import jit, vmap
from jax.lax import cond
from jax.scipy import linalg
from typing import NamedTuple
from pysages.ml.models import MLP
# from pysages.ml.objectives import L2Regularization, estimate_l2_coefficient
from pysages.ml.optimizers import (
    LevenbergMarquardt,
    # LevenbergMarquardtBR,
    # update_hyperparams,
)
from pysages.ml.training import NNData, build_fitting_function, normalize, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.approxfun import compute_mesh, scale as _scale
from pysages.grids import build_indexer
from pysages.utils import Int, JaxArray

from .core import NNSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np


# ======== #
#   FUNN   #
# ======== #

class FUNNState(NamedTuple):
    bias:   JaxArray
    hist:   JaxArray
    Fsum:   JaxArray
    F:      JaxArray
    Wp:     JaxArray
    Wp_:    JaxArray
    nn:     NNData
    nstep:  Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FUNN(NNSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        N = np.asarray(self.kwargs.get('N', 100))
        train_freq = np.asarray(self.kwargs.get("train_freq", 5000))
        optimizer = self.kwargs.get("optimzer", LevenbergMarquardt())
        model = self.kwargs.get("model", MLP)
        model_kwargs = self.kwargs.get("model_kwargs", dict())

        def build_model(ins, outs, topology, **kwargs):
            return model(ins, outs, topology, **model_kwargs, **kwargs)

        return _funn(
            snapshot, self.cv, self.grid, self.topology,
            build_model, N, train_freq, optimizer, helpers
        )


def _funn(snapshot, cv, grid, topology, build_model, N, train_freq, optimizer, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    # Neural network and optimizer
    scale = partial(_scale, grid = grid)
    model = build_model(dims, dims, topology, f_in = scale)
    fit = build_fitting_function(model, optimizer)
    ps, layout = unpack(model.parameters)
    # Training data
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    smooth = partial(
        convolve,
        kernel = blackman_kernel(dims, 7),
        boundary = "wrap" if grid.is_periodic else "edge"
    )
    # Helper methods
    get_grid_index = build_indexer(grid)
    train = jit(partial(_train, fit, lambda y: vmap(smooth)(y.T).T, inputs))
    learn_forces = jit(partial(_learn_forces, train, train_freq, ps))
    estimate_abf = jit(partial(_estimate_abf, N))
    estimate_funn = jit(partial(_estimate_funn, model.apply, layout))

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.ones(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, F, F)
        return FUNNState(bias, hist, Fsum, F, Wp, Wp_, nn, 1)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        nstep = state.nstep
        use_abf = nstep <= 2 * train_freq
        # NN training
        nn = cond((nstep % train_freq == 1) & ~use_abf, learn_forces, lambda s: s.nn, state)
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        p = data.momenta
        Wp = linalg.solve(Jx @ Jx.T, Jx @ p, sym_pos="sym")
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_x = get_grid_index(x)
        N_x = state.hist[I_x] + 1
        F_x = state.Fsum[I_x] + dWp_dt + state.F
        hist = state.hist.at[I_x].set(N_x)
        Fsum = state.Fsum.at[I_x].set(F_x)
        #
        F = cond(use_abf, estimate_abf, estimate_funn, (nn, x, F_x, N_x))
        bias = np.reshape(-Jx.T @ F, state.bias.shape)
        #
        return FUNNState(bias, hist, Fsum, F, Wp, state.Wp, nn, state.nstep + 1)

    return snapshot, initialize, generalize(update, helpers)


def _learn_forces(train, train_freq, ps, state):
    # Reset the network parameters before the first training cycles
    nn = state.nn
    nn = cond(
        state.nstep <= 4 * train_freq,
        lambda nn: NNData(ps, nn.mean, nn.std),
        lambda nn: nn,
        nn
    )
    hist = np.expand_dims(state.hist, state.hist.ndim)
    F = state.Fsum / hist
    return train(nn, F)


def _train(fit, smooth, inputs, nn, y):
    axes = tuple(range(y.ndim - 1))
    y, mean, std = normalize(y, axes = axes)
    reference = smooth(y)
    params = fit(nn.params, inputs, reference).params
    return NNData(params, mean, std / reference.std(axis = axes))


def _estimate_funn(apply, layout, data_bundle):
    nn, x, F, _ = data_bundle
    params = pack(nn.params, layout)
    return nn.std * apply(params, x).reshape(F.shape) + nn.mean


def _estimate_abf(N, data_bundle):
    *_, F_x, N_x = data_bundle
    return F_x / np.maximum(N_x, N)
