# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from jax import grad, vmap
from jax.lax import cond
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

class ANNState(NamedTuple):
    bias:   JaxArray
    hist:   JaxArray
    phi:    JaxArray
    prob:   JaxArray
    nn:     NNData
    nstep:  Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ANN(NNSamplingMethod):
    snapshot_flags = {"positions", "indices"}

    def build(self, snapshot, helpers):
        if "kT" not in self.kwargs:
            raise ValueError("The value of kT must be provided")

        self.kT = self.kwargs["kT"]
        self.train_freq = self.kwargs.get("train_freq", 5000)
        model = self.kwargs.get("model", MLP)
        model_kwargs = self.kwargs.get("model_kwargs", dict())

        def build_model(ins, outs, topology, **kwargs):
            return model(ins, outs, topology, **model_kwargs, **kwargs)

        self.build_model = build_model
        self.optimizer = self.kwargs.get("optimzer", LevenbergMarquardt())

        self.external_force = self.kwargs.get("external_force", lambda rs: 0)

        return _ann(self, snapshot, helpers)


def _ann(method: ANN, snapshot, helpers):
    cv = method.cv
    kT = method.kT
    grid = method.grid
    train_freq = method.train_freq
    external_force = method.external_force

    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    gshape = grid.shape if dims > 1 else (*grid.shape, 1)
    # Neural network and optimizer
    scale = partial(_scale, grid = grid)
    model = method.build_model(dims, 1, method.topology, transform = scale)
    fit = build_fitting_function(model, method.optimizer)
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
    train = partial(_train, fit, smooth if dims > 1 else lambda y: vmap(smooth)(y.T).T)
    learn_pmf = partial(_learn_pmf, kT, train, model.apply, layout, inputs)
    estimate_force = partial(
        _estimate_force,
        lambda params, x: model.apply(params, x).sum(),
        layout
    )

    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.ones(gshape, dtype = np.uint32)
        phi = np.zeros(gshape)
        prob = np.ones(gshape)
        nn = NNData(ps, np.array(0.0), np.array(1.0))
        return ANNState(bias, hist, phi, prob, nn, 1)

    def update(state, data):
        nstep = state.nstep
        use_nn = nstep > 2 * train_freq
        #
        # Learn free energy / train NN
        hist, phi, prob, nn = cond(
            use_nn & (nstep % train_freq == 1),
            learn_pmf,
            lambda state: (state.hist, state.phi, state.prob, state.nn),
            state
        )
        # Compute the collective variable and its jacobian
        x, Jx = cv(data)
        #
        I_x = get_grid_index(x)
        hist = hist.at[I_x].add(1)
        #
        F = cond(use_nn, estimate_force, lambda _: np.zeros(dims), (nn, x))
        bias = np.reshape(-Jx.T @ F, state.bias.shape)
        bias = bias + external_force(data)
        #
        return ANNState(bias, hist, phi, prob, nn, nstep + 1)

    return snapshot, initialize, generalize(update, helpers)


def _learn_pmf(kT, train, apply, layout, inputs, state):
    prob = state.prob + state.hist * np.exp(state.phi / kT)
    phi = kT * np.log(prob)
    #
    nn = train(inputs, state.nn, phi)
    #
    params = pack(nn.params, layout)
    phi = nn.std * apply(params, inputs).reshape(phi.shape)
    phi = phi - phi.min()
    #
    hist = np.zeros_like(state.hist)
    #
    return hist, phi, prob, nn


def _train(fit, smooth, inputs, nn, y):
    y, mean, std = normalize(y)
    reference = smooth(y)
    params = fit(nn.params, inputs, reference).params
    return NNData(params, mean, std / reference.std())


def _estimate_force(apply, layout, data):
    nn, x = data
    params = pack(nn.params, layout)
    return nn.std * np.float64(grad(apply, argnums = 1)(params, x).flatten())
