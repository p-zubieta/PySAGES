# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple

from jax.lax import while_loop

from pysages.ml.optimizers import build
from pysages.utils import JaxArray


class NNData(NamedTuple):
    params: JaxArray
    mean: JaxArray
    std: JaxArray


def normalize(data, axes=None):
    mean = data.mean(axis=axes)
    std = data.std(axis=axes)
    return (data - mean) / std, mean, std


def build_fitting_function(model, optimizer):
    """
    Returns a function that fits the model parameters to the reference data. We
    specialize on both the model and the optimizer to partially evaluate all the
    simulation-time-independent information.
    """
    initialize, keep_iterating, update = build(optimizer, model)

    def fit(params, x, y):
        state = initialize(params, x, y)
        state = while_loop(keep_iterating, update, state)
        return state

    return fit
