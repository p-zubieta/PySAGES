# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple

from jax import numpy as np
from jax import random, vmap
from jax.numpy.linalg import norm
from jax.scipy import signal
from jax.tree_util import PyTreeDef, tree_flatten
from numpy import cumsum
from plum import Dispatcher

# Dispatcher for the `ml` submodule
dispatch = Dispatcher()


# %% Models
class ParametersLayout(NamedTuple):
    """
    Holds the information needed to pack flatten parameters of a
    `jax.example_libraries.stax.serial` model.
    """

    structure: PyTreeDef
    shapes: list
    separators: list


def unpack(params):
    """
    Returns the parameters of a `jax.example_libraries.stax.serial` model stacked
    into a flat vector. This representation is more convenient for computing
    the jacobian of the errors of the model.
    """
    data, structure = tree_flatten(params)
    ps = np.hstack([values.flatten() for values in data])
    shapes = [values.shape for values in data]
    separators = cumsum([prod(s) for s in shapes[:-1]])
    return ps, ParametersLayout(structure, shapes, list(separators))


def pack(params, layout):
    """
    Repacks the flatten parameters of a `jax.example_libraries.stax.serial` model
    previously flatten with `unpack`.
    """
    structure, shapes, separators = layout
    partition = params.split(separators)
    ps = [p.reshape(s) for (p, s) in zip(partition, shapes)]
    return structure.unflatten(ps)


def number_of_weights(topology):
    k = topology[0]
    n = 0
    for i in range(1, len(topology)):
        m = topology[i]
        n += (k + 1) * m
        k = m
    return n


# %% Objectives, Costs, Regularization
def sum_squares(v):
    v = np.asarray(v).flatten()
    return v @ v.T


# %% Data smoothing
def blackman(M, n):
    x = 2 * np.pi * n / (M - 1)
    return 0.42 + 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)


def blackman_kernel(dims, M):
    n = M - 2
    apply = vmap(lambda ns: blackman(M, norm(np.float64(ns)) / 2))
    inds = np.stack(np.meshgrid(*(np.arange(1 - n, n, 2) for _ in range(dims))), axis=-1)
    kernel = apply(inds.reshape(-1, dims))
    return (kernel / kernel.sum()).reshape(*(n for _ in range(dims)))


def kernel_smoother(ndims, size, periodic=False, kernel_dtype=np.float32):
    assert size > 0
    boundary = "wrap" if periodic else "edge"
    kernel = np.asarray(blackman_kernel(ndims, size + 2), dtype=kernel_dtype)
    return vmap(lambda data: smooth(data, kernel, boundary))


def smooth(data, kernel, boundary):
    data_dtype = data.dtype
    data = np.asarray(data, dtype=kernel.dtype)
    return np.asarray(convolve(data.T, kernel, boundary=boundary), dtype=data_dtype).T


# %% Miscellanea
def convolve(data, kernel, boundary):
    """
    Wrapper around `jax.scipy.signal.convolve`. It first pads the data,
    depending on the size of the kernel, and chooses a padding mode depending
    on whether the boundaries are periodic or not.
    """
    n = kernel.ndim
    if n == 1:
        padding = (kernel.size - 1) // 2
    else:
        padding = [tuple((s - 1) // 2 for _ in range(n)) for s in kernel.shape]

    def pad(slice):
        return np.pad(slice, padding, mode=boundary)

    return signal.convolve(pad(data), kernel, mode="valid")


def rng_key(seed=0, n=2):
    """
    Returns a pseudo-randomly generated key, constructed by calling
    `jax.random.PRNGKey(seed)` and then splitting it `n` times.
    """
    key = random.PRNGKey(seed)
    for _ in range(n):
        key, _ = random.split(key)
    return key


def prod(xs):
    y = 1
    for x in xs:
        y *= x
    return y
