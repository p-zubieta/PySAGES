# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from copy import deepcopy
from importlib import import_module
from typing import Union

import numpy

from jax import numpy as np
from jax.tree_util import register_pytree_node
from plum import Dispatcher, parametric, type_of

import jaxlib.xla_extension as xe


# PySAGES main dispatcher
dispatch = Dispatcher()


class ToCPU:
    pass


@parametric(runtime_type_of=True)
class JaxArray(xe.DeviceArrayBase):
    """A type for JAX's arrays where the type parameter specifies the number of
    dimensions."""


# Aliases
JaxScalar = JaxArray[0]
JaxVector = JaxArray[1]
JaxMatrix = JaxArray[2]
Bool = Union[JaxScalar, bool]
Float = Union[JaxScalar, float]
Int = Union[JaxScalar, int]
Scalar = Union[None, bool, int, float]


@type_of.dispatch
def type_of(x: xe.DeviceArrayBase):
    return JaxArray[x.ndim]


# - https://github.com/google/jax/issues/446
# - https://github.com/google/jax/issues/806
def register_pytree_namedtuple(cls):
    register_pytree_node(
        cls,
        lambda xs: (tuple(xs), None),  # tell JAX how to unpack
        lambda _, xs: cls(*xs),  # tell JAX how to pack back
    )
    return cls


@dispatch
def copy(x: Scalar):
    return x


@dispatch(precedence=1)
def copy(t: tuple, *args):  # noqa: F811 # pylint: disable=C0116,E0102
    return tuple(copy(x, *args) for x in t)


@dispatch
def copy(x: JaxArray):  # noqa: F811 # pylint: disable=C0116,E0102
    return x.copy()


@dispatch
def copy(x, _: ToCPU):  # noqa: F811 # pylint: disable=C0116,E0102
    return deepcopy(x)


@dispatch
def copy(x: JaxArray, _: ToCPU):  # noqa: F811 # pylint: disable=C0116,E0102
    return numpy.asarray(x._value)


def identity(x):
    return x


def row_sum(x):
    """
    Sum array `x` along each of its row (`axis = 1`),
    """
    return np.sum(x.reshape(np.size(x, 0), -1), axis=1)


def gaussian(a, sigma, x):
    """
    N-dimensional origin-centered gaussian with height `a` and standard deviation `sigma`.
    """
    return a * np.exp(-row_sum((x / sigma) ** 2) / 2)


def try_import(new_name, old_name):
    try:
        return import_module(new_name)
    except ModuleNotFoundError:
        return import_module(old_name)
