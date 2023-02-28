# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, NamedTuple

from jax import jit
from jax import numpy as np

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.utils import ToCPU, copy


class Sampler:
    def __init__(self, atoms, method_bundle, callback: Callable):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.atoms = atoms
        self.callback = callback
        self.snapshot = initial_snapshot
        self.update = method_update

    def restore(self, prev_snapshot):
        atoms = self.atoms
        velocities, masses = prev_snapshot.vel_mass
        atoms.set_positions(prev_snapshot.positions)
        atoms.set_masses(masses)  # masses need to be set before velocities
        atoms.set_velocities(velocities)
        atoms.set_cell(list(prev_snapshot.box.H))
        self.snapshot = prev_snapshot

    def take_snapshot(self):
        return copy(self.snapshot)


def take_snapshot(simulation):
    atoms = simulation.atoms
    #
    positions = np.asarray(atoms.get_positions())
    forces = np.asarray(atoms.get_forces(md=True))
    ids = np.arange(atoms.get_global_number_of_atoms())
    #
    velocities = np.asarray(atoms.get_velocities())
    masses = np.asarray(atoms.get_masses()).reshape(-1, 1)
    vel_mass = (velocities, masses)
    #
    a = atoms.cell[0]
    b = atoms.cell[1]
    c = atoms.cell[2]
    H = ((a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2]))
    origin = (0.0, 0.0, 0.0)
    dt = simulation.dt
    # ASE doesn't use images explicitely
    return Snapshot(positions, vel_mass, forces, ids, None, Box(H, origin), dt)


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


def build_helpers(context, sampling_method):
    def dimensionality():
        return 3  # are all ASE simulations boxes 3-dimensional?

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers


def override_run_method(simulation, sampler):
    """
    Wraps the original step function of the `ase.md.MolecularDynamics`
    instance `simulation`, and injects calls to the sampling method's `update`
    and the user provided callback.
    """
    number_of_steps = simulation.get_number_of_steps
    simulation._step = simulation.step

    def wrapped_step():
        sampler.snapshot = take_snapshot(simulation)
        sampler.state = sampler.update(sampler.snapshot, sampler.state)
        if sampler.state.bias is not None:
            forces = copy(sampler.snapshot.forces + sampler.state.bias, ToCPU())
        else:
            forces = copy(sampler.snapshot.forces, ToCPU())
        simulation._step(forces=forces)
        if sampler.callback:
            sampler.callback(sampler.snapshot, sampler.state, number_of_steps())

    simulation.step = wrapped_step

    return simulation.run


class View(NamedTuple):
    synchronize: Callable


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    """
    Entry point for the backend code, it gets called when the simulation
    context is wrapped within `pysages.run`.
    """
    context = sampling_context.context
    sampling_method = sampling_context.method
    snapshot = take_snapshot(context)
    helpers = build_helpers(sampling_context, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(context.atoms, method_bundle)
    sampling_context.view = View((lambda: None))
    sampling_context.run = override_run_method(context, sampler)
    return sampler
