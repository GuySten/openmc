"""Branchless collisions in the shadow trees of the adjoint side populations:
the trees' expected scores are those of the branching trees, so every
estimator must agree with the branching run within statistics."""

import numpy as np
import openmc
import pytest

from tests.regression_tests import config


def run_kwargs(**kwargs):
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    return kwargs


def reflected_sphere(branchless, photoneutrons=False):
    """A U235 sphere in a light- (or heavy-) water reflector"""
    openmc.reset_auto_ids()
    model = openmc.Model()
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 18.0)
    water = openmc.Material()
    water.add_nuclide('H2' if photoneutrons else 'H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.1 if photoneutrons else 1.0)
    model.materials = openmc.Materials([fuel, water])

    core = openmc.Sphere(r=6.0)
    outer = openmc.Sphere(r=16.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([
        openmc.Cell(fill=fuel, region=-core),
        openmc.Cell(fill=water, region=+core & -outer),
    ])

    model.settings.particles = 2000
    model.settings.batches = 25
    model.settings.inactive = 5
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    adjoint_populations = {'n_generation': 4, 'root_fraction': 0.5,
                           'branchless': branchless}
    if photoneutrons:
        model.settings.batches = 12
        model.settings.inactive = 2
        model.settings.photon_transport = True
        model.settings.photonuclear_physics = True
        adjoint_populations.update({
            'n_generation': 2,
            'photoneutrons': True,
            'photoneutron_probe_energies': [2.4e6, 2.8e6, 3.2e6],
            'photoneutron_ray_comb_points': 3,
            'photoneutron_probe_fraction': 20.0,
            'photoneutron_probe_root_fraction': 1.0,
            'photoneutron_ray_root_fraction': 1.0,
        })
    model.settings.adjoint_populations = adjoint_populations
    return model


def agree(a, b, n_sigma=4.0):
    """Two independent estimates agree within n_sigma combined deviations"""
    return abs(a.n - b.n) <= n_sigma * np.hypot(a.s, b.s)


def test_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    value = {'n_generation': 3, 'branchless': True, 'branchless_split': 3.0}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value
    with pytest.raises(TypeError):
        s.adjoint_populations = {'n_generation': 3, 'branchless': 1}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3, 'branchless_split': 0.5}


def test_kinetics_match_branching(run_in_tmpdir):
    """beta_eff and the generation time, at every depth, with and without
    splitting the heavy lines"""
    results = {}
    for branchless, split in ((False, None), (True, 0.0), (True, 2.0)):
        model = reflected_sphere(branchless)
        if split is not None:
            model.settings.adjoint_populations['branchless_split'] = split
        sp_path = model.run(path=f'branchless_{branchless}_{split}.xml',
                            **run_kwargs())
        with openmc.StatePoint(sp_path) as sp:
            pops = sp.adjoint_populations
            assert pops.branchless == branchless
            if split is not None:
                assert pops.branchless_split == split
            results[split] = (
                [pops.beta_eff(depth=d) for d in range(1, 5)],
                [pops.generation_time(depth=d) for d in range(1, 5)])

    for split in (0.0, 2.0):
        for quantity in range(2):
            for a, b in zip(results[None][quantity], results[split][quantity]):
                assert a.n > 0.0 and b.n > 0.0
                assert agree(a, b)


def test_photoneutrons_match_branching(run_in_tmpdir):
    """The probe lines' reactivity, direct (the rays') and scattered"""
    results = {}
    for branchless in (False, True):
        model = reflected_sphere(branchless, photoneutrons=True)
        sp_path = model.run(path=f'branchless_{branchless}.xml',
                            **run_kwargs())
        with openmc.StatePoint(sp_path) as sp:
            results[branchless] = sp.adjoint_populations.probe_reactivity()

    for part_a, part_b in zip(results[False], results[True]):
        for a, b in zip(part_a, part_b):
            assert agree(a, b)
    # The rays make every line's direct part
    for a, b in zip(results[False][0], results[True][0]):
        assert a.n > 0.0 and b.n > 0.0


def test_refuses_neutron_energy_max(run_in_tmpdir):
    model = reflected_sphere(True)
    model.settings.batches = 3
    model.settings.inactive = 1
    model.settings.energy_max = {'neutron': 1.0e7}
    with pytest.raises(RuntimeError, match='branchless'):
        model.run(**run_kwargs())
