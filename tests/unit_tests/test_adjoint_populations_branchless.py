"""Branchless collisions in the shadow trees of the adjoint side populations:
the trees' expected scores are those of the branching trees, so every
estimator must agree with the branching run within statistics."""

import h5py
import numpy as np
import openmc
import pytest
from openmc.adjoint_populations import CLASS_FISSION_BRANCH

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
        # So that the trees' photons make photoneutron branches
        model.settings.photoneutron_biasing = True
        adjoint_populations.update({
            'n_generation': 2,
            'photoneutrons': True,
            'perturbed_importance': True,
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
    value = {'n_generation': 3, 'branchless': True, 'branchless_split': 3.0,
             'branchless_depth': 1, 'expected_scores': True,
             'implicit_capture': True}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value
    with pytest.raises(TypeError):
        s.adjoint_populations = {'n_generation': 3, 'branchless': 1}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3, 'branchless_split': 0.5}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3, 'branchless_depth': -1}


def test_kinetics_match_branching(run_in_tmpdir):
    """beta_eff and the generation time, at every depth, with and without
    splitting the heavy lines, branching above depth 1, and branching trees
    with expected scores and implicit capture"""
    variants = {'branching': {},
                'no split': {'branchless_split': 0.0},
                'split': {'branchless_split': 2.0},
                'from depth 1': {'branchless_depth': 1},
                'expected scores': {'expected_scores': True},
                'implicit capture': {'expected_scores': True,
                                     'implicit_capture': True},
                'from depth 1, both': {'branchless_depth': 1,
                                       'expected_scores': True,
                                       'implicit_capture': True}}
    branching = ('branching', 'expected scores', 'implicit capture')
    results = {}
    for name, extra in variants.items():
        model = reflected_sphere(name not in branching)
        model.settings.adjoint_populations.update(extra)
        sp_path = model.run(path=f'{name.replace(" ", "_")}.xml',
                            **run_kwargs())
        with openmc.StatePoint(sp_path) as sp:
            pops = sp.adjoint_populations
            assert pops.branchless == (name not in branching)
            assert pops.expected_scores == extra.get('expected_scores', False)
            assert pops.implicit_capture == extra.get('implicit_capture',
                                                      False)
            if 'branchless_split' in extra:
                assert pops.branchless_split == extra['branchless_split']
            assert pops.branchless_depth == extra.get('branchless_depth', 0)
            results[name] = (
                [pops.beta_eff(depth=d) for d in range(1, 5)],
                [pops.generation_time(depth=d) for d in range(1, 5)])

    for name in variants:
        for quantity in range(2):
            for a, b in zip(results['branching'][quantity],
                            results[name][quantity]):
                assert a.n > 0.0 and b.n > 0.0
                assert agree(a, b)


def test_photoneutrons_match_branching(run_in_tmpdir):
    """The probe lines' reactivity, direct (the rays') and scattered, and the
    photoneutron branches' changes of beta_eff and the generation time, with
    branchless trees and with branching trees that score expected weights and
    capture implicitly"""
    variants = {'branching': (False, {}),
                'branchless': (True, {}),
                'expected, implicit': (False, {'expected_scores': True,
                                               'implicit_capture': True})}
    results = {}
    for name, (branchless, extra) in variants.items():
        model = reflected_sphere(branchless, photoneutrons=True)
        model.settings.adjoint_populations.update(extra)
        sp_path = model.run(path=f'{name.replace(" ", "_")}.xml',
                            **run_kwargs())
        with openmc.StatePoint(sp_path) as sp:
            pops = sp.adjoint_populations
            assert pops.weight[:, CLASS_FISSION_BRANCH].sum() > 0.0
            results[name] = (pops.probe_reactivity(),
                             [pops.delta_beta_eff(),
                              pops.delta_generation_time()])

    for name in variants:
        for part_a, part_b in zip(results['branching'][0], results[name][0]):
            for a, b in zip(part_a, part_b):
                assert agree(a, b)
        # The rays make every line's direct part
        for a, b in zip(results['branching'][0][0], results[name][0][0]):
            assert a.n > 0.0 and b.n > 0.0
        for a, b in zip(results['branching'][1], results[name][1]):
            assert agree(a, b)


def test_probe_photons_killed_below_threshold(run_in_tmpdir):
    """Probe photons are killed below the lowest photoneutron threshold of
    the model, H2's"""
    model = reflected_sphere(False, photoneutrons=True)
    model.settings.batches = 4
    model.settings.inactive = 1
    sp_path = model.run(**run_kwargs())
    with h5py.File(sp_path, 'r') as f:
        e_kill = f['adjoint_populations/probe_kill_energy'][()]
    # H2(gamma,n) opens at its Q value; U235 and O16 open far higher
    assert e_kill == pytest.approx(2.2246e6, rel=1e-3)


def test_refuses_branchless_depth_beyond_trees(run_in_tmpdir):
    model = reflected_sphere(True)
    model.settings.batches = 3
    model.settings.inactive = 1
    model.settings.adjoint_populations['branchless_depth'] = 4
    with pytest.raises(RuntimeError, match='branchless_depth'):
        model.run(**run_kwargs())


def test_refuses_neutron_energy_max(run_in_tmpdir):
    model = reflected_sphere(True)
    model.settings.batches = 3
    model.settings.inactive = 1
    model.settings.energy_max = {'neutron': 1.0e7}
    with pytest.raises(RuntimeError, match='branchless'):
        model.run(**run_kwargs())
