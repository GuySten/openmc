"""Run the adjoint side populations, with probes and rays, in transport."""

import shutil

import numpy as np
import openmc
import pytest

from tests.regression_tests import config


@pytest.fixture
def model():
    """A U235 sphere in a heavy-water reflector, whose photoneutrons are
    probed with lines between the D(gamma,n) threshold and the highest
    allowed probe energy"""
    model = openmc.Model()

    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 18.0)
    heavy_water = openmc.Material()
    heavy_water.add_nuclide('H2', 2.0)
    heavy_water.add_nuclide('O16', 1.0)
    heavy_water.set_density('g/cm3', 1.1)
    model.materials = openmc.Materials([fuel, heavy_water])

    core = openmc.Sphere(r=6.0)
    outer = openmc.Sphere(r=20.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([
        openmc.Cell(fill=fuel, region=-core),
        openmc.Cell(fill=heavy_water, region=+core & -outer),
    ])

    model.settings.particles = 400
    model.settings.batches = 5
    model.settings.inactive = 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.adjoint_populations = {
        'n_generation': 1,
        'photoneutrons': True,
        'photoneutron_probe_energies': [2.4e6, 2.8e6, 3.2e6],
        'photoneutron_ray_comb_points': 3,
    }
    return model


def run_kwargs():
    if config['mpi']:
        return {'mpi_args': [config['mpiexec'], '-n', config['mpi_np']]}
    return {}


def test_ray_and_probe_fission_weights_agree(model, run_in_tmpdir):
    """The rays and the probes are emitted from the same recorded fission
    events, so their fission weights, sums of w/k sigma_f/sigma_t over the
    events, must agree in every batch. Under MPI both must be summed over the
    ranks, with the rays' scores normalized by them."""
    sp_path = model.run(**run_kwargs())

    with openmc.StatePoint(sp_path) as sp:
        pops = sp.adjoint_populations
        probe_f = pops.probe_fission_weight
        ray_f = pops.ray_fission_weight
        ray_w = pops.ray_weight

    assert probe_f.shape == ray_f.shape == (3, 1)
    assert np.all(probe_f > 0.0)
    np.testing.assert_allclose(ray_f, probe_f, rtol=1e-12)
    assert np.all(ray_w.sum(axis=(1, 2, 3)) > 0.0)


RESULTS = ['weight', 'weight_t0', 'site_weight', 'n_histories',
           'probe_weight', 'probe_fission_weight', 'n_probe_histories',
           'ray_weight', 'ray_comb_weight', 'ray_fission_weight',
           'n_ray_segments']


def read_results(path):
    with openmc.StatePoint(path) as sp:
        group = sp._f['adjoint_populations']
        return int(group['n_batches'][()]), \
            {name: group[name][()] for name in RESULTS}


@pytest.mark.parametrize('perturbed_importance', [False, True])
def test_restart(model, perturbed_importance, run_in_tmpdir):
    """A run restarted from a statepoint continues the side populations'
    sums: it ends with the results of the uninterrupted run. With the
    perturbed importance, the branch targets carry over from one batch to the
    next."""
    model.settings.adjoint_populations['perturbed_importance'] = \
        perturbed_importance
    model.settings.batches = 6
    model.settings.statepoint = {'batches': [4, 6]}
    # One thread, so that the sums over threads' fission events are added in
    # the same order in both runs
    model.run(threads=1, **run_kwargs())
    n_batches, expected = read_results('statepoint.6.h5')
    assert n_batches == 4
    shutil.move('statepoint.6.h5', 'uninterrupted.h5')

    model.run(restart_file='statepoint.4.h5', threads=1, **run_kwargs())
    n_batches, results = read_results('statepoint.6.h5')
    assert n_batches == 4
    for name in RESULTS:
        np.testing.assert_array_equal(results[name], expected[name],
                                      err_msg=name)


def test_restart_with_other_settings(model, run_in_tmpdir):
    """Sums made with other side-population settings are not continued."""
    model.settings.batches = 4
    model.run(**run_kwargs())

    model.settings.batches = 5
    model.settings.adjoint_populations['photoneutron_probe_energies'] = \
        [2.4e6, 2.9e6, 3.2e6]
    with pytest.raises(RuntimeError, match='settings differ'):
        model.run(restart_file='statepoint.4.h5', **run_kwargs())


def test_restart_without_results(model, run_in_tmpdir):
    """A restart in the active batches needs the sums of the batches
    before it."""
    adjoint_populations = model.settings.adjoint_populations
    model.settings.adjoint_populations = {}
    model.settings.photonuclear_physics = False
    model.settings.batches = 4
    model.run(**run_kwargs())

    model.settings.batches = 5
    model.settings.adjoint_populations = adjoint_populations
    model.settings.photonuclear_physics = True
    with pytest.raises(RuntimeError, match='no adjoint side-population'):
        model.run(restart_file='statepoint.4.h5', **run_kwargs())


@pytest.fixture
def be_d2o_model(model):
    """The heavy-water model with a beryllium shell around the core: two
    photoneutron targets, probed with lines from below the D(gamma,n)
    threshold, where only beryllium makes photoneutrons"""
    fuel, heavy_water = model.materials
    beryllium = openmc.Material()
    beryllium.add_nuclide('Be9', 1.0)
    beryllium.set_density('g/cm3', 1.85)
    model.materials.append(beryllium)
    core = openmc.Sphere(r=6.0)
    shell = openmc.Sphere(r=9.0)
    outer = openmc.Sphere(r=20.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([
        openmc.Cell(fill=fuel, region=-core),
        openmc.Cell(fill=beryllium, region=+core & -shell),
        openmc.Cell(fill=heavy_water, region=+shell & -outer),
    ])
    model.settings.adjoint_populations['photoneutron_probe_energies'] = \
        [1.9e6, 2.4e6, 2.8e6, 3.2e6]
    return model


def test_photoneutron_targets(be_d2o_model, run_in_tmpdir):
    """Probes and rays tallied by photoneutron target: below the D(gamma,n)
    threshold only beryllium makes photoneutrons, and the targets' sums are
    the untallied run's (the targets change only where a tree scores, not
    which trees grow)."""
    model = be_d2o_model
    model.run(threads=1, **run_kwargs())
    _, plain = read_results('statepoint.5.h5')

    model.settings.adjoint_populations['photoneutron_targets'] = ['Be9', 'H2']
    sp_path = model.run(threads=1, **run_kwargs())
    with openmc.StatePoint(sp_path) as sp:
        pops = sp.adjoint_populations
        assert pops.probe_targets == ['Be9', 'H2', 'other']
        pw = pops.probe_weight_by_target
        rw = pops.ray_weight_by_target
        np.testing.assert_allclose(pops.probe_weight, pw.sum(axis=3))
        lines = pops.probe_energies
        ray_lines = pops.ray_photon_energies

    n_b = pw.shape[0]
    np.testing.assert_allclose(
        pw.sum(axis=3).ravel(), plain['probe_weight'], rtol=1e-9, atol=0)
    np.testing.assert_allclose(
        rw.sum(axis=3).ravel(), plain['ray_weight'], rtol=1e-9, atol=0)
    # Below the D(gamma,n) threshold no deuteron photoneutron, at any depth
    below = lines < 2.2246e6
    assert not pw[:, :, below, 1].any()
    assert not rw[:, :, ray_lines < 2.2246e6, 1].any()
    # Both targets make photoneutrons above it
    for t in (0, 1):
        assert pw[:, :, ~below, t].sum() + rw[:, :, ~(ray_lines < 2.2246e6),
                                              t].sum() > 0.0
    assert n_b == 3

    # The importance rebuilt per target from the materials, each counting
    # its own nuclide's photoneutrons with its own threshold
    fuel, heavy_water, beryllium = model.materials
    with openmc.StatePoint(sp_path) as sp:
        pops = sp.adjoint_populations
        by_target = pops.probe_importance({'Be9': beryllium, 'H2': heavy_water})
    assert by_target.targets['Be9'].threshold < 1.7e6
    assert by_target.targets['H2'].threshold == pytest.approx(2.2246e6,
                                                              rel=1e-3)
    assert by_target.rebuilt(np.array([2.0e6]))[0] >= 0.0
