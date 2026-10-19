"""Run the adjoint side populations, with probes and rays, in transport."""

import shutil

import h5py
import numpy as np
import openmc
import openmc.deplete
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


def test_probe_photons_killed_below_threshold(model, run_in_tmpdir):
    """Probe photons are killed below the lowest photoneutron threshold of
    the model, H2's"""
    sp_path = model.run(**run_kwargs())
    with h5py.File(sp_path, 'r') as f:
        e_kill = f['adjoint_populations/probe_kill_energy'][()]
    # H2(gamma,n) opens at its Q value; U235 and O16 open far higher
    assert e_kill == pytest.approx(2.2246e6, rel=1e-3)


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


def test_probe_reactions(model, run_in_tmpdir, monkeypatch):
    """Probed reactions emit probes and cast rays from their own sites, in
    probe bins after the fissioning-nuclide bin. The driver, the fission,
    delayed and photoneutron trees and the fission bin's events are those of
    the run without them, and a reaction's weight is a collision estimate of
    its rate, per fission as a tally gives it."""
    model.run(threads=1, **run_kwargs())
    _, plain = read_results('statepoint.5.h5')
    with openmc.StatePoint('statepoint.5.h5') as sp:
        k_plain = sp.keff

    model.settings.adjoint_populations['photoneutron_probe_reactions'] = \
        ['H2(n,gamma)', ('O16', 103)]
    rates = openmc.Tally(name='rates')
    rates.nuclides = ['H2']
    rates.scores = ['(n,gamma)']
    fission = openmc.Tally(name='fission')
    fission.scores = ['fission']
    model.tallies = openmc.Tallies([rates, fission])
    sp_path = model.run(threads=1, **run_kwargs())
    _, results = read_results(sp_path)
    with openmc.StatePoint(sp_path) as sp:
        assert sp.keff.n == k_plain.n and sp.keff.s == k_plain.s
        pops = sp.adjoint_populations
        capture = sp.get_tally(name='rates')
        fissions = sp.get_tally(name='fission')
        tally_ratio = capture.mean.sum() / fissions.mean.sum()
        tally_sd = tally_ratio * np.hypot(
            capture.std_dev.sum() / capture.mean.sum(),
            fissions.std_dev.sum() / fissions.mean.sum())
    # (n_histories also counts the probe roots' trees, whose budget the
    # reaction bins share)
    for name in ('weight', 'weight_t0', 'site_weight'):
        np.testing.assert_array_equal(results[name], plain[name],
                                      err_msg=name)

    assert pops.probe_fission_nuclides == ['all']
    assert pops.probe_reactions == ['H2(n,gamma)', 'O16(n,p)']
    assert pops.probe_reaction_mt == {'H2(n,gamma)': ('H2', 102),
                                      'O16(n,p)': ('O16', 103)}
    n_b = pops.n_batches
    assert pops.probe_weight.shape[:2] == (n_b, 3)
    assert pops.ray_weight.shape[:2] == (n_b, 3)
    np.testing.assert_allclose(pops.probe_fission_weight[:, 0],
                               plain['probe_fission_weight'], rtol=1e-12)
    np.testing.assert_allclose(pops.ray_fission_weight,
                               pops.probe_fission_weight, rtol=1e-12)
    assert pops.probe_reaction_weight.shape == (n_b, 2)
    assert np.all(pops.probe_reaction_weight[:, 0] > 0.0)
    r = pops.reactions_per_fission('H2(n,gamma)')
    assert abs(r.n - tally_ratio) < 5.0 * np.hypot(r.s, tally_sd)
    with pytest.raises(ValueError, match='fissioning-nuclide bin'):
        pops.reactions_per_fission('all')
    # The reaction sites' probes made photoneutrons, and the table rebuilds
    assert pops.probe_weight[:, 1].sum() > 0.0
    heavy_water = model.materials[1]
    imp = pops.probe_importance(heavy_water, nuclide='H2:102')
    assert imp.name == 'H2(n,gamma)'
    assert imp.mean.max() > 0.0

    monkeypatch.delitem(openmc.config, 'chain_file', raising=False)
    with pytest.warns(UserWarning, match='No depletion chain'):
        pops.export_photon_importance('importance.h5', heavy_water)
    with h5py.File('importance.h5') as f:
        assert list(f.attrs['version']) == [1, 1]
        assert [n.decode() for n in f.attrs['reactions']] == \
            ['H2(n,gamma)', 'O16(n,p)']
        assert f['nuclides/all'].attrs['fission_fraction'] == 1.0
        g = f['reactions/H2(n,gamma)']
        assert g.attrs['nuclide'] == b'H2' and g.attrs['mt'] == 102
        assert g.attrs['reactions_per_fission'] == pytest.approx(r.n)
        np.testing.assert_array_equal(g['reaction_weight'][()],
                                      pops.probe_reaction_weight[:, 0])
        assert g['batch_numerator'].shape == (n_b, g['energy'].size)
        # Without a chain, the products are left to the folding code
        assert g.attrs['branching_source'] == b'none'
        assert 'products' not in g

    # A chain gives each reaction's ground and isomeric products
    chain = openmc.deplete.Chain()
    for name, reactions in (
            ('H2', [('(n,gamma)', 'H3', 0.75), ('(n,gamma)', 'H3_m1', 0.25)]),
            ('O16', [('(n,p)', 'N16', 1.0)])):
        nuc = openmc.deplete.Nuclide(name)
        for kind, target, br in reactions:
            nuc.add_reaction(kind, target, 0.0, br)
        chain.add_nuclide(nuc)
    chain.export_to_xml('chain.xml')
    assert pops.reaction_branching('chain.xml') == {
        'H2(n,gamma)': {'H3': 0.75, 'H3_m1': 0.25},
        'O16(n,p)': {'N16': 1.0}}
    pops.export_photon_importance('importance.h5', heavy_water,
                                  chain_file='chain.xml')
    with h5py.File('importance.h5') as f:
        g = f['reactions/H2(n,gamma)']
        assert g.attrs['branching_source'] == b'chain'
        assert list(g['products'][()]) == [b'H3', b'H3_m1']
        np.testing.assert_array_equal(g['branching_ratio'][()], [0.75, 0.25])
        assert list(f['reactions/O16(n,p)/products'][()]) == [b'N16']


def test_probe_reaction_errors(model, run_in_tmpdir):
    pops = model.settings.adjoint_populations
    for reactions, message in ((['Al27(n,gamma)'], 'not in the model'),
                               (['H2(n,a)'], 'has no reaction')):
        pops['photoneutron_probe_reactions'] = reactions
        with pytest.raises(RuntimeError, match=message):
            model.run(**run_kwargs())
    del pops['photoneutron_probe_energies']
    del pops['photoneutron_ray_comb_points']
    pops['photoneutron_probe_reactions'] = ['H2(n,gamma)']
    with pytest.raises(RuntimeError, match='require photoneutron_probe'):
        model.run(**run_kwargs())


def test_restart_with_probe_reactions(model, run_in_tmpdir):
    """A restart continues the probed reactions' sums, and refuses a
    statepoint made with other probed reactions."""
    pops = model.settings.adjoint_populations
    pops['photoneutron_probe_reactions'] = ['H2(n,gamma)']
    model.settings.batches = 6
    model.settings.statepoint = {'batches': [4, 6]}
    model.run(threads=1, **run_kwargs())
    _, expected = read_results('statepoint.6.h5')
    shutil.move('statepoint.6.h5', 'uninterrupted.h5')
    model.run(restart_file='statepoint.4.h5', threads=1, **run_kwargs())
    _, results = read_results('statepoint.6.h5')
    for name in RESULTS:
        np.testing.assert_array_equal(results[name], expected[name],
                                      err_msg=name)

    pops['photoneutron_probe_reactions'] = ['O16(n,p)']
    with pytest.raises(RuntimeError, match='settings differ'):
        model.run(restart_file='statepoint.4.h5', **run_kwargs())
