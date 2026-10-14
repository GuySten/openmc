"""Detector trees of the adjoint side populations: depth-0 trees grown from
the roots, scoring detector tallies binned by AdjointRootFilter."""

import shutil

import h5py
import numpy as np
import openmc
import pytest

from tests.regression_tests import config


def run_kwargs(**kwargs):
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    return kwargs


def detector_model(photoneutrons=False):
    """A U235 sphere in a water (or heavy-water) reflector, whose outer shell
    is the detector. Returns the model, the detector tally and the analog
    tally of the same detector."""
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

    r_det = 14.0 if photoneutrons else 10.0
    core = openmc.Sphere(r=6.0)
    inner = openmc.Sphere(r=r_det)
    outer = openmc.Sphere(r=r_det + 2.0, boundary_type='vacuum')
    detector = openmc.Cell(fill=water, region=+inner & -outer)
    model.geometry = openmc.Geometry([
        openmc.Cell(fill=fuel, region=-core),
        openmc.Cell(fill=water, region=+core & -inner),
        detector,
    ])

    model.settings.particles = 2000
    model.settings.batches = 25 if not photoneutrons else 12
    model.settings.inactive = 5 if not photoneutrons else 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    # Origins binned in the two halves of the core
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-6.0, -6.0, -6.0)
    mesh.upper_right = (6.0, 6.0, 6.0)
    mesh.dimension = (2, 1, 1)
    det = openmc.Tally(name='detector')
    det.filters = [openmc.CellFilter([detector]),
                   openmc.AdjointRootFilter(None if photoneutrons else mesh)]
    det.scores = ['flux']
    analog = openmc.Tally(name='analog')
    analog.filters = [openmc.CellFilter([detector])]
    analog.scores = ['flux']
    model.tallies = [det, analog]

    adjoint_populations = {'n_generation': 1, 'root_fraction': 0.5,
                           'detector_tallies': [det]}
    if photoneutrons:
        model.settings.photon_transport = True
        model.settings.photonuclear_physics = True
        adjoint_populations.update({
            'photoneutrons': True,
            'photoneutron_probe_energies': [2.4e6, 2.8e6, 3.2e6],
            'photoneutron_ray_comb_points': 3,
            'photoneutron_ray_probe_check': True,
            'photoneutron_probe_fraction': 20.0,
            'photoneutron_probe_root_fraction': 1.0,
            'photoneutron_ray_root_fraction': 1.0,
        })
        model.tallies = [det]
    model.settings.adjoint_populations = adjoint_populations
    return model, det, analog


def test_settings_roundtrip(run_in_tmpdir):
    tally = openmc.Tally(tally_id=7)
    s = openmc.Settings()
    s.adjoint_populations = {'n_generation': 2,
                             'detector_tallies': [tally, 9]}
    assert s.adjoint_populations['detector_tallies'] == [7, 9]
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == \
        s.adjoint_populations

    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 2, 'detector_tallies': []}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 2,
                                 'detector_tallies': [3, 3]}


def test_filter_roundtrip(run_in_tmpdir):
    mesh = openmc.RegularMesh(mesh_id=4)
    mesh.lower_left = (-1.0, -1.0, -1.0)
    mesh.upper_right = (1.0, 1.0, 1.0)
    mesh.dimension = (2, 2, 1)
    with_mesh = openmc.Tally(tally_id=1)
    with_mesh.filters = [openmc.AdjointRootFilter(mesh, filter_id=11)]
    with_mesh.scores = ['flux']
    without = openmc.Tally(tally_id=2)
    without.filters = [openmc.AdjointRootFilter(filter_id=12)]
    without.scores = ['flux']
    openmc.Tallies([with_mesh, without]).export_to_xml()

    tallies = openmc.Tallies.from_xml()
    f1, f2 = tallies[0].filters[0], tallies[1].filters[0]
    assert isinstance(f1, openmc.AdjointRootFilter)
    assert f1.mesh.id == 4 and f1.bins == [4]
    assert f2.mesh is None and f2.bins == []
    with pytest.raises(ValueError):
        f1.num_bins


def test_detector_trees_match_analog(run_in_tmpdir):
    """Each neutron history of the driver ends at fission, so its tally of
    the detector is the non-multiplying response to one generation's fission
    neutrons. The detector trees' responses per unit source, prompt and per
    delayed group, folded with the generation's source weights, must
    estimate the same."""
    model, det, analog = detector_model()
    sp_path = model.run(**run_kwargs())

    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='detector')
        a = sp.get_tally(name='analog')
        pops = sp.adjoint_populations
        n_cat = pops.detector_categories[3]
        assert tally.filters[1].num_bins == 2 * n_cat
        g = tally.mean.reshape(n_cat, 2)
        sg = tally.std_dev.reshape(n_cat, 2)
        source = pops.detector_source.mean(axis=0)
        assert pops.n_detector_histories > 0

    # Prompt and the six U235 delayed groups respond; nothing else has a
    # source
    assert np.all(g[:7] > 0.0)
    assert np.all(source[:7] > 0.0)
    assert np.all(g[7:] == 0.0) and np.all(source[7:] == 0.0)

    n = model.settings.particles
    folded = (g * source).sum() / n
    sigma = np.hypot(np.sqrt(((sg * source) ** 2).sum()) / n,
                     a.std_dev.ravel()[0])
    assert abs(folded - a.mean.ravel()[0]) < 4.0 * sigma


def test_ray_trees_match_probe_trees(run_in_tmpdir):
    """With the ray check on, the probes' uncollided photoneutrons grow
    detector trees as well; per line, their response must agree with the
    rays', whose shared trees are spread over the lines by their factors."""
    model, det, _ = detector_model(photoneutrons=True)
    sp_path = model.run(**run_kwargs())

    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='detector')
        m = tally.mean.ravel()
        sd = tally.std_dev.ravel()
        off_pn, off_probe, off_ray, n_cat = \
            sp.adjoint_populations.detector_categories

    assert tally.filters[1].num_bins == n_cat
    for k in range(3):
        probe = off_probe + 3 * k
        ray = off_ray + k
        assert m[ray] > 0.0 and m[probe] > 0.0
        assert abs(m[probe] - m[ray]) < 4.0 * np.hypot(sd[probe], sd[ray])


def test_restart(run_in_tmpdir):
    """A restarted run continues the detector tallies and source sums."""
    model, det, _ = detector_model()
    model.settings.batches = 10
    model.settings.statepoint = {'batches': [8, 10]}
    model.run(threads=1, **run_kwargs())
    shutil.move('statepoint.10.h5', 'uninterrupted.h5')
    model.run(restart_file='statepoint.08.h5', threads=1, **run_kwargs())

    results = []
    for path in ('uninterrupted.h5', 'statepoint.10.h5'):
        with openmc.StatePoint(path) as sp:
            results.append((sp.get_tally(name='detector').mean,
                            sp.adjoint_populations.detector_source))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_filter_needs_detector_tally(run_in_tmpdir):
    model, det, _ = detector_model()
    model.settings.adjoint_populations = {'n_generation': 1}
    with pytest.raises(RuntimeError, match='not a detector tally'):
        model.run()


def test_detector_tally_needs_filter(run_in_tmpdir):
    model, det, _ = detector_model()
    det.filters = det.filters[:1]
    with pytest.raises(RuntimeError, match='needs an adjoint root filter'):
        model.run()


def test_detector_tallies_share_mesh(run_in_tmpdir):
    model, det, analog = detector_model()
    other = openmc.Tally(name='other')
    other.filters = [openmc.AdjointRootFilter()]
    other.scores = ['flux']
    model.tallies.append(other)
    model.settings.adjoint_populations = {
        **model.settings.adjoint_populations,
        'detector_tallies': [det, other]}
    with pytest.raises(RuntimeError, match='same mesh'):
        model.run()


def radial_weight_windows(radius, n, decay):
    """Weight windows on a cube of n^3 elements whose lower bounds fall as
    exp(-r / decay): importance grows towards the detector shell."""
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-radius,) * 3
    mesh.upper_right = (radius,) * 3
    mesh.dimension = (n,) * 3
    half = radius / n
    c = np.linspace(-radius + half, radius - half, n)
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    r = np.minimum(np.sqrt(x**2 + y**2 + z**2), radius - half)
    lower = (0.5 * np.exp(-r / decay)).transpose(2, 1, 0).ravel()
    return openmc.WeightWindows(mesh, lower_ww_bounds=lower,
                                upper_bound_ratio=5.0,
                                energy_bounds=[0.0, 2.0e7],
                                particle_type='neutron')


def test_weight_windows_act_in_detector_trees_only(run_in_tmpdir):
    """Weight windows act only in the detector trees: the driver and the
    kinetics trees, and every result they make, are the same to the bit with
    and without them, while the detector trees are split and stay
    unbiased."""
    results = {}
    for weight_windows_on in (False, True):
        model, det, analog = detector_model()
        if weight_windows_on:
            model.settings.weight_windows = radial_weight_windows(12.0, 6, 6.0)
            model.settings.weight_windows_on = True
        sp_path = model.run(threads=1, **run_kwargs())
        with openmc.StatePoint(sp_path) as sp:
            tally = sp.get_tally(name='detector')
            pops = sp.adjoint_populations
            n_cat = pops.detector_categories[3]
            results[weight_windows_on] = dict(
                keff=sp.keff.nominal_value,
                analog=sp.get_tally(name='analog').mean.copy(),
                analog_sd=sp.get_tally(name='analog').std_dev.copy(),
                weight=pops.weight.copy(),
                source=pops.detector_source.copy(),
                g=tally.mean.reshape(n_cat, 2).copy(),
                sg=tally.std_dev.reshape(n_cat, 2).copy(),
                histories=pops.n_detector_histories,
            )

    off, on = results[False], results[True]
    assert on['keff'] == off['keff']
    np.testing.assert_array_equal(on['analog'], off['analog'])
    np.testing.assert_array_equal(on['weight'], off['weight'])
    np.testing.assert_array_equal(on['source'], off['source'])
    assert on['histories'] > off['histories']

    n = 2000
    s = on['source'].mean(axis=0)
    folded = (on['g'] * s).sum() / n
    sigma = np.hypot(np.sqrt(((on['sg'] * s) ** 2).sum()) / n,
                     on['analog_sd'].ravel()[0])
    assert abs(folded - on['analog'].ravel()[0]) < 4.0 * sigma


def test_weight_windows_need_detector_tallies(run_in_tmpdir):
    model, det, analog = detector_model()
    model.tallies = [analog]
    model.settings.adjoint_populations = {'n_generation': 1}
    model.settings.weight_windows = radial_weight_windows(12.0, 6, 6.0)
    model.settings.weight_windows_on = True
    with pytest.raises(RuntimeError, match='needs\\s+.detector_tallies'):
        model.run()
