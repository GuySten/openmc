"""Weight windows for forced photoneutrons.

A forced photoneutron is born with a weight far below that of the photon. It
must still see the weight windows the photon's history is normalized to, so
that neutron weight windows roulette it where it is unimportant, rather than
windows scaled down by its own small weight that never act on it.
"""

import os

import numpy as np
import pytest
import openmc


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    """Cross section library with photonuclear data for H2, or skip."""
    xs_path = openmc.config.get('cross_sections')
    if xs_path is None:
        pytest.skip('No cross section library configured')
    lib = openmc.data.DataLibrary.from_xml(xs_path)
    if lib.get_by_material('H2', data_type='photonuclear') is None:
        path = os.path.join(os.path.dirname(xs_path), 'photonuclear', 'H2.h5')
        if not os.path.isfile(path):
            pytest.skip('No photonuclear data for H2')
        lib.register_file(path)
    out = tmp_path_factory.mktemp('library') / 'cross_sections.xml'
    lib.export_to_xml(out)
    return str(out)


def run(library, weight_windows):
    """Neutron leakage and number of neutron events in a heavy-water sphere
    around a 5 MeV photon source, with photoneutron biasing."""
    model = openmc.Model()
    d2o = openmc.Material()
    d2o.add_nuclide('H2', 2.0)
    d2o.add_nuclide('O16', 1.0)
    d2o.set_density('g/cm3', 1.1)
    model.materials = openmc.Materials([d2o])
    model.materials.cross_sections = library

    sph = openmc.Sphere(r=20.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=d2o, region=-sph)
    model.geometry = openmc.Geometry([cell])

    model.settings.run_mode = 'fixed source'
    model.settings.particles = 4000
    model.settings.batches = 5
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.photoneutron_biasing = True
    model.settings.cutoff = {'energy_photon': 2.2e6}
    model.settings.source = openmc.IndependentSource(
        particle='photon', energy=openmc.stats.delta_function(5.0e6))

    if weight_windows:
        # Neutron windows far above the forced photoneutron weights, so that
        # they roulette nearly every forced photoneutron at birth
        mesh = openmc.RegularMesh()
        mesh.lower_left = (-20.0, -20.0, -20.0)
        mesh.upper_right = (20.0, 20.0, 20.0)
        mesh.dimension = (1, 1, 1)
        model.settings.weight_windows = openmc.WeightWindows(
            mesh, lower_ww_bounds=[0.5], upper_bound_ratio=5.0,
            particle_type='neutron')

    leakage = openmc.Tally(name='leakage')
    leakage.filters = [openmc.SurfaceFilter(sph),
                       openmc.ParticleFilter('neutron')]
    leakage.scores = ['current']
    events = openmc.Tally(name='events')
    events.filters = [openmc.ParticleFilter('neutron')]
    events.scores = ['events']
    model.tallies = [leakage, events]

    with openmc.StatePoint(model.run()) as sp:
        t = sp.get_tally(name='leakage')
        n_events = sp.get_tally(name='events').mean.ravel()[0]
        return t.mean.ravel()[0], t.std_dev.ravel()[0], n_events


def test_weight_windows_act_on_forced_photoneutrons(run_in_tmpdir, library):
    mean_ww, std_ww, events_ww = run(library, weight_windows=True)
    mean, std, events = run(library, weight_windows=False)
    assert mean > 0.0 and mean_ww > 0.0

    # Both estimate the same leakage
    assert abs(mean_ww - mean) < 4.0 * np.hypot(std_ww, std)

    # The neutron windows roulette most forced photoneutrons, which are
    # otherwise all transported
    assert events_ww < 0.2 * events
