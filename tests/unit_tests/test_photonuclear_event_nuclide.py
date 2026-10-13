"""A photonuclear collision is attributed to the nuclide it occurred on."""

import os

import h5py
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


def test_photonuclear_event_nuclide(run_in_tmpdir, library):
    # Heavy water around a 5 MeV photon source. Below the 15.7 MeV threshold
    # of O16, every photonuclear collision is D(g,n)p (MT 50) on H2, while the
    # photon's other collisions are shared between H2 and O16.
    model = openmc.Model()
    d2o = openmc.Material()
    d2o.add_nuclide('H2', 2.0)
    d2o.add_nuclide('O16', 1.0)
    d2o.set_density('g/cm3', 1.1)
    model.materials = openmc.Materials([d2o])
    model.materials.cross_sections = library

    sph = openmc.Sphere(r=20.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([openmc.Cell(fill=d2o, region=-sph)])

    model.settings.run_mode = 'fixed source'
    model.settings.particles = 10000
    model.settings.batches = 2
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.cutoff = {'energy_photon': 2.2e6}
    model.settings.source = openmc.IndependentSource(
        particle='photon', energy=openmc.stats.delta_function(5.0e6))
    model.settings.collision_track = {'reactions': [50],
                                      'max_collisions': 100000}
    model.run()

    with h5py.File('collision_track.h5', 'r') as f:
        sites = f['collision_track_bank'][:]
    photon = openmc.ParticleType.PHOTON.pdg_number
    photonuclear = sites[(sites['particle'] == photon) & (sites['event_mt'] == 50)]
    assert len(photonuclear) > 0

    h2 = openmc.ParticleType('H2').pdg_number
    assert np.all(photonuclear['nuclide_id'] == h2)
