"""Heating with photoneutron biasing when photoneutrons fall below the neutron
energy cutoff.

A photoneutron below the cutoff is not banked, and its energy is deposited
locally, as in an analog run. The forced photoneutron must then remove none of
the energy from the local deposition, so that heating agrees with the analog
treatment.
"""

import os

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


def heating(library, biasing):
    """Heating in a heavy-water sphere around a 3 MeV photon source. D(g,n)
    photoneutrons are then below about 0.8 MeV, all under the 1 MeV neutron
    energy cutoff."""
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
    model.settings.particles = 5000
    model.settings.batches = 5
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.photoneutron_biasing = biasing
    model.settings.cutoff = {'energy_neutron': 1.0e6}
    model.settings.source = openmc.IndependentSource(
        particle='photon', energy=openmc.stats.delta_function(3.0e6))

    tally = openmc.Tally(name='heating')
    tally.filters = [openmc.CellFilter(cell)]
    tally.scores = ['heating']
    model.tallies = [tally]

    with openmc.StatePoint(model.run()) as sp:
        t = sp.get_tally(name='heating')
        return t.mean.ravel()[0], t.std_dev.ravel()[0]


def test_heating_matches_analog_below_cutoff(run_in_tmpdir, library):
    mean_b, std_b = heating(library, biasing=True)
    mean_a, std_a = heating(library, biasing=False)
    assert abs(mean_b - mean_a) < 4.0 * (std_b**2 + std_a**2)**0.5
