"""Photoneutron biasing combined with survival biasing.

The forced photoneutron is born with a weight far below the photon weight. It
must be born with that weight as its birth weight, so that survival biasing
with weight normalization does not roulette it at its first collision against
the photon's birth weight, which would undo most of the variance reduction.
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


def leakage(library, survival_biasing):
    """Neutron leakage of a heavy-water sphere around a 5 MeV photon source,
    with photoneutron biasing."""
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
    model.settings.particles = 2000
    model.settings.batches = 5
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.photoneutron_biasing = True
    model.settings.cutoff = {'energy_photon': 2.2e6}
    if survival_biasing:
        model.settings.survival_biasing = True
        model.settings.cutoff['survival_normalization'] = True
    model.settings.source = openmc.IndependentSource(
        particle='photon', energy=openmc.stats.delta_function(5.0e6))

    tally = openmc.Tally(name='leakage')
    tally.filters = [openmc.SurfaceFilter(sph), openmc.ParticleFilter('neutron')]
    tally.scores = ['current']
    model.tallies = [tally]

    with openmc.StatePoint(model.run()) as sp:
        t = sp.get_tally(name='leakage')
        return t.mean.ravel()[0], t.std_dev.ravel()[0]


def test_survival_normalization_keeps_biasing_gain(run_in_tmpdir, library):
    mean_sb, std_sb = leakage(library, survival_biasing=True)
    mean, std = leakage(library, survival_biasing=False)
    assert mean > 0.0 and mean_sb > 0.0

    # Both estimate the same leakage
    assert abs(mean_sb - mean) < 4.0 * (std_sb**2 + std**2)**0.5

    # Rouletting every forced neutron at its first collision made the relative
    # error several times larger than without survival biasing
    assert std_sb / mean_sb < 2.0 * std / mean
