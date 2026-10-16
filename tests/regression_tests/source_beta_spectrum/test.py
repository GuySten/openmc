import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture
def model():
    # Sr90/Y90 source in a steel capsule surrounded by water
    source_material = openmc.Material()
    source_material.add_element('Sr', 1.0)
    source_material.add_element('O', 1.0)
    source_material.set_density('g/cm3', 4.7)
    steel = openmc.Material()
    steel.add_element('Fe', 0.7, 'wo')
    steel.add_element('Cr', 0.19, 'wo')
    steel.add_element('Ni', 0.11, 'wo')
    steel.set_density('g/cm3', 8.0)
    water = openmc.Material()
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', 1.0)

    pellet_sphere = openmc.Sphere(r=0.1)
    capsule_sphere = openmc.Sphere(r=0.15)
    water_sphere = openmc.Sphere(r=5.0, boundary_type='vacuum')
    pellet = openmc.Cell(fill=source_material, region=-pellet_sphere)
    capsule = openmc.Cell(fill=steel, region=+pellet_sphere & -capsule_sphere)
    phantom = openmc.Cell(fill=water, region=+capsule_sphere & -water_sphere)

    # Unique first-forbidden beta spectra of Sr90 and Y90, and an allowed
    # positron spectrum
    sr90 = openmc.stats.BetaSpectrum(546.0e3, 39, 90, forbiddenness=1,
                                     unique=True)
    y90 = openmc.stats.BetaSpectrum(2279.8e3, 40, 90, forbiddenness=1,
                                    unique=True)
    electrons = openmc.IndependentSource(
        space=openmc.stats.spherical_uniform(r_outer=0.1),
        energy=openmc.stats.Mixture([1.0, 1.0], [sr90, y90]),
        particle='electron',
        strength=2.0
    )
    positrons = openmc.IndependentSource(
        space=openmc.stats.spherical_uniform(r_outer=0.1),
        energy=openmc.stats.BetaSpectrum(545.7e3, 10, 22, positron=True),
        particle='positron'
    )

    model = openmc.Model()
    model.geometry = openmc.Geometry([pellet, capsule, phantom])
    model.settings.run_mode = 'fixed source'
    model.settings.source = [electrons, positrons]
    model.settings.particles = 1000
    model.settings.batches = 2
    model.settings.photon_transport = True

    heating = openmc.Tally()
    heating.filters = [
        openmc.ParticleFilter(['electron', 'positron', 'photon']),
        openmc.CellFilter([pellet, capsule, phantom])
    ]
    heating.scores = ['heating']
    spectrum = openmc.Tally()
    spectrum.filters = [
        openmc.ParticleFilter(['photon']),
        openmc.EnergyFilter([0.0, 0.1e6, 0.5e6, 0.511e6, 1.0e6, 2.5e6])
    ]
    spectrum.scores = ['flux']
    model.tallies = [heating, spectrum]
    return model


def test_source_beta_spectrum(model):
    harness = PyAPITestHarness('statepoint.2.h5', model)
    harness.main()
