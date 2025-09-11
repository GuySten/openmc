import openmc
import pytest
import numpy as np
import os

from tests.testing_harness import PyAPITestHarness

@pytest.fixture
def model():
    model = openmc.Model()
    
    mat = openmc.Material()
    mat.set_density('g/cm3', 10.0)
    mat.add_nuclide('H1', 1.0)
    
    # A single sphere
    s1 = openmc.Sphere(r=100, boundary_type='reflective')
    sphere = openmc.Cell()
    sphere.region = -s1
    sphere.fill = mat
    model.geometry = openmc.Geometry([sphere])

    # Set the running parameters
    settings_file = openmc.Settings()
    settings_file.cutoff = {'time_neutron': 10.0}
    settings_file.run_mode = 'fixed source'
    settings_file.batches = 10
    settings_file.particles = 1000
    settings_file.source = openmc.IndependentSource(
        space=openmc.stats.Point(), energy=openmc.stats.Discrete([1e6], [1]))
    model.settings = settings_file

    # Tally flux under time cutoff
    tallies = openmc.Tallies()
    tally = openmc.Tally()
    tally.scores = ['flux']
    energy_filter = openmc.EnergyFilter(np.linspace(0, 4*0.0253, 1000))
    tally.filters = [energy_filter]
    tallies.append(tally)
    model.tallies = tallies

    return model

@pytest.fixture
def cross_sections():
    library = openmc.data.DataLibrary.from_xml(openmc.config.get('cross_sections'))
    path = library.get_by_material('H1', data_type='neutron')['path']
    data = openmc.data.IncidentNeutron.from_hdf5(path)
    for mt in list(data.reactions):
      if mt!=2:
        del data.reactions[mt]
    data.export_to_hdf5("H1.h5")
    new_lib = openmc.data.DataLibrary()
    new_lib.register_file("H1.h5")
    new_lib.export_to_xml("cross_sections.xml")
    yield "cross_sections.xml"
    os.remove("cross_sections.xml")
    os.remove("H1.h5")

def test_thermal_equilibrium(cross_sections, model):
    model.materials.cross_sections = cross_sections
    harness = PyAPITestHarness('statepoint.10.h5', model)
    harness.main()
