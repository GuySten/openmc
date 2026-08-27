"""Test the super-history method to compute adjoint-weighted
kinetics parameters using dedicated tallies."""

import openmc
import pytest

from tests.testing_harness import PyAPITestHarness

@pytest.fixture()
def simple_model():
    model = openmc.Model()

    # Material
    material = openmc.Material(name="core")
    material.add_nuclide("U235", 1.0)
    material.set_density('g/cm3', 16.0)

    # Geometry
    radius = 10.0
    sphere = openmc.Sphere(r=radius, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)
    model.geometry = openmc.Geometry([cell])

    # Settings
    model.settings.particles = 1000
    model.settings.batches = 20
    model.settings.inactive = 5
    model.settings.superhistory_n_generation = 5

    space = openmc.stats.Box(*cell.bounding_box)
    model.settings.source = openmc.IndependentSource(
        space=space, constraints={'fissionable': True})

    # Tally IFP scores
    tally = openmc.Tally(name="superhistory-scores")
    tally.adjoint = True
    tally.scores = ["inverse-velocity", "nu-fission", "delayed-nu-fission"]
        
    model.tallies = [tally]

    return model


def test_superhistory_method(simple_model):
    harness = PyAPITestHarness("statepoint.20.h5", model=simple_model)
    harness.main()
