from pathlib import Path

import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture
def model():
    model = openmc.Model()

    # Fuel whose fission products decay and nickel cladding whose activation
    # products decay
    fuel = openmc.Material()
    fuel.set_density('g/cm3', 10.0)
    fuel.add_nuclide('U235', 0.05)
    fuel.add_nuclide('U238', 0.95)
    clad = openmc.Material()
    clad.set_density('g/cm3', 8.9)
    clad.add_nuclide('Ni58', 0.68)
    clad.add_nuclide('Ni60', 0.26)
    clad.add_nuclide('Ni62', 0.04)
    clad.add_nuclide('Ni64', 0.02)
    water = openmc.Material()
    water.set_density('g/cm3', 1.0)
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)

    fuel_or = openmc.ZCylinder(r=0.4)
    clad_or = openmc.ZCylinder(r=0.5)
    box = openmc.model.RectangularPrism(1.3, 1.3, boundary_type='reflective')
    fuel_cell = openmc.Cell(fill=fuel, region=-fuel_or)
    clad_cell = openmc.Cell(fill=clad, region=+fuel_or & -clad_or)
    water_cell = openmc.Cell(fill=water, region=+clad_or & -box)
    model.geometry = openmc.Geometry([fuel_cell, clad_cell, water_cell])

    # Without photon transport, decay energy is deposited locally
    model.settings.particles = 1000
    model.settings.batches = 5
    model.settings.inactive = 2
    model.settings.decay_sources = {'irradiation_time': (30, 'd')}

    heating = openmc.Tally()
    heating.filters = [openmc.CellFilter([fuel_cell, clad_cell, water_cell])]
    heating.scores = ['heating']
    model.tallies = [heating]
    return model


def test_decay_sources_local(model):
    chain = Path(__file__).parent.parent / 'decay_sources' / 'chain.xml'
    with openmc.config.patch('chain_file', chain):
        harness = PyAPITestHarness('statepoint.5.h5', model)
        harness.main()
