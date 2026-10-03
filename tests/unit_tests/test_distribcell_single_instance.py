import pytest

import openmc
import openmc.lib


def test_single_instance_universes(run_in_tmpdir):
    """Cells in universes used once have instance 0, while cells in a
    universe used many times keep distinct instances and per-instance
    temperatures."""
    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    # A universe used in several lattice elements, with a temperature per
    # instance, and universes that are each used in one lattice element
    shared_cell = openmc.Cell(fill=water)
    shared = openmc.Universe(cells=[shared_cell])
    unique_cells = []
    universes = []
    for i in range(3):
        row = []
        for j in range(3):
            if (i + j) % 2 == 0:
                row.append(shared)
            else:
                cell = openmc.Cell(fill=water)
                unique_cells.append(cell)
                row.append(openmc.Universe(cells=[cell]))
        universes.append(row)
    n_shared = 5
    shared_cell.temperature = [300.0 + 10.0*k for k in range(n_shared)]

    lattice = openmc.RectLattice()
    lattice.lower_left = (-1.5, -1.5)
    lattice.pitch = (1.0, 1.0)
    lattice.universes = universes
    box = openmc.model.RectangularParallelepiped(
        -1.5, 1.5, -1.5, 1.5, -1.0, 1.0, boundary_type='vacuum')
    root = openmc.Universe(cells=[openmc.Cell(fill=lattice, region=-box)])

    model = openmc.Model()
    model.geometry = openmc.Geometry(root)
    model.materials = openmc.Materials([water])
    model.settings.particles = 10
    model.settings.batches = 1
    model.settings.run_mode = 'fixed source'
    model.settings.temperature = {'method': 'nearest', 'tolerance': 1000.0}
    model.export_to_model_xml()

    with openmc.lib.run_in_memory():
        shared_instances = []
        for i in range(3):
            for j in range(3):
                # Lattice element (i, j) has row i from the top
                point = (-1.0 + j, 1.0 - i, 0.0)
                cell, instance = openmc.lib.find_cell(point)
                if cell.id == shared_cell.id:
                    shared_instances.append(instance)
                    assert cell.get_temperature(instance) == \
                        pytest.approx(shared_cell.temperature[instance])
                else:
                    assert instance == 0
        assert sorted(shared_instances) == list(range(n_shared))
        for cell in unique_cells:
            assert openmc.lib.cells[cell.id].num_instances == 1
