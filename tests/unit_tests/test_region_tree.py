import numpy as np
import pytest
import openmc
import openmc.lib


@pytest.fixture
def many_objects_model():
    """Spheres and finite rods in a box, with the background defined as the
    space outside of all of them. The background region and the universe are
    searched with trees over bounding boxes."""
    rng = np.random.default_rng(1)
    mat = openmc.Material()
    mat.add_nuclide('H1', 1.0)
    mat.set_density('g/cm3', 0.1)

    box = openmc.model.RectangularParallelepiped(
        -10, 10, -10, 10, -10, 10, boundary_type='vacuum')
    objects = []
    # Spheres, as for TRISO particles
    for _ in range(40):
        center = rng.uniform(-8, 8, 3)
        objects.append(-openmc.Sphere(*center, r=rng.uniform(0.3, 1.0)))
    # Finite rods, whose complements are unions
    for _ in range(20):
        x0, y0 = rng.uniform(-8, 8, 2)
        z0 = rng.uniform(-8, 6)
        objects.append(-openmc.ZCylinder(x0, y0, r=rng.uniform(0.2, 0.8)) &
                       +openmc.ZPlane(z0) & -openmc.ZPlane(z0 + 2.0))

    # Objects may overlap, so each cell is the part of an object outside of
    # the preceding objects
    cells = []
    for i, obj in enumerate(objects):
        region = openmc.Intersection([obj])
        for other in objects[:i]:
            region &= ~other
        cells.append(openmc.Cell(fill=mat, region=region))
    background = openmc.Intersection([-box])
    for obj in objects:
        background &= ~obj
    cells.append(openmc.Cell(fill=mat, region=background))

    model = openmc.Model()
    model.geometry = openmc.Geometry(cells)
    model.materials = openmc.Materials([mat])
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 1000
    model.settings.batches = 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-9, -9, -9), (9, 9, 9)))
    return model


def test_find_cell_outside_many_objects(run_in_tmpdir, many_objects_model):
    model = many_objects_model
    model.export_to_model_xml()
    cells = model.geometry.root_universe.cells
    points = np.random.default_rng(2).uniform(-9.9, 9.9, size=(2000, 3))
    openmc.lib.init()
    try:
        for p in points:
            cell, _ = openmc.lib.find_cell(p)
            expected = [c.id for c in cells.values() if tuple(p) in c.region]
            assert [cell.id] == expected
    finally:
        openmc.lib.finalize()


def test_transport_outside_many_objects(run_in_tmpdir, many_objects_model):
    model = many_objects_model
    model.settings.max_lost_particles = 1
    model.run()
