import openmc
import pytest


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_ww_no_split_entering_void(run_in_tmpdir, direction):
    """A particle must not be split by a surface checkpoint when it enters a
    void cell.

    The particle is born in a material cell, crosses a void cell, and enters a
    material cell on the other side. The weight window mesh element containing
    the void has a window below the particle weight. Splitting on entry would
    produce identical copies that are then rouletted when they leave the void,
    so the particle should instead cross the void unchanged.
    """

    # Void-like one-group material for the cells on either side of the void
    # so that the only weight window checks happen at surface crossings
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata('void', groups)
    xsdata.order = 0
    xsdata.set_total([0.0])
    xsdata.set_absorption([0.0])
    xsdata.set_scatter_matrix([[[0.0]]])
    mg_library = openmc.MGXSLibrary(groups)
    mg_library.add_xsdata(xsdata)
    mg_library.export_to_hdf5('mgxs.h5')

    mat = openmc.Material()
    mat.add_macroscopic('void')
    materials = openmc.Materials([mat])
    materials.cross_sections = 'mgxs.h5'

    # Material cells on either side of a void cell spanning -1 < x < 1
    x0 = openmc.XPlane(-2.0, boundary_type='vacuum')
    x1 = openmc.XPlane(-1.0)
    x2 = openmc.XPlane(1.0)
    x3 = openmc.XPlane(2.0, boundary_type='vacuum')
    yz = openmc.model.RectangularPrism(2.0, 2.0, axis='x',
                                       boundary_type='vacuum')
    left = openmc.Cell(fill=mat, region=+x0 & -x1 & -yz)
    void = openmc.Cell(region=+x1 & -x2 & -yz)
    right = openmc.Cell(fill=mat, region=+x2 & -x3 & -yz)
    geometry = openmc.Geometry([left, void, right])

    # The window of the mesh element containing the void is below the
    # particle weight. Mesh planes are offset from the void surfaces so that
    # each surface checkpoint lies inside a single mesh element.
    mesh = openmc.RectilinearMesh()
    mesh.x_grid = [-2.0, -1.0 - 0.1*direction, 1.0 - 0.1*direction, 2.0]
    mesh.y_grid = [-1.0, 1.0]
    mesh.z_grid = [-1.0, 1.0]
    ww = openmc.WeightWindows(mesh, lower_ww_bounds=[0.5, 0.1, 0.5],
                              upper_ww_bounds=[1.5, 0.2, 1.5])

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.particles = 10
    settings.batches = 1
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((-1.5*direction, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((direction, 0.0, 0.0)),
    )
    settings.weight_windows = ww
    settings.weight_window_checkpoints = {'collision': False, 'surface': True}

    exit_cell = right if direction > 0 else left
    tally = openmc.Tally()
    tally.filters = [
        openmc.CellFilter([void, exit_cell]),
        openmc.WeightFilter([0.0, 0.5, 2.0])
    ]
    tally.scores = ['flux']

    model = openmc.Model(geometry, materials, settings, openmc.Tallies([tally]))
    model.run(apply_tally_results=True)

    # The particle crosses the 2 cm void and the 1 cm exit cell with its
    # original weight
    flux = tally.mean.reshape(2, 2)
    assert flux[0] == pytest.approx([0.0, 2.0])
    assert flux[1] == pytest.approx([0.0, 1.0])
