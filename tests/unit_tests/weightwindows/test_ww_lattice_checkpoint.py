import openmc
import pytest


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_ww_surface_checkpoint_lattice(run_in_tmpdir, direction):
    """Weight windows must be applied at the surface checkpoint when a particle
    crosses from one lattice element to the next.

    The particle is born in one element of a two-element lattice and moves
    toward the other. A weight window mesh plane lies inside the birth element,
    so that the element entered at the lattice boundary (x = 0) is entirely in
    the mesh element with the lower window. The only checkpoint at which the
    particle can be split is therefore the lattice crossing.
    """

    # Void-like one-group material so the only weight window checks happen at
    # surface/lattice crossings
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

    # Two-element lattice along x with its interior boundary at x = 0
    pin = openmc.Universe(cells=[openmc.Cell(fill=mat)])
    lattice = openmc.RectLattice()
    lattice.lower_left = (-1.0, -1.0, -1.0)
    lattice.pitch = (1.0, 2.0, 2.0)
    lattice.universes = [[[pin, pin]]]
    box = openmc.model.RectangularParallelepiped(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, boundary_type='vacuum')
    geometry = openmc.Geometry([openmc.Cell(fill=lattice, region=-box)])

    # The weight window mesh plane lies inside the birth element. The particle
    # is born in the window of its birth mesh element and must be split five
    # ways when it enters the other lattice element.
    mesh = openmc.RectilinearMesh()
    mesh.x_grid = [-1.0, -0.1*direction, 1.0]
    mesh.y_grid = [-1.0, 1.0]
    mesh.z_grid = [-1.0, 1.0]
    birth_bounds = (0.5, 1.5)
    target_bounds = (0.1, 0.2)
    if direction > 0:
        lower = [birth_bounds[0], target_bounds[0]]
        upper = [birth_bounds[1], target_bounds[1]]
    else:
        lower = [target_bounds[0], birth_bounds[0]]
        upper = [target_bounds[1], birth_bounds[1]]
    ww = openmc.WeightWindows(mesh, lower_ww_bounds=lower,
                              upper_ww_bounds=upper)

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.particles = 10
    settings.batches = 1
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((-0.5*direction, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((direction, 0.0, 0.0)),
    )
    settings.weight_windows = ww
    settings.weight_window_checkpoints = {'collision': False, 'surface': True}

    # Tally flux in each lattice element, binned by particle weight
    tally_mesh = openmc.RegularMesh()
    tally_mesh.lower_left = (-1.0, -1.0, -1.0)
    tally_mesh.upper_right = (1.0, 1.0, 1.0)
    tally_mesh.dimension = (2, 1, 1)
    tally = openmc.Tally()
    tally.filters = [
        openmc.MeshFilter(tally_mesh),
        openmc.WeightFilter([0.0, 0.5, 2.0])
    ]
    tally.scores = ['flux']

    model = openmc.Model(geometry, materials, settings, openmc.Tallies([tally]))
    model.run(apply_tally_results=True)

    # In the birth element the particle travels 0.5 cm with weight 1. All flux
    # in the element entered should be carried by split particles with weight
    # 0.2; total flux is conserved (1 cm path length per source particle).
    flux = tally.mean.reshape(2, 2)
    target = 1 if direction > 0 else 0
    assert flux[1 - target] == pytest.approx([0.0, 0.5])
    assert flux[target] == pytest.approx([1.0, 0.0])
