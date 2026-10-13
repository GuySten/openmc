"""Test that particles split by weight windows keep the birth state of the
particle they were split from in a transport run."""

import numpy as np
import openmc
import pytest

from tests.regression_tests import config


@pytest.fixture
def model():
    """Neutrons born at x = -3 cm in the left half of a water box. Weight
    windows only apply to the right half, where they split every neutron that
    enters it."""
    model = openmc.Model()

    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    box = openmc.model.RectangularParallelepiped(
        -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, boundary_type='vacuum')
    mid = openmc.XPlane(0.0)
    left = openmc.Cell(region=-box & -mid, fill=water)
    right = openmc.Cell(region=-box & +mid, fill=water)
    model.geometry = openmc.Geometry([left, right])

    model.settings.run_mode = 'fixed source'
    model.settings.particles = 200
    model.settings.batches = 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((-3.0, 0.0, 0.0)),
        energy=openmc.stats.delta_function(1.0e6))

    ww_mesh = openmc.RegularMesh()
    ww_mesh.lower_left = (-10.0, -10.0, -10.0)
    ww_mesh.upper_right = (10.0, 10.0, 10.0)
    ww_mesh.dimension = (2, 1, 1)
    model.settings.weight_windows = openmc.WeightWindows(
        ww_mesh,
        lower_ww_bounds=[-1.0, 0.05],
        upper_ww_bounds=[-1.0, 0.1],
        energy_bounds=[0.0, 2.0e7],
        particle_type='neutron',
    )
    model.settings.weight_windows_on = True

    # Birth mesh with the source in bin 1 of 4 along x
    born_mesh = openmc.RegularMesh()
    born_mesh.lower_left = (-10.0, -10.0, -10.0)
    born_mesh.upper_right = (10.0, 10.0, 10.0)
    born_mesh.dimension = (4, 1, 1)

    cell_born = openmc.Tally(name='cell born')
    cell_born.filters = [openmc.CellBornFilter([left, right])]
    cell_born.scores = ['events']
    mesh_born = openmc.Tally(name='mesh born')
    mesh_born.filters = [openmc.MeshBornFilter(born_mesh)]
    mesh_born.scores = ['events']
    split = openmc.Tally(name='split')
    split.filters = [openmc.CellFilter([right])]
    split.scores = ['events']
    model.tallies = [cell_born, mesh_born, split]
    return model


@pytest.mark.parametrize('shared_secondary_bank', [False, True],
                         ids=['local_bank', 'shared_bank'])
def test_split_particles_keep_birth_state(
        model, shared_secondary_bank, run_in_tmpdir):
    """Every neutron continues the history of a source neutron born in the
    left cell and in birth mesh bin 1, so no event may be scored with another
    birth cell or birth mesh bin, although split neutrons collide in the right
    cell and elsewhere in the mesh."""
    model.settings.shared_secondary_bank = shared_secondary_bank
    sp_path = model.run(event_based=config['event'])

    with openmc.StatePoint(sp_path) as sp:
        cell_born = sp.get_tally(name='cell born').sum.ravel()
        mesh_born = sp.get_tally(name='mesh born').sum.ravel()
        split = sp.get_tally(name='split').sum.ravel()

    # Neutrons are split in the right cell
    assert split[0] > 0.0

    assert cell_born[0] > 0.0
    assert cell_born[1] == 0.0
    np.testing.assert_array_equal(mesh_born[[0, 2, 3]], 0.0)
    assert mesh_born[1] == cell_born[0]


def test_derivatives_refused_with_weight_windows(model, run_in_tmpdir):
    """Split neutrons start with zero flux derivatives, so tally derivatives
    are refused when weight windows are on."""
    deriv = openmc.TallyDerivative(
        variable='density',
        material=next(iter(model.geometry.get_all_materials())))
    tally = openmc.Tally()
    tally.scores = ['flux']
    tally.derivative = deriv
    model.tallies.append(tally)

    with pytest.raises(RuntimeError, match='not supported with weight windows'):
        model.run()
