"""Photonuclear physics must be refused in an eigenvalue calculation."""

import pytest
import openmc


def test_eigenvalue_refused(run_in_tmpdir):
    model = openmc.Model()
    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)
    model.materials = openmc.Materials([water])

    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([openmc.Cell(fill=water, region=-sph)])

    model.settings.run_mode = 'eigenvalue'
    model.settings.particles = 100
    model.settings.batches = 2
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True

    # The settings are checked before any nuclear data is read
    with pytest.raises(RuntimeError, match='not supported in an eigenvalue'):
        model.run()
