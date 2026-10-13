"""Photonuclear settings are checked at export, and statepoints written
without the photonuclear attributes can still be read."""

import h5py
import pytest
import openmc


def test_photonuclear_physics_needs_photon_transport(run_in_tmpdir):
    settings = openmc.Settings()
    settings.photonuclear_physics = True
    with pytest.raises(ValueError, match='Photon transport must be enabled'):
        settings.export_to_xml()

    settings.photon_transport = False
    with pytest.raises(ValueError, match='Photon transport must be enabled'):
        settings.export_to_xml()

    settings.photon_transport = True
    settings.export_to_xml()


def test_photoneutron_biasing_needs_photonuclear_physics(run_in_tmpdir):
    settings = openmc.Settings()
    settings.photon_transport = True
    settings.photoneutron_biasing = True
    with pytest.raises(ValueError, match='Photonuclear physics must be enabled'):
        settings.export_to_xml()

    settings.photonuclear_physics = True
    settings.export_to_xml()


def test_disabled_settings_export(run_in_tmpdir):
    settings = openmc.Settings()
    settings.photonuclear_physics = False
    settings.photoneutron_biasing = False
    settings.export_to_xml()


def test_statepoint_without_photonuclear_attributes(run_in_tmpdir):
    model = openmc.Model()
    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)
    model.materials = openmc.Materials([water])

    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([openmc.Cell(fill=water, region=-sph)])

    model.settings.run_mode = 'fixed source'
    model.settings.particles = 10
    model.settings.batches = 1
    model.settings.source = openmc.IndependentSource()
    sp_path = model.run()

    with openmc.StatePoint(sp_path) as sp:
        assert not sp.photonuclear_physics
        assert not sp.photoneutron_biasing

    # Remove the attributes, as in a statepoint written by an older version
    with h5py.File(sp_path, 'r+') as f:
        del f.attrs['photonuclear_physics']
        del f.attrs['photoneutron_biasing']

    with openmc.StatePoint(sp_path) as sp:
        assert not sp.photonuclear_physics
        assert not sp.photoneutron_biasing
