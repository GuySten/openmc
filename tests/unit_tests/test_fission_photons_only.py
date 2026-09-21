"""Tests for Settings.fission_photons_only.

The setting discards every secondary photon whose parent reaction is not
fission, as it is born. Fission photons carry most of the yield above the
photonuclear thresholds, so a calculation whose answer they dominate can drop
the rest and stop paying to transport them.
"""

import lxml.etree as ET
import numpy as np
import pytest

import openmc


@pytest.fixture
def model():
    """Water-moderated fuel sphere, tallying the photon flux it produces.

    Water is deliberate: capture on hydrogen emits a 2.22 MeV photon, so the
    non-fission photon source here is large enough that removing it is
    unmistakable.
    """
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.0)

    water = openmc.Material()
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    inner = openmc.Sphere(r=6.0)
    outer = openmc.Sphere(r=12.0, boundary_type='vacuum')
    core = openmc.Cell(fill=fuel, region=-inner)
    reflector = openmc.Cell(fill=water, region=+inner & -outer)

    model = openmc.Model()
    model.geometry = openmc.Geometry([core, reflector])
    model.settings.run_mode = 'eigenvalue'
    model.settings.particles = 2000
    model.settings.batches = 15
    model.settings.inactive = 5
    model.settings.photon_transport = True
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())
    model.settings.seed = 1

    tally = openmc.Tally(name='photon flux')
    tally.filters = [openmc.ParticleFilter(['photon'])]
    tally.scores = ['flux']
    model.tallies = openmc.Tallies([tally])
    return model


def _photon_flux(model):
    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='photon flux')
        return tally.mean.ravel()[0], tally.std_dev.ravel()[0]


def test_fission_photons_only_removes_the_rest(run_in_tmpdir, model):
    """Fewer photons, but not none.

    Capture on hydrogen is a large photon source here, so dropping the
    non-fission reactions has to reduce the photon flux far outside its
    error bar. It must not reduce it to zero: the fission photons, which are
    the whole point of the setting, are still there.
    """
    model.settings.fission_photons_only = False
    everything, sigma_all = _photon_flux(model)

    model.settings.fission_photons_only = True
    fission_only, sigma_fis = _photon_flux(model)

    assert fission_only > 0.0
    assert fission_only < everything - 3.0 * np.hypot(sigma_all, sigma_fis)


def test_fission_photons_only_requires_photon_transport(run_in_tmpdir, model):
    """Without photon transport there are no secondary photons to restrict."""
    model.settings.photon_transport = False
    model.settings.fission_photons_only = True

    with pytest.raises(RuntimeError, match='fission_photons_only'):
        model.run()


def test_fission_photons_only_xml_roundtrip():
    s = openmc.Settings()
    assert s.fission_photons_only is None

    s.photon_transport = True
    s.fission_photons_only = True
    elem = s.to_xml_element()
    assert elem.find('fission_photons_only').text == 'true'
    assert openmc.Settings.from_xml_element(elem).fission_photons_only

    with pytest.raises(TypeError):
        s.fission_photons_only = 'yes'
