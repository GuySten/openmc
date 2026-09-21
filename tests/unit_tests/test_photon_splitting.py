"""Tests for Settings.photon_splits and Settings.photoneutron_splits.

Both emit several particles of a fraction of the weight in place of one of
the whole, each sampled independently. The expected weight emitted is
unchanged, so every score linear in it must come back with the same mean and
a smaller variance -- which is what makes them variance reduction rather than
a different calculation.
"""

import numpy as np
import pytest

import openmc


@pytest.fixture
def model():
    """Fuel sphere in a water reflector, tallying the photon flux."""
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
    # Split by energy. Splitting is for the rare, hard photons -- the ones
    # that can reach a photonuclear threshold -- and the bulk of the spectrum
    # is already sampled so well that it has nothing to gain.
    tally.filters = [
        openmc.ParticleFilter(['photon']),
        openmc.EnergyFilter([0.0, 6.0e6, 2.0e7]),
    ]
    tally.scores = ['flux']
    model.tallies = openmc.Tallies([tally])
    return model


def _photon_flux(model):
    """Flux in the soft and hard bins, each as (mean, std_dev)."""
    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='photon flux')
        mean = tally.mean.ravel()
        std_dev = tally.std_dev.ravel()
        return list(zip(mean, std_dev))


def test_photon_splitting_is_unbiased_and_reduces_hard_photon_variance(
        run_in_tmpdir, model):
    """Same flux, far better determined above 6 MeV.

    Both halves matter. Agreement alone would also hold if the setting did
    nothing; a smaller error bar alone would also hold if it quietly changed
    the physics.

    The assertion is on the hard bin because that is where splitting is
    supposed to pay and where the effect is big enough to resolve in seconds:
    measured 4x on sigma there, against about 1.15x on the total flux. A
    15% change needs some 200 active batches before sigma itself is pinned
    down well enough to test, which is no unit test; a 4x change is decided
    in ten.
    """
    (soft, soft_sd), (hard, hard_sd) = _photon_flux(model)

    model.settings.photon_splits = 8
    (soft8, soft8_sd), (hard8, hard8_sd) = _photon_flux(model)

    # Unbiased, in both bins
    assert soft8 == pytest.approx(soft, abs=4.0 * np.hypot(soft_sd, soft8_sd))
    assert hard8 == pytest.approx(hard, abs=4.0 * np.hypot(hard_sd, hard8_sd))

    # And the hard photons, which are what splitting is for, are much better
    # determined. Threshold well inside the measured 4x.
    assert hard8_sd < 0.5 * hard_sd, (
        f'hard-photon sigma went from {hard_sd:.3g} to {hard8_sd:.3g}; '
        'splitting is not taking effect')


def test_sampling_above_cutoff_is_unbiased_and_much_cheaper(run_in_tmpdir,
                                                            model):
    """Drawing only above the cutoff gives the same answer, far better.

    With the cutoff at 6 MeV, almost every photon drawn from the full
    production spectrum is below it and discarded at birth. Restricting the
    draw to what survives and carrying forward the probability mass is
    unbiased -- those photons were going to be discarded anyway -- and turns
    roughly one useful draw in a hundred into every draw.

    Measured: sigma on the flux above 6 MeV falls from 5.8% to 1.7% for a
    10% increase in runtime, a factor of 12 in figure of merit. The
    threshold below is well inside that.
    """
    model.settings.cutoff = {'energy_photon': 6.0e6}
    (_, _), (hard, hard_sd) = _photon_flux(model)

    model.settings.sample_photons_above_cutoff = True
    (_, _), (hard_t, hard_t_sd) = _photon_flux(model)

    assert hard_t == pytest.approx(
        hard, abs=4.0 * np.hypot(hard_sd, hard_t_sd)), (
        'restricting the draw changed the answer, so the probability mass '
        'being carried forward does not match what was sampled from')
    assert hard_t_sd < 0.5 * hard_sd, (
        f'sigma above the cutoff went from {hard_sd:.3g} to {hard_t_sd:.3g}; '
        'the restriction is not taking effect')


def test_sampling_above_cutoff_requires_photon_transport(run_in_tmpdir,
                                                         model):
    model.settings.photon_transport = False
    model.settings.sample_photons_above_cutoff = True

    with pytest.raises(RuntimeError, match='sample_photons_above_cutoff'):
        model.run()


def test_sample_photons_above_cutoff_xml_roundtrip():
    s = openmc.Settings()
    assert s.sample_photons_above_cutoff is None

    s.photon_transport = True
    s.sample_photons_above_cutoff = True
    elem = s.to_xml_element()
    assert elem.find('sample_photons_above_cutoff').text == 'true'
    assert openmc.Settings.from_xml_element(elem).sample_photons_above_cutoff

    with pytest.raises(TypeError):
        s.sample_photons_above_cutoff = 1


def test_splitting_rejects_pulse_height_tallies(run_in_tmpdir, model):
    """A pulse height is not linear in the weight, so splitting corrupts it."""
    cell = next(iter(model.geometry.get_all_cells().values()))
    tally = openmc.Tally(name='pulse height')
    tally.scores = ['pulse-height']
    tally.filters = [
        openmc.CellFilter([cell]),
        openmc.EnergyFilter(np.linspace(0.0, 2.0e6, 5)),
    ]
    model.tallies.append(tally)
    model.settings.photon_splits = 4

    with pytest.raises(RuntimeError, match='pulse-height'):
        model.run()


def test_splitting_requires_photon_transport(run_in_tmpdir, model):
    model.settings.photon_transport = False
    model.settings.photon_splits = 4

    with pytest.raises(RuntimeError, match='photon_splits'):
        model.run()


def test_photoneutron_splitting_requires_photon_transport(run_in_tmpdir,
                                                          model):
    model.settings.photon_transport = False
    model.settings.photoneutron_splits = 4

    with pytest.raises(RuntimeError, match='photoneutron_splits'):
        model.run()


@pytest.mark.parametrize('attr', ['photon_splits', 'photoneutron_splits'])
def test_splits_xml_roundtrip(attr):
    s = openmc.Settings()
    assert getattr(s, attr) is None

    s.photon_transport = True
    setattr(s, attr, 8)
    elem = s.to_xml_element()
    assert elem.find(attr).text == '8'
    assert getattr(openmc.Settings.from_xml_element(elem), attr) == 8

    with pytest.raises(ValueError):
        setattr(s, attr, 0)
    with pytest.raises(TypeError):
        setattr(s, attr, 2.5)
