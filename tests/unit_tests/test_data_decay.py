#!/usr/bin/env python

import os
from math import log
from pathlib import Path

import numpy as np
import pytest
from uncertainties import ufloat
import openmc.data
from openmc.exceptions import DataError


def ufloat_close(a, b):
    assert a.nominal_value == pytest.approx(b.nominal_value)
    assert a.std_dev == pytest.approx(b.std_dev)


@pytest.fixture(scope='module')
def nb90(endf_data):
    """Nb90 decay data."""
    filename = os.path.join(endf_data, 'decay', 'dec-041_Nb_090.endf')
    return openmc.data.Decay.from_endf(filename)


@pytest.fixture(scope='module')
def ba137m(endf_data):
    """Ba137_m1 decay data."""
    filename = os.path.join(endf_data, 'decay', 'dec-056_Ba_137m1.endf')
    return openmc.data.Decay.from_endf(filename)


@pytest.fixture(scope='module')
def u235_yields(endf_data):
    """U235 fission product yield data."""
    filename = os.path.join(endf_data, 'nfy', 'nfy-092_U_235.endf')
    return openmc.data.FissionProductYields.from_endf(filename)


def test_get_decay_modes():
    assert openmc.data.get_decay_modes(1.0) == ['beta-']
    assert openmc.data.get_decay_modes(6.0) == ['sf']
    assert openmc.data.get_decay_modes(10.0) == ['unknown']

    assert openmc.data.get_decay_modes(1.5) == ['beta-', 'n']
    assert openmc.data.get_decay_modes(1.4) == ['beta-', 'alpha']
    assert openmc.data.get_decay_modes(1.55) == ['beta-', 'n', 'n']
    assert openmc.data.get_decay_modes(1.555) == ['beta-', 'n', 'n', 'n']
    assert openmc.data.get_decay_modes(2.4) == ['ec/beta+', 'alpha']


def test_nb90_halflife(nb90):
    ufloat_close(nb90.half_life, ufloat(52560.0, 180.0))
    ufloat_close(nb90.decay_constant, log(2.)/nb90.half_life)
    ufloat_close(nb90.decay_energy, ufloat(2265527.5, 25159.400474401213))


def test_nb90_nuclide(nb90):
    assert nb90.nuclide['atomic_number'] == 41
    assert nb90.nuclide['mass_number'] == 90
    assert nb90.nuclide['isomeric_state'] == 0
    assert nb90.nuclide['parity'] == 1.0
    assert nb90.nuclide['spin'] == 8.0
    assert not nb90.nuclide['stable']
    assert nb90.nuclide['mass'] == pytest.approx(89.13888)


def test_nb90_modes(nb90):
    assert len(nb90.modes) == 2
    ec = nb90.modes[0]
    assert ec.modes == ['ec/beta+']
    assert ec.parent == 'Nb90'
    assert ec.daughter == 'Zr90'
    assert 'Nb90 -> Zr90' in str(ec)
    ufloat_close(ec.branching_ratio, ufloat(0.0147633, 0.0003386195))
    ufloat_close(ec.energy, ufloat(6111000., 4000.))

    # Make sure branching ratios sum to 1
    total = sum(m.branching_ratio for m in nb90.modes)
    assert total.nominal_value == pytest.approx(1.0)


def test_nb90_spectra(nb90):
    assert sorted(nb90.spectra.keys()) == ['e-', 'ec/beta+', 'gamma', 'xray']


def test_fpy(u235_yields):
    assert u235_yields.nuclide['atomic_number'] == 92
    assert u235_yields.nuclide['mass_number'] == 235
    assert u235_yields.nuclide['isomeric_state'] == 0
    assert u235_yields.nuclide['name'] == 'U235'
    assert u235_yields.energies == pytest.approx([0.0253, 500.e3, 1.4e7])

    assert len(u235_yields.cumulative) == 3
    thermal = u235_yields.cumulative[0]
    ufloat_close(thermal['I135'], ufloat(0.0628187, 0.000879461))

    assert len(u235_yields.independent) == 3
    thermal = u235_yields.independent[0]
    ufloat_close(thermal['I135'], ufloat(0.0292737, 0.000819663))


def test_decay_from_endf_material(endf_data):
    filename = os.path.join(endf_data, 'decay', 'dec-041_Nb_090.endf')
    material = openmc.data.endf.get_evaluations(filename)[0]

    data = openmc.data.Decay.from_endf(material)

    assert data.nuclide['name'] == 'Nb90'
    assert not data.nuclide['stable']
    assert len(data.modes) == 2


def test_fpy_from_endf_material(endf_data):
    filename = os.path.join(endf_data, 'nfy', 'nfy-092_U_235.endf')
    material = openmc.data.endf.get_evaluations(filename)[0]

    data = openmc.data.FissionProductYields.from_endf(material)

    assert data.nuclide['name'] == 'U235'
    assert data.energies == pytest.approx([0.0253, 500.e3, 1.4e7])
    assert 'I135' in data.cumulative[0]


def test_sources(ba137m, nb90):
    # Running .sources twice should give same objects
    sources = ba137m.sources
    sources2 = ba137m.sources
    for key in sources:
        assert sources[key] is sources2[key]

    # Each source should be a univariate distribution
    for dist in sources.values():
        assert isinstance(dist, openmc.stats.Univariate)

    # Check for presence of 662 keV gamma ray in decay of Ba137m
    gamma_source = ba137m.sources['photon']
    assert isinstance(gamma_source, openmc.stats.Discrete)
    b = np.isclose(gamma_source.x, 661657.)
    assert np.count_nonzero(b) == 1

    # Check value of decay/s/atom
    idx = np.flatnonzero(b)[0]
    assert gamma_source.p[idx] == pytest.approx(0.004069614)

    # Nb90 decays by β+ and should emit positrons, electrons, and photons
    sources = nb90.sources
    assert len(set(sources.keys()) ^ {'positron', 'electron', 'photon'}) == 0


def _beta_mean_energy(dist):
    """Energy emission rate of the beta spectra in a decay source"""
    return sum(p * d.mean() for p, d in zip(dist.probability, dist.distribution)
               if isinstance(d, openmc.stats.BetaSpectrum))


def test_beta_sources(endf_data):
    # Na22 decays by beta+ (90%) and electron capture. Positrons are emitted
    # with the positron intensity and a spectrum that ends below the electron
    # capture decay energy by twice the electron mass.
    na22 = openmc.data.Decay.from_endf(
        os.path.join(endf_data, 'decay', 'dec-011_Na_022.endf'))
    lam = na22.decay_constant.n
    spectrum = na22.spectra['ec/beta+']
    n_positron = sum(d['positron_intensity'].n for d in spectrum['discrete'])
    positron = na22.sources['positron']
    assert isinstance(positron, openmc.stats.Mixture)
    assert positron.integral() == pytest.approx(lam * n_positron)
    for d in positron.distribution:
        assert isinstance(d, openmc.stats.BetaSpectrum)
        assert d.positron
    assert max(d.endpoint for d in positron.distribution) == pytest.approx(
        2843020.0 - 1021997.9)

    # The evaluated average energy is preserved
    assert _beta_mean_energy(positron) / lam == pytest.approx(
        spectrum['energy_average'].n, rel=1e-6)

    # Fe55 decays only by electron capture and emits no positrons
    fe55 = openmc.data.Decay.from_endf(
        os.path.join(endf_data, 'decay', 'dec-026_Fe_055.endf'))
    assert 'positron' not in fe55.sources

    # Sr90 and Y90 undergo unique first-forbidden decays whose average energy
    # differs from that of the allowed shape, but the average energy of the
    # evaluation is preserved
    for name in ('dec-038_Sr_090.endf', 'dec-039_Y_090.endf'):
        decay = openmc.data.Decay.from_endf(
            os.path.join(endf_data, 'decay', name))
        electron = decay.sources['electron']
        mean = _beta_mean_energy(electron) / decay.decay_constant.n
        assert mean == pytest.approx(
            decay.spectra['beta-']['energy_average'].n, rel=1e-6)


def test_beta_spectrum_forbidden(endf_data):
    # With the shape factor of unique first-forbidden transitions, the average
    # energy of Sr90 and Y90 beta particles agrees with the evaluation, which
    # the allowed shape underestimates by 11% for Sr90
    for name, Z, A in (('dec-038_Sr_090.endf', 39, 90),
                       ('dec-039_Y_090.endf', 40, 90)):
        decay = openmc.data.Decay.from_endf(
            os.path.join(endf_data, 'decay', name))
        spectrum = decay.spectra['beta-']
        (transition,) = [d for d in spectrum['discrete']
                         if d['intensity'].n > 0.99]
        endpoint = transition['energy'].n
        expected = spectrum['energy_average'].n / transition['intensity'].n
        unique = openmc.stats.BetaSpectrum(
            endpoint, Z, A, forbiddenness=1, unique=True)
        assert unique.mean() == pytest.approx(expected, rel=5e-3)
    allowed = openmc.stats.BetaSpectrum(endpoint, Z, A)
    assert unique.mean() > allowed.mean()



@pytest.mark.parametrize('initial, final, expected', [
    ((0.5, 1.0), (0.5, 1.0), (0, False)),   # H3: allowed
    ((7.0, 1.0), (6.0, 1.0), (0, False)),
    ((1.0, -1.0), (0.0, 1.0), (1, False)),  # Bi210: non-unique first
    ((0.0, 1.0), (2.0, -1.0), (1, True)),   # Sr90: unique first
    ((2.0, 1.0), (0.0, 1.0), (2, False)),   # Cl36: non-unique second
    ((3.0, 1.0), (0.0, 1.0), (2, True)),    # Na22: unique second
    ((4.0, -1.0), (0.0, 1.0), (3, True)),   # K40: unique third
    ((4.5, 1.0), (0.5, 1.0), (4, False)),   # In115: non-unique fourth
])
def test_transition_shape(initial, final, expected):
    assert openmc.data.decay._transition_shape(initial, final) == expected


def test_beta_spectrum_unique_inferred(endf_data):
    # Unique forbidden transitions are identified from the spins and parities
    # of the parent and of the daughter state fed by the transition, which are
    # known from the decay data of the daughter
    names = ['Sr_090', 'Y_090', 'Zr_090', 'K_040', 'Ca_040', 'Ar_040',
             'Cs_137', 'Ba_137', 'Ba_137m1']
    decay_data = {}
    for name in names:
        Z = openmc.data.ATOMIC_NUMBER[name.split('_')[0]]
        filename = os.path.join(endf_data, 'decay', f'dec-{Z:03}_{name}.endf')
        decay = openmc.data.Decay.from_endf(filename)
        decay_data[decay.nuclide['name']] = decay

    def beta_spectra(name, particle='electron'):
        sources = decay_data[name].get_sources(decay_data)
        return {round(d.endpoint): d for d in sources[particle].distribution
                if isinstance(d, openmc.stats.BetaSpectrum)}

    # Sr90 (0+ -> 2-) and Y90 (2- -> 0+) undergo unique first-forbidden
    # transitions, whose shape reproduces the evaluated average energy with
    # only a small correction
    for name, endpoint in (('Sr90', 546000), ('Y90', 2279800)):
        spectrum = beta_spectra(name)[endpoint]
        assert spectrum.forbiddenness == 1
        assert spectrum.unique
        assert abs(spectrum.shape_factor * endpoint) < 0.01

    # The K40 transition to Ca40 (4- -> 0+) is unique third forbidden, although
    # the evaluation lists it as allowed
    k40 = decay_data['K40']
    (transition,) = k40.spectra['beta-']['discrete']
    assert transition['type'] == 'allowed'
    spectrum = beta_spectra('K40')[1311070]
    assert (spectrum.forbiddenness, spectrum.unique) == (3, True)

    # The main Cs137 transition feeds the Ba137 isomer (7/2+ -> 11/2-) and is
    # unique first forbidden, while the transition to the ground state
    # (7/2+ -> 3/2+) is non-unique second forbidden and has the allowed shape
    spectra = beta_spectra('Cs137')
    main, ground = spectra[513971], spectra[1175630]
    assert (main.forbiddenness, main.unique) == (1, True)
    assert (ground.forbiddenness, ground.unique) == (0, False)

    # The evaluated average energy is preserved
    cs137 = decay_data['Cs137']
    electron = cs137.get_sources(decay_data)['electron']
    assert _beta_mean_energy(electron) / cs137.decay_constant.n == \
        pytest.approx(cs137.spectra['beta-']['energy_average'].n, rel=1e-6)

    # Without decay data for the daughters, the allowed shape is used
    for d in cs137.sources['electron'].distribution:
        if isinstance(d, openmc.stats.BetaSpectrum):
            assert not d.unique


def test_decay_photon_energy():
    # If chain file is not set, we should get a data error
    if 'chain_file' in openmc.config:
        del openmc.config['chain_file']
    with pytest.raises(DataError):
        openmc.data.decay_photon_energy('I135')

    # Temporarily Set chain file to simple chain
    with openmc.config.patch('chain_file', Path(__file__).parents[1] / 'chain_simple.xml'):

        # Check strength of I135 source and presence of specific spectral line
        src = openmc.data.decay_photon_energy('I135')
        assert isinstance(src, openmc.stats.Discrete)
        assert src.integral() == pytest.approx(3.920996223799345e-05)
        assert 1260409. in src.x

        # Check Xe135 source, which should be tabular
        src = openmc.data.decay_photon_energy('Xe135')
        assert isinstance(src, openmc.stats.Tabular)
        assert src.integral() == pytest.approx(2.076506258964966e-05)
