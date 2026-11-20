#!/usr/bin/env python

import os
from math import log
from pathlib import Path

import h5py
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


def _same_values(a, b):
    """Equal nested data: values with uncertainties compared by value and
    standard deviation"""
    if hasattr(a, 'nominal_value'):
        assert (a.nominal_value, a.std_dev) == (b.nominal_value, b.std_dev) \
            or (np.isinf(a.nominal_value) and np.isinf(b.nominal_value))
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _same_values(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same_values(x, y)
    elif isinstance(a, openmc.data.Tabulated1D):
        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
        assert list(a.interpolation) == list(b.interpolation)
    else:
        assert a == b


@pytest.mark.parametrize('name', ['dec-041_Nb_090.endf', 'dec-056_Ba_137m1.endf',
                                  'dec-095_Am_242m1.endf', 'dec-026_Fe_056.endf'])
def test_decay_hdf5_round_trip(endf_data, tmp_path, name):
    d = openmc.data.Decay.from_endf(os.path.join(endf_data, 'decay', name))
    d.export_to_hdf5(tmp_path / 'decay.h5', 'w')
    r = openmc.data.Decay.from_hdf5(tmp_path / 'decay.h5')
    _same_values(d.nuclide, r.nuclide)
    _same_values(d.half_life, r.half_life)
    _same_values(d.average_energies, r.average_energies)
    assert [(m.modes, m.daughter) for m in d.modes] == \
        [(m.modes, m.daughter) for m in r.modes]
    _same_values([(m.energy, m.branching_ratio) for m in d.modes],
                 [(m.energy, m.branching_ratio) for m in r.modes])
    _same_values(d.spectra, r.spectra)
    with h5py.File(tmp_path / 'decay.h5') as f:
        assert f.attrs['filetype'].decode() == 'data_decay'


def test_decay_excitation_energy(endf_data):
    am242m = openmc.data.Decay.from_endf(
        os.path.join(endf_data, 'decay', 'dec-095_Am_242m1.endf'))
    assert am242m.nuclide['excitation_energy'] == pytest.approx(48.6e3, rel=1e-2)


@pytest.mark.parametrize('sub, name', [('nfy', 'nfy-092_U_235.endf'),
                                       ('sfy', 'sfy-098_Cf_252.endf')])
def test_fpy_hdf5_round_trip(endf_data, tmp_path, sub, name):
    y = openmc.data.FissionProductYields.from_endf(os.path.join(endf_data, sub, name))
    assert y.sublibrary == sub
    y.export_to_hdf5(tmp_path / 'fpy.h5', 'w')
    r = openmc.data.FissionProductYields.from_hdf5(tmp_path / 'fpy.h5')
    assert r.sublibrary == sub
    assert r.nuclide == y.nuclide
    assert np.array_equal(r.energies, y.energies)
    assert r.interpolation == y.interpolation
    _same_values(y.independent, r.independent)
    _same_values(y.cumulative, r.cumulative)
    with pytest.raises(DataError):
        openmc.data.Decay.from_hdf5(tmp_path / 'fpy.h5')


def test_sublibraries_in_cross_sections_xml(endf_data, tmp_path):
    d = openmc.data.Decay.from_endf(os.path.join(endf_data, 'decay', 'dec-041_Nb_090.endf'))
    y = openmc.data.FissionProductYields.from_endf(
        os.path.join(endf_data, 'nfy', 'nfy-092_U_235.endf'))
    d.export_to_hdf5(tmp_path / 'Nb90.h5', 'w')
    y.export_to_hdf5(tmp_path / 'U235.h5', 'w')
    lib = openmc.data.DataLibrary()
    lib.register_file(tmp_path / 'Nb90.h5')
    lib.register_file(tmp_path / 'U235.h5')
    assert [(x['type'], x['materials']) for x in lib] == [('decay', ['Nb90']), ('nfy', ['U235'])]
    lib.export_to_xml(tmp_path / 'cross_sections.xml')
    # not <library> elements: the transport solver does not read them
    text = (tmp_path / 'cross_sections.xml').read_text()
    assert '<library' not in text and '<decay' in text and '<nfy' in text
    lib2 = openmc.data.DataLibrary.from_xml(tmp_path / 'cross_sections.xml')
    assert lib2.get_by_material('Nb90', 'decay')['path'] == str(tmp_path / 'Nb90.h5')
    assert lib2.get_by_material('U235', 'nfy')['path'] == str(tmp_path / 'U235.h5')
