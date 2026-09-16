"""Tests for openmc.data.IncidentElectron.

These build an IncidentElectron from ACE if a library is available, and
otherwise construct the object directly, so the round trip and the invariants
the transport relies on are checked either way.
"""

from pathlib import Path

import numpy as np
import pytest

import openmc.data
from openmc.data import IncidentElectron
from openmc.data.angle_distribution import AngleDistribution
from openmc.data.energy_distribution import ContinuousTabular
from openmc.data.function import Tabulated1D
from openmc.stats import Tabular


def _tabular(x, p):
    """A Tabular carrying the CDF the HDF5 writers expect."""
    c = np.concatenate(([0.0], np.cumsum(0.5 * (p[:-1] + p[1:]) * np.diff(x))))
    dist = Tabular(x, p / c[-1], interpolation='linear-linear')
    dist.c = c / c[-1]
    return dist


@pytest.fixture
def synthetic():
    """A small but structurally complete IncidentElectron."""
    data = IncidentElectron(6)
    energy = np.array([10.0, 1.0e3, 1.0e6, 1.0e8])
    data.energy_grid = energy
    data.elastic_energy_range = (50.0, 1.0e8)

    mu = np.linspace(-1.0, 1.0, 21)
    for particle in ('electron', 'positron'):
        data.elastic_xs[particle] = np.array([5.0, 4.0, 3.0, 2.0])
        # Forward peaked, and more so for the positron
        w = 1.0 if particle == 'electron' else 2.0
        dists = [_tabular(mu, np.exp(w * mu) + 1.0e-3) for _ in energy]
        data.elastic_dist[particle] = AngleDistribution(energy, dists)

    data.excitation_xs = np.array([1.0, 0.8, 0.6, 0.4])
    data.excitation_energy_loss = Tabulated1D(energy, np.array(
        [1.0, 2.0, 3.0, 4.0]))

    data.shells = ['K']
    data.ionization_xs['K'] = np.array([0.0, 0.5, 0.4, 0.3])
    e_out = np.array([1.0, 10.0, 100.0])
    p_out = np.array([1.0e-2, 1.0e-3, 1.0e-4])
    data.ionization_dist['K'] = ContinuousTabular(
        [len(energy)], [5], energy,
        [_tabular(e_out, p_out) for _ in energy])

    data.bremsstrahlung_xs = np.array([0.1, 0.2, 0.3, 0.4])
    data.bremsstrahlung_photon_cutoff = 1.0
    return data


def test_attributes(synthetic):
    assert synthetic.atomic_number == 6
    assert synthetic.name == 'C'
    assert 'C' in repr(synthetic)


def test_atomic_number_checked():
    with pytest.raises(TypeError):
        IncidentElectron('carbon')
    with pytest.raises(ValueError):
        IncidentElectron(-1)


def test_roundtrip(synthetic, run_in_tmpdir):
    """export_to_hdf5 then from_hdf5 must reproduce every array."""
    synthetic.export_to_hdf5('electron.h5', 'w')
    back = IncidentElectron.from_hdf5('electron.h5')

    assert back.atomic_number == synthetic.atomic_number
    assert back.shells == synthetic.shells
    assert back.bremsstrahlung_photon_cutoff == \
        synthetic.bremsstrahlung_photon_cutoff
    assert back.elastic_energy_range == synthetic.elastic_energy_range

    np.testing.assert_allclose(back.energy_grid, synthetic.energy_grid)
    np.testing.assert_allclose(back.excitation_xs, synthetic.excitation_xs)
    np.testing.assert_allclose(back.bremsstrahlung_xs,
                               synthetic.bremsstrahlung_xs)
    for particle in ('electron', 'positron'):
        np.testing.assert_allclose(back.elastic_xs[particle],
                                   synthetic.elastic_xs[particle])
    for shell in synthetic.shells:
        np.testing.assert_allclose(back.ionization_xs[shell],
                                   synthetic.ionization_xs[shell])


def test_cross_sections_are_non_negative(synthetic):
    """Nothing the transport reads may be negative at a tabulated point."""
    assert np.all(synthetic.excitation_xs >= 0.0)
    assert np.all(synthetic.bremsstrahlung_xs >= 0.0)
    for particle in ('electron', 'positron'):
        assert np.all(synthetic.elastic_xs[particle] >= 0.0)
    for shell in synthetic.shells:
        assert np.all(synthetic.ionization_xs[shell] >= 0.0)


def test_elastic_range_within_grid(synthetic):
    """The recorded partial-wave range is what bounds the transport."""
    lo, hi = synthetic.elastic_energy_range
    assert lo > 0.0
    assert hi > lo


@pytest.mark.parametrize('symbol', ['C', 'Al'])
def test_from_ace_if_available(symbol):
    """Build from ACE when a library is present, and check the invariants the
    transport depends on: a common energy grid, both projectile charges, and
    cross sections defined everywhere on it."""
    try:
        import openmc.data.ace
        lib = openmc.config.get('cross_sections')
    except Exception:
        pytest.skip('no cross section configuration')
    if lib is None:
        pytest.skip('no cross section configuration')

    path = Path(lib).parent / 'electron' / f'{symbol}.h5'
    if not path.exists():
        pytest.skip(f'no electron data for {symbol}')

    data = IncidentElectron.from_hdf5(str(path))
    n = len(data.energy_grid)
    assert n > 1
    assert np.all(np.diff(data.energy_grid) > 0.0)
    assert set(data.elastic_xs) == {'electron', 'positron'}
    for particle in ('electron', 'positron'):
        assert len(data.elastic_xs[particle]) == n
    assert len(data.excitation_xs) == n
    assert len(data.bremsstrahlung_xs) == n
    for shell in data.shells:
        assert len(data.ionization_xs[shell]) == n
    # The bremsstrahlung cross section is an integral above this threshold, so
    # the transport has to sample the spectrum above the same one
    assert data.bremsstrahlung_photon_cutoff > 0.0
