import numpy as np
import pytest

import openmc.data
import openmc.stats
from openmc.data import DiscreteCDF, TabularCDF, cdf_values
from openmc.stats import Discrete, Tabular, Uniform


def test_tabular_cdf_roundtrip():
    """Values tabulated in a file are returned unchanged."""
    # A cumulative distribution that disagrees with the one implied by the
    # probability density, as happens for unnormalized evaluations
    c = [0.0, 0.30, 0.55]
    d = TabularCDF([0.0, 1.0, 2.0], [0.1, 0.4, 0.1], 'linear-linear',
                   tabulated_cdf=c)
    assert np.array_equal(cdf_values(d), c)
    assert not np.array_equal(d.cdf(), c)


def test_cdf_values_fallback():
    """Distributions with no tabulated CDF get one computed from the PDF."""
    d = Tabular([0.0, 1.0, 2.0], [0.25, 0.5, 0.25])
    assert np.array_equal(cdf_values(d), d.cdf())


def test_cdf_values_discrete_convention():
    """The file convention has one cumulative value per point, no leading 0."""
    x, p = [1.0, 2.0, 3.0], [0.2, 0.3, 0.5]

    # Discrete.cdf() prepends a zero; the file convention does not
    assert len(Discrete(x, p).cdf()) == len(x) + 1
    assert len(cdf_values(Discrete(x, p))) == len(x)
    assert np.array_equal(cdf_values(Discrete(x, p)), np.cumsum(p))

    stored = DiscreteCDF(x, p, tabulated_cdf=np.cumsum(p))
    assert np.array_equal(cdf_values(stored), np.cumsum(p))


def test_cdf_values_unsupported():
    with pytest.raises(TypeError):
        cdf_values(Uniform(-1.0, 1.0))


def test_tabulated_cdf_wrong_length():
    with pytest.raises(ValueError):
        TabularCDF([0.0, 1.0, 2.0], [0.1, 0.4, 0.1], tabulated_cdf=[0.0, 1.0])


def test_tabulated_cdf_decreasing_warns():
    """Bad data is warned about but still passed through."""
    c = [0.0, 0.5, 0.2]
    with pytest.warns(UserWarning, match='decreases'):
        d = TabularCDF([0.0, 1.0, 2.0], [0.1, 0.4, 0.1], tabulated_cdf=c)
    assert np.array_equal(cdf_values(d), c)


def test_tabulated_cdf_optional():
    d = TabularCDF([0.0, 1.0], [0.5, 0.5])
    assert d.tabulated_cdf is None
    d.tabulated_cdf = [0.0, 1.0]
    assert np.array_equal(d.tabulated_cdf, [0.0, 1.0])
    d.tabulated_cdf = None
    assert d.tabulated_cdf is None


def test_normalize_keeps_cdf_consistent():
    d = TabularCDF([0.0, 1.0, 2.0], [0.2, 0.8, 0.2],
                   tabulated_cdf=[0.0, 1.0, 2.0])
    d.normalize()
    assert np.array_equal(d.tabulated_cdf, [0.0, 0.5, 1.0])


def test_prepend():
    d = TabularCDF([1.0, 2.0], [0.5, 0.5], tabulated_cdf=[0.3, 1.0])
    d.prepend(0.0)
    assert np.array_equal(d.x, [0.0, 1.0, 2.0])
    assert np.array_equal(d.p, [0.0, 0.5, 0.5])
    assert np.array_equal(d.tabulated_cdf, [0.0, 0.3, 1.0])

    # A distribution with no tabulated CDF is still consistent afterwards
    d = TabularCDF([1.0, 2.0], [0.5, 0.5])
    d.prepend(0.0)
    assert np.array_equal(d.x, [0.0, 1.0, 2.0])
    assert d.tabulated_cdf is None


def test_deprecated_c_attribute():
    d = TabularCDF([0.0, 1.0], [0.5, 0.5], tabulated_cdf=[0.0, 1.0])
    with pytest.warns(FutureWarning):
        assert np.array_equal(d.c, [0.0, 1.0])
    with pytest.warns(FutureWarning):
        d.c = [0.0, 2.0]
    assert np.array_equal(d.tabulated_cdf, [0.0, 2.0])


def test_legacy_c_attribute_honored():
    """Values attached the old way are still used, with a warning."""
    d = Tabular([0.0, 1.0, 2.0], [0.1, 0.4, 0.1])
    d.c = [0.0, 0.30, 0.55]
    with pytest.warns(FutureWarning):
        assert np.array_equal(cdf_values(d), [0.0, 0.30, 0.55])


def test_angle_distribution_without_tabulated_cdf(run_in_tmpdir):
    """An angular distribution not read from ACE/HDF5 can be exported.

    Distributions from AngleDistribution.from_endf carry no tabulated CDF,
    which used to raise AttributeError on export.
    """
    import h5py

    mu = [Tabular([-1.0, 0.0, 1.0], [0.25, 0.5, 0.25]) for _ in range(2)]
    original = openmc.data.AngleDistribution([1.0e3, 2.0e7], mu)

    with h5py.File('angle.h5', 'w') as f:
        original.to_hdf5(f.create_group('angle'))
    with h5py.File('angle.h5', 'r') as f:
        copy = openmc.data.AngleDistribution.from_hdf5(f['angle'])

    assert np.array_equal(copy.energy, original.energy)
    for mu_orig, mu_copy in zip(original.mu, copy.mu):
        assert np.array_equal(mu_copy.x, mu_orig.x)
        assert np.array_equal(mu_copy.p, mu_orig.p)
        assert np.array_equal(mu_copy.tabulated_cdf, mu_orig.cdf())


def test_isotropic_angle_distribution_export(run_in_tmpdir):
    """Uniform angular distributions no longer need a CDF attached."""
    import h5py

    original = openmc.data.AngleDistribution(
        [1.0e3, 2.0e7], [Uniform(-1.0, 1.0), Uniform(-1.0, 1.0)])
    with h5py.File('angle.h5', 'w') as f:
        original.to_hdf5(f.create_group('angle'))
    with h5py.File('angle.h5', 'r') as f:
        copy = openmc.data.AngleDistribution.from_hdf5(f['angle'])

    for mu_i in copy.mu:
        assert np.array_equal(mu_i.tabulated_cdf, [0.0, 1.0])


def _mixture_energy_out():
    """An outgoing energy distribution with discrete and continuous parts.

    The cumulative values are deliberately unnormalized and inconsistent with
    the probability density, as happens for data from some evaluations. Reading
    them back must reproduce them exactly, since the probability of the
    discrete component is recovered from the last discrete cumulative value.
    """
    discrete = DiscreteCDF([1.0e3, 2.0e3], [0.1, 0.15],
                           tabulated_cdf=[0.1, 0.25])
    continuous = TabularCDF([2.0e3, 5.0e3, 1.0e4], [1.0e-4, 2.0e-4, 1.0e-5],
                            'linear-linear',
                            tabulated_cdf=[0.25, 0.7, 0.95])
    return openmc.stats.Mixture([0.25, 0.75], [discrete, continuous])


@pytest.mark.parametrize('cls_name', ['ContinuousTabular',
                                      'CorrelatedAngleEnergy', 'KalbachMann'])
def test_hdf5_cdf_roundtrip(cls_name, run_in_tmpdir):
    """Tabulated cumulative values survive an HDF5 write/read/write cycle."""
    import h5py

    energy = [1.0e-5, 1.0e6]
    energy_out = [_mixture_energy_out() for _ in energy]

    if cls_name == 'ContinuousTabular':
        dist = openmc.data.ContinuousTabular([2], [2], energy, energy_out)
    elif cls_name == 'CorrelatedAngleEnergy':
        mu = [[TabularCDF([-1.0, 0.0, 1.0], [0.25, 0.5, 0.25],
                          tabulated_cdf=[0.0, 0.4, 1.0])
               for _ in range(len(d))] for d in energy_out]
        dist = openmc.data.CorrelatedAngleEnergy([2], [2], energy, energy_out,
                                                 mu)
    else:
        # Precompound factor and slope are tabulated at the outgoing energies
        n = len(energy_out[0])
        km = [openmc.data.Tabulated1D(np.linspace(1.0, float(n), n),
                                      np.full(n, 0.5)) for _ in energy]
        slope = [openmc.data.Tabulated1D(np.linspace(1.0, float(n), n),
                                         np.full(n, 0.1)) for _ in energy]
        dist = openmc.data.KalbachMann([2], [2], energy, energy_out, km, slope)

    def write(d, filename):
        with h5py.File(filename, 'w') as f:
            d.to_hdf5(f.create_group('dist'))

    write(dist, 'first.h5')
    with h5py.File('first.h5', 'r') as f:
        copy = type(dist).from_hdf5(f['dist'])
    write(copy, 'second.h5')

    # Every dataset must be bit-identical across the round trip
    with h5py.File('first.h5', 'r') as f1, h5py.File('second.h5', 'r') as f2:
        names = []
        f1['dist'].visit(names.append)
        assert names
        for name in names:
            if isinstance(f1['dist'][name], h5py.Dataset):
                assert np.array_equal(f1['dist'][name][()],
                                      f2['dist'][name][()]), name

    # The unnormalized cumulative values are preserved, not recomputed
    discrete, continuous = copy.energy_out[0].distribution
    assert np.array_equal(discrete.tabulated_cdf, [0.1, 0.25])
    assert np.array_equal(continuous.tabulated_cdf, [0.25, 0.7, 0.95])
    assert copy.energy_out[0].probability[0] == pytest.approx(0.25)
