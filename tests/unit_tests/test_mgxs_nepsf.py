import numpy as np
from numpy.polynomial import legendre
import pytest

import openmc
from openmc.mgxs.nepsf import discrete_angles, nepsf


def step_moments(lower, upper, weights, order):
    """Legendre moments of a set of uniform steps."""
    moments = np.zeros(order + 1)
    for a, b, w in zip(lower, upper, weights):
        for l in range(order + 1):
            P = legendre.Legendre.basis(l)
            if b > a:
                Q = P.integ()
                moments[l] += w * (Q(b) - Q(a)) / (b - a)
            else:
                moments[l] += w * P(a)
    return moments


def kernel_moments(f, order):
    """Legendre moments of a density on [-1, 1]."""
    mu = np.linspace(-1., 1., 200001)
    y = f(mu)
    y /= np.trapezoid(y, mu)
    return np.array([np.trapezoid(y * legendre.Legendre.basis(l)(mu), mu)
                     for l in range(order + 1)])


# Truncated expansions of a forward-peaked kernel go negative
PEAKED = kernel_moments(lambda mu: np.exp(8. * mu), 9)


def test_discrete_angles():
    angles, weights = discrete_angles(PEAKED[:6])
    assert angles.size == 3
    assert np.all(np.diff(angles) > 0.)
    assert weights.sum() == pytest.approx(1.)
    for l in range(6):
        assert weights @ legendre.Legendre.basis(l)(angles) == \
            pytest.approx(PEAKED[l], abs=1e-13)

    # E[mu^2] > 1 is not the moment of any distribution on [-1, 1]
    assert discrete_angles([1., 0., 1.5, 0.]) == (None, None)


@pytest.mark.parametrize('order', [1, 3, 5, 7, 9])
def test_nepsf_preserves_moments(order):
    lower, upper, weights, order_used = nepsf(PEAKED, order)
    assert order_used == order
    assert lower.size == (order + 1) // 2
    assert weights.sum() == pytest.approx(1.)
    assert np.all(weights > 0.)
    # Ordered, nonoverlapping steps of nonzero width within [-1, 1]
    bounds = np.ravel(np.column_stack([lower, upper]))
    assert bounds[0] >= -1. and bounds[-1] <= 1.
    assert np.all(np.diff(bounds) >= 0.)
    assert np.all(upper > lower)
    assert step_moments(lower, upper, weights, order) == \
        pytest.approx(PEAKED[:order + 1], abs=1e-9)


def test_nepsf_even_order():
    # The largest odd order not exceeding the requested one is preserved
    lower, upper, weights, order_used = nepsf(PEAKED, 4)
    assert order_used == 3
    assert lower.size == 2


def test_nepsf_strongly_peaked():
    # Klein-Nishina at 10 MeV restricted to outgoing energies above 9 MeV
    k = 10. / 0.51099895

    def compton(mu):
        ratio = 1. / (1. + k * (1. - mu))
        kn = ratio**2 * (ratio + 1. / ratio - (1. - mu**2))
        return np.where(ratio >= 0.9, kn, 0.)

    moments = kernel_moments(compton, 5)
    lower, upper, weights, order_used = nepsf(moments, 5)
    assert order_used == 5
    # The steps cover the kernel's support rather than collapsing to points
    mu_min = 1. - (1. / 0.9 - 1.) / k
    assert lower[0] >= mu_min - 1e-3
    assert np.sum(upper - lower) > 0.5 * (1. - mu_min)


def test_nepsf_point():
    # The only distribution with moments P_l(mu0) is a point at mu0
    moments = legendre.legvander(np.array([0.3]), 5)[0]
    lower, upper, weights, order_used = nepsf(moments, 5)
    used = weights > 0.
    assert np.all(lower[used] == upper[used])
    assert step_moments(lower, upper, weights, 5) == \
        pytest.approx(moments, abs=1e-12)


def test_nepsf_unrealizable():
    # Truncated moments that no distribution has fall back to a lower order
    moments = np.array([1., 0.2, 1.2, 0.1])
    lower, upper, weights, order_used = nepsf(moments, 3)
    assert order_used == 1
    assert weights[0] == 1.
    assert 0.5 * (lower[0] + upper[0]) == pytest.approx(0.2)


@pytest.fixture
def xsdata():
    groups = openmc.mgxs.EnergyGroups([0., 0.625, 20.e6])
    data = openmc.XSdata('test', groups)
    data.order = 5
    scatter = np.zeros((2, 2, 6))
    scatter[0, 0] = 2.0 * PEAKED[:6]
    scatter[0, 1] = 0.5 * kernel_moments(lambda mu: 1. + mu, 5)
    scatter[1, 1] = 3.0 * kernel_moments(lambda mu: np.exp(-2. * mu), 5)
    data.set_total([3., 4.])
    data.set_absorption([0.5, 1.])
    data.set_scatter_matrix(scatter)
    return data


def test_xsdata_nepsf(xsdata, tmp_path):
    steps = xsdata.convert_scatter_format('nepsf')
    assert steps.scatter_format == 'nepsf'
    assert steps.order == 5
    assert steps.scatter_matrix[0].shape == (2, 2, 9)
    weights = steps.scatter_matrix[0][..., 0::3]
    assert weights.sum(axis=-1) == \
        pytest.approx(xsdata.scatter_matrix[0][..., 0])
    assert np.all(steps.scatter_matrix[0][1, 0] == 0.)

    # The Legendre moments are recovered exactly
    back = steps.convert_scatter_format('legendre', 5)
    assert back.scatter_matrix[0] == \
        pytest.approx(xsdata.scatter_matrix[0], abs=1e-10)

    # Histograms and tables keep the group-to-group cross section
    hist = steps.convert_scatter_format('histogram', 16)
    assert hist.scatter_matrix[0].sum(axis=-1) == \
        pytest.approx(xsdata.scatter_matrix[0][..., 0])
    table = steps.convert_scatter_format('tabular', 33)
    mu = np.linspace(-1., 1., 33)
    assert np.trapezoid(table.scatter_matrix[0], mu, axis=-1) == \
        pytest.approx(xsdata.scatter_matrix[0][..., 0])

    # Lower orders and odd orders only
    assert xsdata.convert_scatter_format('nepsf', 3).order == 3
    with pytest.raises(ValueError):
        xsdata.convert_scatter_format('nepsf', 4)
    with pytest.raises(ValueError):
        xsdata.convert_scatter_format('nepsf', 7)

    # HDF5 round trip
    library = openmc.MGXSLibrary(steps.energy_groups)
    library.add_xsdata(steps)
    library.export_to_hdf5(tmp_path / 'mgxs.h5')
    read = openmc.MGXSLibrary.from_hdf5(tmp_path / 'mgxs.h5')
    data = read.get_by_name('test')
    assert data.scatter_format == 'nepsf'
    assert data.order == 5
    assert np.array_equal(data.scatter_matrix[0], steps.scatter_matrix[0])
