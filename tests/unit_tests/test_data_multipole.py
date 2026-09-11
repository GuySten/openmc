import os
import pathlib

import numpy as np
import pytest
import openmc.data


@pytest.fixture(scope='module')
def u235():
    directory = pathlib.Path(openmc.config.get('cross_sections')).parent
    u235 = directory / 'wmp' / '092235.h5'
    return openmc.data.WindowedMultipole.from_hdf5(u235)


@pytest.fixture(scope='module')
def b10():
    directory = pathlib.Path(openmc.config.get('cross_sections')).parent
    b10 = directory / 'wmp' / '005010.h5'
    return openmc.data.WindowedMultipole.from_hdf5(b10)


def test_evaluate(u235):
    """Test the cross section evaluation of a library."""
    energies = [1e-3, 1.0, 10.0, 50.]
    scattering, absorption, fission = u235(energies, 0.0)
    assert (scattering[1], absorption[1], fission[1]) == \
           pytest.approx((13.09, 77.56, 67.36), rel=1e-3)
    scattering, absorption, fission = u235(energies, 300.0)
    assert (scattering[2], absorption[2], fission[2]) == \
           pytest.approx((11.24, 21.26, 15.50), rel=1e-3)


def test_evaluate_none_poles(b10):
    """Test a library with no poles, i.e., purely polynomials."""
    energies = [1e-3, 1.0, 10.0, 1e3, 1e5]
    scattering, absorption, fission = b10(energies, 0.0)
    assert (scattering[0], absorption[0], fission[0]) == \
           pytest.approx((2.201, 19330., 0.), rel=1e-3)
    scattering, absorption, fission = b10(energies, 300.0)
    assert (scattering[-1], absorption[-1], fission[-1]) == \
           pytest.approx((2.878, 1.982, 0.), rel=1e-3)


def test_export_to_hdf5(tmpdir, u235):
    filename = str(tmpdir.join('092235.h5'))
    u235.export_to_hdf5(filename)
    assert os.path.exists(filename)


def test_from_endf(endf_data):
    endf_file = os.path.join(endf_data, 'neutrons', 'n-001_H_001.endf')
    assert openmc.data.WindowedMultipole.from_endf(
            endf_file,
            log=True,
            # Keep the test lightweight
            vf_options={
                "njoy_error": 5e-3,
                "vf_pieces": 1,
                "rtol": 5e-2,
                "atol": 1e-3,
                "orders": [8, 12],
                "n_vf_iter": 6,
            },
            wmp_options={"n_win": 50, "n_cf": 3, "rtol": 5e-2, "atol": 1e-3},
        )


def test_from_endf_search(endf_data):
    endf_file = os.path.join(endf_data, 'neutrons', 'n-001_H_001.endf')
    assert openmc.data.WindowedMultipole.from_endf(
            endf_file,
            log=True,
            vf_options={
                "njoy_error": 5e-3,
                "vf_pieces": 1,
                "rtol": 5e-2,
                "atol": 1e-3,
                "orders": [8, 12],
                "n_vf_iter": 6,
            },
            wmp_options={
                "search": True,
                "rtol": 5e-2,
                "search_n_win": 3,
                "search_cf_orders": [5, 3],
            },
        )


def _rational_xs(n_res=3, seed=11, tol=3e-5):
    """Cross sections that are exactly representable in multipole form.

    Because the target is an exact sum of pole terms, vector fitting can
    reproduce it to arbitrary accuracy, which makes it a clean check that
    ``rtol`` is really enforced.

    The energy grid is refined adaptively, the way NJOY reconstructs
    point-wise data, so that linear interpolation between grid points stays
    within ``tol``. Without a separate checking grid :func:`_vectfit_xs` judges
    the fit at these points, so they have to resolve the resonances for a tight
    ``rtol`` to mean anything.
    """
    rng = np.random.default_rng(seed)
    E_r = np.sort(rng.uniform(50.0, 9e4, n_res))
    sqrt_E_r = np.sqrt(E_r)
    gamma = sqrt_E_r * rng.uniform(3e-3, 1e-2, n_res)

    # purely imaginary residues give strictly positive resonance peaks, and
    # the two extra poles supply a smooth positive background
    poles = np.concatenate([sqrt_E_r + 1j*gamma, [120 + 260j, 20 + 60j]])
    residues = np.hstack([
        np.vstack([-1j*rng.uniform(5, 60, n_res)*sqrt_E_r*gamma,
                   -1j*rng.uniform(2, 30, n_res)*sqrt_E_r*gamma]),
        np.array([[-6.0e5j, -3.0e4j], [-2.0e3j, -8.0e2j]])
    ])
    # both members of each conjugate pair, stored adjacently
    all_poles = np.empty(2*poles.size, dtype=complex)
    all_poles[0::2], all_poles[1::2] = poles, np.conj(poles)
    all_residues = np.empty((2, 2*poles.size), dtype=complex)
    all_residues[:, 0::2], all_residues[:, 1::2] = residues, np.conj(residues)

    def xs_at(energy):
        return openmc.data.multipole.evaluate(
            np.sqrt(energy), all_poles, all_residues) / energy

    energy = np.unique(np.concatenate(
        [np.geomspace(1e-3, 1e5, 400), E_r, E_r*(1 + 1e-6), E_r*(1 - 1e-6)]))
    for _ in range(40):
        midpoint = 0.5*(energy[:-1] + energy[1:])
        xs = xs_at(energy)
        interpolated = 0.5*(xs[:, :-1] + xs[:, 1:])
        exact = xs_at(midpoint)
        coarse = np.any(np.abs(interpolated - exact) > tol*np.abs(exact), axis=0)
        if not coarse.any():
            break
        energy = np.unique(np.concatenate([energy, midpoint[coarse]]))

    return energy, xs_at(energy)


def test_vectfit_rtol_is_enforced():
    """The relative error tolerance must be honored, not merely penalized."""
    energy, xs = _rational_xs()

    for rtol in (1e-3, 1e-4):
        poles, residues = openmc.data.multipole._vectfit_xs(
            energy, xs, [2, 27], rtol=rtol, orders=range(6, 18, 2),
            n_vf_iter=10)
        fit = openmc.data.multipole.evaluate(
            np.sqrt(energy), poles, residues*1j) / energy
        assert np.max(np.abs(fit - xs)/np.abs(xs)) <= rtol


def test_vectfit_rtol_unreachable():
    """An unreachable tolerance must fail loudly rather than silently.

    ``atol`` is lowered along with ``rtol``: a point whose absolute error is
    already within ``atol`` counts as converged regardless of ``rtol``, so a
    tight relative tolerance is only meaningful with a tight absolute one.
    Which of the two failures is reported depends on whether any candidate was
    usable at all; either way it must raise rather than return a poor fit, and
    name the energy range so the failure can be located.
    """
    energy, xs = _rational_xs()
    with pytest.raises(RuntimeError, match='Vector fitting'):
        openmc.data.multipole._vectfit_xs(
            energy, xs, [2, 27], rtol=1e-12, atol=1e-14, orders=[6, 8],
            n_vf_iter=4)


def test_vectfit_orders_are_searched_in_order():
    """A user-supplied list of orders must not be scrambled by set().

    ``list(set([2, 4, 6, 8]))`` gives ``[8, 2, 4, 6]``, so an unsorted
    normalization shows up as a reversed range in the reported error.
    """
    energy, xs = _rational_xs()
    with pytest.raises(RuntimeError, match='orders 2 to 8'):
        openmc.data.multipole._vectfit_xs(
            energy, xs, [2, 27], rtol=1e-12, atol=1e-14,
            orders=range(2, 10, 2), n_vf_iter=4)


def test_vectfit_gives_up_when_error_stops_improving(capsys):
    """A hopeless tolerance must abandon the search, not exhaust it.

    Adding poles beyond a certain point cannot reduce the error further, and
    the search must notice rather than grinding through every order in the
    range, which for a real nuclide can be hundreds of them.
    """
    energy, xs = _rational_xs(n_res=2)
    with pytest.raises(RuntimeError, match='rtol'):
        openmc.data.multipole._vectfit_xs(
            energy, xs, [2, 27], rtol=1e-14, atol=1e-16,
            orders=range(2, 100, 2), n_vf_iter=2, log=True)
    assert 'stopped improving' in capsys.readouterr().out


def test_vectfit_check_grid():
    """A finer checking grid must be used as the reference for accuracy."""
    energy, xs = _rational_xs()
    # a grid twice as dense, sampled from the same exact target
    fine = np.unique(np.concatenate([energy, 0.5*(energy[:-1] + energy[1:])]))
    coarse = energy[::2]

    poles, residues = openmc.data.multipole._vectfit_xs(
        coarse, xs[:, ::2], [2, 27], rtol=1e-3,
        check_energy=fine, check_xs=np.vstack(
            [np.interp(fine, energy, xs[i]) for i in range(2)]),
        orders=range(6, 20, 2), n_vf_iter=10)

    # accuracy must hold on the finer grid, not merely at the fitted points
    fit = openmc.data.multipole.evaluate(
        np.sqrt(fine), poles, residues*1j) / fine
    ref = np.vstack([np.interp(fine, energy, xs[i]) for i in range(2)])
    assert np.max(np.abs(fit - ref)/np.abs(ref)) <= 1e-3


def test_background_kinks(endf_data):
    """Only pronounced corners in the MF3 background become piece boundaries.

    Backgrounds are interpolated linearly between tabulated energies, so the
    cross section has a corner at each one. A sum of poles cannot reproduce a
    corner, but splitting at every one of them would fragment the fit, so only
    those whose step exceeds the threshold are returned, and both ends of the
    stepping interval are needed to keep the corner off the interior.
    """
    endf_file = os.path.join(endf_data, 'neutrons', 'n-026_Fe_056.endf')
    if not os.path.exists(endf_file):
        pytest.skip('Fe-56 evaluation not available')

    kinks = openmc.data.multipole._background_kinks(
        endf_file, [2, 27], 0.25, 1e-5, 8.5e5)

    # the capture background steps by roughly a factor of two at 650 keV over
    # a 1 keV interval, and both ends must be returned
    assert 650000.0 in kinks
    assert 651000.0 in kinks

    # a tighter threshold can only add boundaries, a looser one only remove
    loose = openmc.data.multipole._background_kinks(
        endf_file, [2, 27], 0.5, 1e-5, 8.5e5)
    tight = openmc.data.multipole._background_kinks(
        endf_file, [2, 27], 0.1, 1e-5, 8.5e5)
    assert set(loose) <= set(kinks) <= set(tight)

    # boundaries are sorted and strictly inside the requested range
    assert np.all(np.diff(kinks) > 0)
    assert kinks.min() > 1e-5 and kinks.max() < 8.5e5
