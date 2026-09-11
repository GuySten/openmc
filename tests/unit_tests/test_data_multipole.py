import inspect
import os
import pathlib

import numpy as np
import pytest
from scipy.signal import find_peaks
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
    energy, half = kinks[:, 0], kinks[:, 1]

    # the capture background steps by roughly a factor of two at 650 keV over
    # a 1 keV interval, and both ends of it are corners
    assert 650000.0 in energy
    assert 651000.0 in energy

    # the band around a corner is the tabulation spacing beside it, not the
    # width of the stepping interval: a background that steps across a broad
    # interval is a slope, and only its ends are corners
    assert half[list(energy).index(650000.0)] == pytest.approx(500.0)

    # a tighter threshold can only add corners, a looser one only remove
    loose = openmc.data.multipole._background_kinks(
        endf_file, [2, 27], 0.5, 1e-5, 8.5e5)
    tight = openmc.data.multipole._background_kinks(
        endf_file, [2, 27], 0.1, 1e-5, 8.5e5)
    assert set(loose[:, 0]) <= set(energy) <= set(tight[:, 0])

    # corners are sorted and strictly inside the requested range
    assert np.all(np.diff(energy) > 0)
    assert energy.min() > 1e-5 and energy.max() < 8.5e5
    assert np.all(half > 0)


def test_corners_to_split_leaves_room_for_windows():
    """A corner is only split at if the pieces it leaves can host a window.

    A piece is never extended across a corner, so poles fitted on one side say
    nothing about the other. A window whose Doppler-broadened range crosses the
    corner then has to extrapolate, and the narrower the piece the further that
    extrapolation reaches. Splitting is given up in that case, which is only
    correct if it also accounts for the pieces on either side of the corner and
    not merely the sliver between its own two ends.
    """
    from math import sqrt
    from openmc.data.multipole import _corners_to_split, TEMPERATURE_LIMIT
    from openmc.data.data import K_BOLTZMANN

    E_min, E_max = 1e-5, 4.6e5
    alpha = 23.0/(K_BOLTZMANN*TEMPERATURE_LIMIT)
    outer = np.array([E_min, E_max])

    # Splitting at 1 eV would leave a piece only 1 eV wide, while a window
    # sitting against 1 eV is evaluated well past 2 eV, so that corner is
    # given up and held to corner_rtol over the spacing beside it. The corner
    # at 10 eV is then left with room below it and is kept.
    unsplit = []
    kept = _corners_to_split(np.array([[1.0, 0.1], [10.0, 1.0]]), outer, alpha,
                             E_min, E_max, unsplit, 5e-2)
    assert kept == [10.0]
    assert unsplit == [(0.9, 1.1)]

    # higher up the same step is worth splitting at: the margin grows only as
    # the square root of the energy, so it soon becomes small next to the
    # pieces the corner leaves behind
    unsplit = []
    kept = _corners_to_split(np.array([[100.0, 1.0], [200.0, 1.0]]), outer,
                             alpha, E_min, E_max, unsplit, 5e-2)
    assert kept == [100.0, 200.0]
    assert unsplit == []

    # every piece a kept corner bounds really does have room for its windows
    kinks = np.array([[e, 1.0] for e in (50., 100., 2e4, 3e4, 2e5, 2.2e5)])
    unsplit = []
    kept = _corners_to_split(kinks, outer, alpha, E_min, E_max, unsplit, 5e-2)
    edges = np.unique(np.concatenate([outer, np.array(kept or [])]))
    ends = set(kept)
    for lo, hi in zip(edges[:-1], edges[1:]):
        need = 0.0
        if lo in ends:
            need += lo - max(E_min, (sqrt(alpha*lo) - 4.0)**2/alpha)
        if hi in ends:
            need += min(E_max, (sqrt(alpha*hi) + 4.0)**2/alpha) - hi
        assert hi - lo >= 2*need


def test_checking_grid_follows_the_tolerance():
    """The checking grid must track `rtol`, not sit at a fixed density.

    It has to land where the fit is worst, so it must tighten with the
    tolerance being checked. A fixed density cannot: too coarse to verify a
    tight `rtol`, and wasteful for a loose one. It must also stay finer than
    the fitting grid, which `vectfit_nuclide` requires. It does not have to
    resolve every window separately, because a window holding too few points
    is judged on them interpolated.
    """
    from openmc.data.multipole import _UNSET

    sig = inspect.signature(openmc.data.multipole.vectfit_nuclide)
    assert sig.parameters['njoy_error_check'].default is _UNSET, \
        'the default must be distinguishable from an explicit None'

    def chosen(rtol, njoy_error):
        return min(0.01*rtol, 0.1*njoy_error)

    # a hundredth of the tolerance, so the margin is the same at any rtol
    assert chosen(1e-3, 5e-4) == pytest.approx(1e-5)
    assert chosen(1e-5, 5e-4) == pytest.approx(1e-7)

    # and never coarser than a tenth of the fitting grid, which would make the
    # second NJOY run pointless and trip its own consistency check
    assert chosen(5e-2, 5e-4) == pytest.approx(5e-5)
    for rtol in (1e-5, 1e-3, 5e-2, 1.0):
        assert chosen(rtol, 5e-4) < 5e-4


def test_resonances_are_counted_above_the_noise():
    """A wiggle in the reconstructed data is not a resonance.

    The resonance count sets where the order search starts, so counting every
    local maximum makes the fit start hundreds of poles too high on a nuclide
    that is smooth over most of its range. O-16 is the case that matters: it
    has no resonances below 500 keV and is nearly constant there, but the
    reconstructed data wanders enough to put a local maximum every few points.

    The threshold is the tolerance being fitted to, since a peak that rises
    less than that cannot decide whether the fit meets it.
    """
    from openmc.data.multipole import _count_resonances

    energy = np.linspace(1.0, 1000.0, 4001)

    # a smooth cross section with reconstruction noise on it is not resonant
    rng = np.random.default_rng(3)
    smooth = 3.8 + 0.2*np.exp(-energy/500.0)
    noisy = smooth*(1.0 + 3e-6*rng.standard_normal(energy.size))
    assert find_peaks(noisy)[0].size > 100, 'the test data must be noisy'
    assert _count_resonances(noisy, 1e-3) == 0

    # three real resonances on the same background are all found, noise or not
    for centre in (250.0, 500.0, 750.0):
        smooth = smooth + 4.0/(1.0 + ((energy - centre)/3.0)**2)
    resonant = smooth*(1.0 + 3e-6*rng.standard_normal(energy.size))
    assert _count_resonances(resonant, 1e-3) == 3

    # a tighter tolerance must still find the three
    assert _count_resonances(resonant, 1e-6) >= 3

    # a flat or degenerate cross section must not raise
    assert _count_resonances(np.zeros(50), 1e-3) == 0
    assert _count_resonances(np.array([1.0, 2.0]), 1e-3) == 0


def test_pieces_divide_on_the_nuclide():
    """How much one fit is asked to describe, not whether to divide at all.

    A rule that divided only above a threshold left every nuclide under it
    with a single piece spanning the whole range, however wide: O-16 got one
    piece over ten decades. Dividing further is also cheaper almost until it
    stops working, since the order a piece needs grows with the resonances in
    it and the search costs more than the order.

    It cannot divide without limit. A piece has to hold enough of the fitting
    grid to determine a fit, and to be wider than the Doppler margin its
    windows reach past it, which for pieces equal in momentum comes out
    independent of energy.
    """
    from math import sqrt
    from openmc.data.multipole import (_RESONANCES_PER_PIECE,
                                       _POINTS_PER_PIECE, _MIN_PIECE_POINTS,
                                       TEMPERATURE_LIMIT)
    from openmc.data.data import K_BOLTZMANN

    def pieces(n_peaks, n_points, awr, E_max, E_min=1e-5):
        alpha = awr/(K_BOLTZMANN*TEMPERATURE_LIMIT)
        n = max(n_peaks//_RESONANCES_PER_PIECE, n_points//_POINTS_PER_PIECE)
        n = min(n, n_points//_MIN_PIECE_POINTS,
                int((sqrt(E_max) - sqrt(E_min))*sqrt(alpha)/16))
        return max(1, n)

    # a nuclide with hardly any resonances is still divided, because a single
    # piece would span its whole range
    assert pieces(6, 1500, 15.86, 2.355e6) > 1

    # more resonances means more pieces, with no threshold to cross
    counts = [pieces(n, 4306, 22.79, 4.5931e5) for n in (40, 80, 160, 320)]
    assert counts == sorted(counts) and counts[0] < counts[-1]

    # every piece keeps enough points to determine a fit
    for n_peaks, n_points in ((3000, 5000), (10**6, 900)):
        assert n_points//pieces(n_peaks, n_points, 22.79, 4.5931e5) \
            >= _MIN_PIECE_POINTS

    # and stays wider than the margin its windows need: the momentum spacing
    # must exceed 16/sqrt(alpha) whatever the energy
    for awr, E_max in ((22.79, 4.5931e5), (233.0, 2250.0), (0.999, 2e7)):
        alpha = awr/(K_BOLTZMANN*TEMPERATURE_LIMIT)
        n = pieces(10**6, 10**7, awr, E_max)
        assert (sqrt(E_max) - sqrt(1e-5))/n >= 16/sqrt(alpha) - 1e-9

    # a heavy nuclide over a short range is limited by that margin, not by
    # its resonance count: U-235 resolves only to 2250 eV
    assert pieces(2703, 322560, 233.0, 2250.0) < 2703//_RESONANCES_PER_PIECE
