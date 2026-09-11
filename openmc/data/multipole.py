from numbers import Real
from io import StringIO
from math import exp, erf, pi, sqrt
from copy import deepcopy

import os
import h5py
import pickle
import numpy as np
from scipy.signal import find_peaks

import openmc.checkvalue as cv
from ..exceptions import DataError
from ..mixin import EqualityMixin
from . import WMP_VERSION, WMP_VERSION_MAJOR
from .data import K_BOLTZMANN
from .neutron import IncidentNeutron
from .resonance import ResonanceRange
from .vectfit import vectfit, evaluate

# Constants that determine which value to access
_MP_EA = 0       # Pole

# Residue indices
_MP_RS = 1       # Residue scattering
_MP_RA = 2       # Residue absorption
_MP_RF = 3       # Residue fission

# Polynomial fit indices
_FIT_S = 0       # Scattering
_FIT_A = 1       # Absorption
_FIT_F = 2       # Fission

# Upper temperature limit (K)
#
# This is not only the temperature the library may be broadened to. Fitting and
# windowing both widen their energy range by four Doppler widths on each side so
# that broadening up to this temperature stays accurate, and that margin grows
# as its square root. The margin in turn sets how narrow a piece may be and
# still hold a window, which decides whether a piece can be split either side of
# a corner in the ENDF background; a corner left inside a piece cannot be fitted
# by a sum of poles at any order. Lowering this limit shrinks the margin and so
# can let a nuclide reach a tolerance it cannot reach at 3000 K, at the cost of
# the temperature range the library is valid over. For Fe-56 the margin at
# 650 keV is about 880 eV at 3000 K against corners 1000 eV apart, too wide to
# split; at 600 K it is about 310 eV, which is not.
TEMPERATURE_LIMIT = 3000

# Logging control
DETAILED_LOGGING = 2


def _faddeeva(z):
    r"""Evaluate the complex Faddeeva function.

    Technically, the value we want is given by the equation:

    .. math::
        w(z) = \frac{i}{\pi} \int_{-\infty}^{\infty} \frac{1}{z - t}
        \exp(-t^2) \text{d}t

    as shown in Equation 63 from Hwang, R. N. "A rigorous pole
    representation of multilevel cross sections and its practical
    applications." Nuclear Science and Engineering 96.3 (1987): 192-209.

    The :func:`scipy.special.wofz` function evaluates
    :math:`w(z) = \exp(-z^2) \text{erfc}(-iz)`. These two forms of the Faddeeva
    function are related by a transformation.

    If we call the integral form :math:`w_\text{int}`, and the function form
    :math:`w_\text{fun}`:

    .. math::
        w_\text{int}(z) =
        \begin{cases}
            w_\text{fun}(z) & \text{for } \text{Im}(z) > 0\\
            -w_\text{fun}(z^*)^* & \text{for } \text{Im}(z) < 0
        \end{cases}

    Parameters
    ----------
    z : complex
        Argument to the Faddeeva function.

    Returns
    -------
    complex
        :math:`\frac{i}{\pi} \int_{-\infty}^{\infty} \frac{1}{z - t} \exp(-t^2)
        \text{d}t`

    """
    from scipy.special import wofz
    if np.angle(z) > 0:
        return wofz(z)
    else:
        return -np.conj(wofz(z.conjugate()))


def _broaden_wmp_polynomials(E, dopp, n):
    r"""Evaluate Doppler-broadened windowed multipole curvefit.

    The curvefit is a polynomial of the form :math:`\frac{a}{E}
    + \frac{b}{\sqrt{E}} + c + d \sqrt{E} + \ldots`

    Parameters
    ----------
    E : float
        Energy to evaluate at.
    dopp : float
        sqrt(atomic weight ratio / kT) in units of eV.
    n : int
        Number of components to the polynomial.

    Returns
    -------
    np.ndarray
        The value of each Doppler-broadened curvefit polynomial term.

    """
    sqrtE = sqrt(E)
    beta = sqrtE * dopp
    half_inv_dopp2 = 0.5 / dopp**2
    quarter_inv_dopp4 = half_inv_dopp2**2

    if beta > 6.0:
        # Save time, ERF(6) is 1 to machine precision.
        # beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
        erf_beta = 1.0
        exp_m_beta2 = 0.0
    else:
        erf_beta = erf(beta)
        exp_m_beta2 = exp(-beta**2)

    # Assume that, for sure, we'll use a second order (1/E, 1/V, const)
    # fit, and no less.

    factors = np.zeros(n)

    factors[0] = erf_beta / E
    factors[1] = 1.0 / sqrtE
    factors[2] = (factors[0] * (half_inv_dopp2 + E)
                  + exp_m_beta2 / (beta * sqrt(pi)))

    # Perform recursive broadening of high order components. range(1, n-2)
    # replaces a do i = 1, n-3.  All indices are reduced by one due to the
    # 1-based vs. 0-based indexing.
    for i in range(1, n-2):
        if i != 1:
            factors[i+2] = (-factors[i-2] * (i - 1.0) * i * quarter_inv_dopp4
                + factors[i] * (E + (1.0 + 2.0 * i) * half_inv_dopp2))
        else:
            factors[i+2] = factors[i]*(E + (1.0 + 2.0 * i) * half_inv_dopp2)

    return factors


def _vectfit_xs(energy, ce_xs, mts, rtol=1e-3, atol=1e-5, check_energy=None,
                check_xs=None, relaxed=None, relaxed_rtol=5e-2,
                excursion=0.1, orders=None, n_vf_iter=30, log=False,
                path_out=None):
    """Convert point-wise cross section to multipole data via vector fitting.

    Parameters
    ----------
    energy : np.ndarray
        Energy array
    ce_xs : np.ndarray
        Point-wise cross sections to be fitted, with shape (number of reactions,
        number of energy points)
    mts : Iterable of int
        Reaction list
    rtol : float, optional
        Maximum relative error tolerance. A fit is accepted only if the
        relative error stays within this bound at every energy point whose
        absolute error also exceeds `atol`. If no order in the search range
        achieves it, a :class:`RuntimeError` is raised. Defaults to 1e-3,
        matching the default tolerance used by NJOY when reconstructing
        point-wise cross sections from ENDF.
    atol : float, optional
        Absolute error tolerance
    check_energy : np.ndarray, optional
        Energies at which the fit is judged. These should come from a finer
        NJOY reconstruction than `energy`, so that the reference is evaluated
        data rather than an interpolant of it, and so that the fit is
        constrained between the fitting points as well as at them. Defaults to
        `energy`, which leaves the behaviour between fitting points checked
        only by `excursion`.
    check_xs : np.ndarray, optional
        Cross sections at `check_energy`, same reactions and order as `ce_xs`.
    relaxed : iterable of 2-tuple, optional
        Energy ranges held to `relaxed_rtol` rather than `rtol`. Use this only
        where the cross section cannot be represented by a sum of poles
        however many are used, as at a corner in the ENDF background, so that
        one such feature does not force the whole piece to a looser tolerance.
    relaxed_rtol : float, optional
        Tolerance over the `relaxed` ranges. Defaults to 5e-2.
    excursion : float, optional
        How far, in relative terms, the fit may stray outside the range of the
        two fitting points bracketing it. This rejects a fit that oscillates
        between the fitting points; it is not an accuracy tolerance. Defaults
        to 0.1.
    orders : Iterable of int, optional
        A list of orders (number of poles) to be searched
    n_vf_iter : int, optional
        Number of maximum VF iterations
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to save the figures to show discrepancies between the original and
        fitted cross sections for different reactions

    Returns
    -------
    tuple
        (poles, residues)

    """
    ne = energy.size
    nmt = len(mts)
    if ce_xs.shape != (nmt, ne):
        raise ValueError('Inconsistent cross section data.')

    # Accuracy is judged against evaluated data, never against an interpolant
    # of it: NJOY only guarantees linear interpolation of its grid to its own
    # reconstruction tolerance, so scoring between grid points would measure
    # that interpolant rather than the fit.
    if check_energy is None:
        check_energy, check_xs = energy, ce_xs
    else:
        inside = (check_energy >= energy[0]) & (check_energy <= energy[-1])
        check_energy, check_xs = check_energy[inside], check_xs[:, inside]
        if check_xs.shape[0] != nmt:
            raise ValueError('Inconsistent check cross section data.')
    check_s = np.sqrt(check_energy)

    # a per-point tolerance, looser only where the cross section cannot be
    # represented by poles at all
    tol = np.full(check_energy.size, float(rtol))
    for lo_x, hi_x in (relaxed or ()):
        tol[(check_energy >= lo_x) & (check_energy <= hi_x)] = relaxed_rtol
    if np.all(tol > rtol):
        raise ValueError('Every checking point is held to the relaxed '
                         'tolerance.')

    # A refined grid guards against the rational fit oscillating between the
    # fitting points, where nothing else constrains it. The bracketing data
    # values give the envelope; no interpolated reference is involved.
    n_finer = 10
    ne_test = (ne - 1)*n_finer + 1
    test_energy = np.interp(np.arange(ne_test),
                            np.arange(ne_test, step=n_finer), energy)
    test_energy[[0, -1]] = energy[[0, -1]]  # avoid numerical issue
    segment = np.minimum(np.arange(ne_test)//n_finer, ne - 2)
    envelope_lo = np.minimum(ce_xs[:, :-1], ce_xs[:, 1:])[:, segment]
    envelope_hi = np.maximum(ce_xs[:, :-1], ce_xs[:, 1:])[:, segment]
    envelope_lo = envelope_lo*(1.0 - excursion) - atol
    envelope_hi = envelope_hi*(1.0 + excursion) + atol

    if log:
        print(f"\tenergy: {energy[0]:.3e} to {energy[-1]:.3e} eV ({ne} points)")
        print(f"\terror tolerance: rtol={rtol}, atol={atol}")
        print(f"\tchecked at {check_energy.size} points")

    # transform xs (sigma) and energy (E) to f (sigma*E) and s (sqrt(E)) to be
    # compatible with the multipole representation
    f = ce_xs * energy
    s = np.sqrt(energy)
    test_s = np.sqrt(test_energy)

    # inverse weighting is used for minimizing the relative deviation instead of
    # absolute deviation in vector fitting
    with np.errstate(divide='ignore'):
        weight = 1.0/f

    # avoid too large weights which will harm the fitting accuracy
    min_cross_section = 1e-7
    for i in range(nmt):
        if np.all(ce_xs[i] <= min_cross_section):
            weight[i] = 1.0
        elif np.any(ce_xs[i] <= min_cross_section):
            weight[i, ce_xs[i] <= min_cross_section] = \
               max(weight[i, ce_xs[i] > min_cross_section])

    # detect peaks (resonances) and determine VF order search range
    peaks, _ = find_peaks(ce_xs[0] + ce_xs[1])
    n_peaks = peaks.size
    if orders is not None:
        # make sure orders are even integers, searched in increasing order
        orders = sorted({int(i/2)*2 for i in orders if i >= 2})
    else:
        lowest_order = max(2, 2*n_peaks)
        highest_order = max(200, 4*n_peaks)
        orders = list(range(lowest_order, highest_order + 1, 2))

    # The constrained least-squares system in the pole identification step
    # needs at least as many samples as poles; beyond that the fit is not
    # determined by the data anyway.
    order_limit = 2*((ne - 1)//2)
    orders = [o for o in orders if o <= order_limit]
    if not orders:
        raise ValueError(
            f"Energy range {energy[0]:.3e} to {energy[-1]:.3e} eV has only "
            f"{ne} points, too few to fit even two poles.")

    if log:
        print(f"Found {n_peaks} peaks")
        print(f"Fitting orders from {orders[0]} to {orders[-1]}")

    # perform VF with increasing orders
    found_ideal = False
    n_discarded = 0  # for accelation, number of discarded searches
    best_maxre = best_severity = np.inf
    best_poles = None
    for i, order in enumerate(orders):
        if log:
            print(f"Order={order}({i}/{len(orders)})")
        # initial guessed poles
        poles_r = np.linspace(s[0], s[-1], order//2)
        poles = poles_r + poles_r*0.01j
        poles = np.sort(np.append(poles, np.conj(poles)))

        maxre_before = best_severity
        # fitting iteration
        for i_vf in range(n_vf_iter):
            if log >= DETAILED_LOGGING:
                print(f"VF iteration {i_vf + 1}/{n_vf_iter}")

            # call vf
            poles, residues, *_ = vectfit(f, s, poles, weight)

            # convert real pole to conjugate pairs
            n_real_poles = 0
            new_poles = []
            for p in poles:
                p_r, p_i = np.real(p), np.imag(p)
                if (s[0] <= p_r <= s[-1]) and p_i == 0.:
                    new_poles += [p_r+p_r*0.01j, p_r-p_r*0.01j]
                    n_real_poles += 1
                else:
                    new_poles += [p]
            new_poles = np.array(new_poles)
            # re-calculate residues if poles changed
            if n_real_poles > 0:
                if log >= DETAILED_LOGGING:
                    print(f"  # real poles: {n_real_poles}")
                new_poles, residues, *_ = \
                      vectfit(f, s, new_poles, weight, skip_pole_update=True)

            # assess against the evaluated data. A fit with negative cross
            # sections or a non-finite result is unusable, whatever its error.
            check_fit = evaluate(check_s, new_poles, residues) / check_energy
            abserr = np.abs(check_fit - check_xs)
            with np.errstate(invalid='ignore', divide='ignore'):
                relerr = abserr / check_xs
                if np.any(np.isnan(abserr)) or np.any(check_fit < -atol):
                    maxre = severity = np.inf
                elif np.all(abserr <= atol):
                    maxre = severity = 0.
                else:
                    scored = np.where(abserr > atol, relerr, 0.0)
                    # measured against each point's own tolerance, so a corner
                    # held to a looser one does not dominate the search
                    severity = np.max(scored/tol)
                    maxre = np.max(scored)

            # reject a fit that swings outside the data envelope between the
            # fitting points
            if np.isfinite(maxre):
                test_xs = evaluate(test_s, new_poles, residues) / test_energy
                if np.any(test_xs > envelope_hi) or np.any(test_xs < envelope_lo):
                    if log >= DETAILED_LOGGING:
                        print("  Rejected: fit oscillates between grid points")
                    maxre = severity = np.inf

            if log >= DETAILED_LOGGING:
                print(f"  # poles: {new_poles.size}")
                print(f"  Max relative error: {maxre * 100:.3f}%")

            if severity < best_severity:
                best_severity = severity
                if log >= DETAILED_LOGGING:
                    print("  Best so far!")
                best_maxre = maxre
                best_poles, best_residues = new_poles, residues
                best_check_fit, best_relerr = check_fit, relerr
            elif log >= DETAILED_LOGGING:
                print("  Discarded!")

            # the search is done as soon as every point is within tolerance;
            # because orders are tried in increasing order, this is also the
            # smallest number of poles that achieves it
            if severity <= 1.0:
                if log:
                    print("Found ideal results. Stop!")
                found_ideal = True
                break

        if found_ideal:
            break

        # acceleration: keep adding poles for as long as the maximum relative
        # error keeps coming down, then give up. Stopping here never returns a
        # poor fit because the tolerance check after the loop raises.
        if best_severity < maxre_before:
            n_discarded = 0
        else:
            n_discarded += 1
            if n_discarded >= 10:
                if log:
                    print("Maximum relative error stopped improving. Stop!")
                break

    if best_poles is None:
        raise RuntimeError(
            f"Vector fitting produced no usable fit for energy range "
            f"{energy[0]:.3e} to {energy[-1]:.3e} eV with orders "
            f"{orders[0]} to {orders[-1]}. This usually means every candidate "
            f"gave negative cross sections or a non-finite result.")

    if best_severity > 1.0:
        raise RuntimeError(
            f"Vector fitting could not reach the maximum relative error "
            f"tolerance (rtol={rtol:.3g}) for energy range "
            f"{energy[0]:.3e} to {energy[-1]:.3e} eV with orders "
            f"{orders[0]} to {orders[-1]}. The best fit had a maximum "
            f"relative error of {best_maxre:.3%} using {best_poles.size} "
            f"poles. Either relax 'rtol' or widen the pole search range "
            f"with 'orders'.")

    # merge conjugate poles
    real_idx = []
    conj_idx = []
    found_conj = False
    for i, p in enumerate(best_poles):
        if found_conj:
            found_conj = False
            continue
        if np.imag(p) == 0.:
            real_idx.append(i)
        else:
            if i < best_poles.size and np.conj(p) == best_poles[i + 1]:
                found_conj = True
                conj_idx.append(i)
            else:
                raise RuntimeError("Complex poles are not conjugate!")
    if log:
        print("Found {} real poles and {} conjugate complex pairs.".format(
               len(real_idx), len(conj_idx)))
    mp_poles = best_poles[real_idx + conj_idx]
    mp_residues = np.concatenate((best_residues[:, real_idx],
                                  best_residues[:, conj_idx]*2), axis=1)/1j
    if log:
        print(f"Final number of poles: {mp_poles.size}")
        print(f"Maximum relative error: {best_maxre:.3%} (rtol={rtol:.3g})")

    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        for i, mt in enumerate(mts):
            if not check_xs[i].any():
                continue
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()
            lns1 = ax1.loglog(check_energy, check_xs[i], 'g', label="ACE xs")
            lns2 = ax1.loglog(check_energy, best_check_fit[i], 'b', label="VF xs")
            ax2 = ax1.twinx()
            lns3 = ax2.loglog(check_energy, best_relerr[i], 'r',
                              label="Relative error", alpha=0.5)
            lns = lns1 + lns2 + lns3
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='best')
            ax1.set_xlabel('energy (eV)')
            ax1.set_ylabel('cross section (b)', color='b')
            ax1.tick_params('y', colors='b')
            ax2.set_ylabel('relative error', color='r')
            ax2.tick_params('y', colors='r')

            plt.title(f"MT {mt} vector fitted with {mp_poles.size} poles")
            fig.tight_layout()
            fig_file = os.path.join(path_out, "{:.0f}-{:.0f}_MT{}.png".format(
                                    energy[0], energy[-1], mt))
            plt.savefig(fig_file)
            plt.close()
            if log:
                print(f"Saved figure: {fig_file}")

    return (mp_poles, mp_residues)

def _background_kinks(endf_file, mts, threshold, E_min, E_max):
    """Energies where the ENDF background cross section has a sharp corner.

    Backgrounds in MF3 are tabulated on a coarse grid and interpolated
    linearly, so the cross section has discontinuous slope at every tabulated
    energy. A sum of poles is analytic and cannot reproduce a corner at any
    order, so pieces must not straddle the pronounced ones. Both ends of an
    interval whose background steps by more than `threshold` are returned.

    Parameters
    ----------
    endf_file : str
        Path to ENDF evaluation
    mts : Iterable of int
        Reactions being fitted
    threshold : float
        Relative step in the background above which a corner is considered
        pronounced
    E_min, E_max : float
        Energy range of interest

    Returns
    -------
    np.ndarray
        Sorted energies, excluding the ends of the range

    """
    from .endf import Evaluation, get_head_record, get_tab1_record

    # MT27 (absorption) is a derived sum and has no MF3 section of its own
    wanted = set()
    for mt in mts:
        wanted.update({2: [2], 27: [102, 103, 107], 18: [18]}.get(mt, [mt]))

    ev = Evaluation(endf_file)
    kinks = []
    for mt in sorted(wanted):
        if (3, mt) not in ev.section:
            continue
        file_obj = StringIO(ev.section[3, mt])
        get_head_record(file_obj)
        _, tab = get_tab1_record(file_obj)
        e, xs = np.asarray(tab.x), np.asarray(tab.y)
        scale = np.maximum(np.abs(xs[:-1]), np.abs(xs[1:]))
        with np.errstate(invalid='ignore', divide='ignore'):
            step = np.abs(np.diff(xs))/np.where(scale > 0, scale, np.inf)
        for i in np.flatnonzero(step > threshold):
            kinks.extend((e[i], e[i+1]))

    kinks = np.unique([k for k in kinks if E_min < k < E_max])
    return kinks


def vectfit_nuclide(endf_file, njoy_error=5e-4, njoy_error_check=1e-6,
                    vf_pieces=None, kink_threshold=0.25, rtol=1e-3,
                    corner_rtol=5e-2, log=False, path_out=None,
                    mp_filename=None, **kwargs):
    r"""Generate multipole data for a nuclide from ENDF.

    Parameters
    ----------
    endf_file : str
        Path to ENDF evaluation
    njoy_error : float, optional
        Fractional error tolerance for processing the point-wise data that is
        fitted.
    njoy_error_check : float or None, optional
        Fractional error tolerance for a second, finer NJOY reconstruction used
        only to judge the fit. Because NJOY reconstructs to `errmax`, which
        defaults to ten times the requested tolerance, a fit cannot be verified
        to better than roughly ten times `njoy_error` on the fitting grid
        alone. Evaluating the fit on a finer grid checks it between the fitting
        points against evaluated data instead of an interpolant. Set to None to
        skip the second NJOY run and judge the fit on the fitting grid only.
        Defaults to 1e-6.
    vf_pieces : integer, optional
        Number of equal-in-momentum spaced energy pieces for data fitting
    kink_threshold : float or None, optional
        Relative step in the ENDF background cross section above which an extra
        piece boundary is inserted. Backgrounds in MF3 are interpolated
        linearly between tabulated energies, so the cross section has a corner
        at each of them, and a sum of poles cannot reproduce a corner at any
        order. Splitting there puts the corners on piece boundaries instead of
        inside a piece. Set to None to disable. Defaults to 0.25.

        A corner can only be split out where the interval it spans is wide
        enough to hold a window, which `TEMPERATURE_LIMIT` governs; see that
        constant. Corners too narrow to split are left inside a piece, which is
        then fitted to `corner_rtol`.
    rtol : float, optional
        Maximum relative error tolerance for the fit, passed to
        :func:`openmc.data.multipole._vectfit_xs`. Defaults to 1e-3.
    corner_rtol : float, optional
        Tolerance around a corner that could not be split out. A sum of poles
        cannot reproduce a corner at any order, so a piece holding one cannot
        meet `rtol` and would otherwise abort generation. Only the corner's
        own neighbourhood is held to this looser tolerance; the rest of the
        piece must still meet `rtol`, so one narrow feature does not degrade a
        whole piece. The error each piece achieves over its full range,
        exempt neighbourhoods included, is recorded under ``max_error`` in the
        returned data. Defaults to 5e-2.
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to write out mutipole data file and vector fitting figures
    mp_filename : str, optional
        File name to write out multipole data
    **kwargs
        Keyword arguments passed to :func:`openmc.data.multipole._vectfit_xs`

    Returns
    -------
    mp_data
        Dictionary containing necessary multipole data of the nuclide

    """

    # ======================================================================
    # PREPARE POINT-WISE XS

    # make 0K ACE data using njoy
    if log:
        print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

    nuc_ce = IncidentNeutron.from_njoy(endf_file, temperatures=[0.0],
             error=njoy_error, broadr=False, heatr=False, purr=False)

    nuc_check = None
    if njoy_error_check is not None:
        if njoy_error_check >= njoy_error:
            raise ValueError('njoy_error_check must be finer than njoy_error')
        if log:
            print("Running NJOY again for the finer checking grid "
                  f"(error={njoy_error_check})...")
        nuc_check = IncidentNeutron.from_njoy(
            endf_file, temperatures=[0.0], error=njoy_error_check,
            broadr=False, heatr=False, purr=False)

    if log:
        print("Parsing cross sections within resolved resonance range...")

    # Determine upper energy: the lower of RRR upper bound and first threshold
    endf_res = IncidentNeutron.from_endf(endf_file).resonances
    if hasattr(endf_res, 'resolved') and \
       hasattr(endf_res.resolved, 'energy_max') and \
       type(endf_res.resolved) is not ResonanceRange:
        E_max = endf_res.resolved.energy_max
    elif hasattr(endf_res, 'unresolved') and \
         hasattr(endf_res.unresolved, 'energy_min'):
        E_max = endf_res.unresolved.energy_min
    else:
        E_max = nuc_ce.energy['0K'][-1]
    E_max_idx = np.searchsorted(nuc_ce.energy['0K'], E_max, side='right') - 1
    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs['0K'], '_threshold_idx'):
            threshold_idx = nuc_ce.reactions[mt].xs['0K']._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx

    # parse energy and cross sections
    energy = nuc_ce.energy['0K'][:E_max_idx + 1]
    E_min, E_max = energy[0], energy[-1]
    n_points = energy.size
    total_xs = nuc_ce[1].xs['0K'](energy)
    elastic_xs = nuc_ce[2].xs['0K'](energy)

    try:
        absorption_xs = nuc_ce[27].xs['0K'](energy)
    except KeyError:
        absorption_xs = np.zeros_like(total_xs)

    fissionable = False
    try:
        fission_xs = nuc_ce[18].xs['0K'](energy)
        fissionable = True
    except KeyError:
        pass

    # make vectors
    if fissionable:
        ce_xs = np.vstack((elastic_xs, absorption_xs, fission_xs))
        mts = [2, 27, 18]
    else:
        ce_xs = np.vstack((elastic_xs, absorption_xs))
        mts = [2, 27]

    # the same reactions on the finer grid, used only to judge the fit
    check_energy = check_xs = None
    if nuc_check is not None:
        check_energy = nuc_check.energy['0K']
        check_energy = check_energy[(check_energy >= E_min) &
                                    (check_energy <= E_max)]
        rows = []
        for mt in mts:
            try:
                rows.append(nuc_check[mt].xs['0K'](check_energy))
            except KeyError:
                rows.append(np.zeros_like(check_energy))
        check_xs = np.vstack(rows)

    if log:
        print(f"  MTs: {mts}")
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({n_points} points)")
        if check_energy is not None:
            print(f"  Checking grid: {check_energy.size} points")

    # ======================================================================
    # PERFORM VECTOR FITTING

    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs)
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100*10000:
            vf_pieces = max(5, n_peaks // 50,  n_points // 2000)
        else:
            vf_pieces = 1
    piece_width = (sqrt(E_max) - sqrt(E_min)) / vf_pieces

    # Boundaries of the equal-in-momentum pieces, plus the pronounced corners
    # in the background. A piece may be extended past its boundary to leave
    # room for Doppler broadening, but never past a corner, which would put the
    # corner back inside a piece.
    alpha = nuc_ce.atomic_weight_ratio/(K_BOLTZMANN*TEMPERATURE_LIMIT)
    bounds = [(sqrt(E_min) + piece_width*i)**2 for i in range(vf_pieces + 1)]
    bounds[0], bounds[-1] = E_min, E_max
    kinks = np.array([])
    unsplit = []
    if kink_threshold is not None:
        kinks = _background_kinks(endf_file, mts, kink_threshold, E_min, E_max)
        # A window is evaluated over its own width plus a margin of four
        # Doppler widths on each side, and takes its poles from a single
        # piece, so a piece narrower than that margin cannot host any window
        # at all however many are used. Splitting there would only produce
        # poles that every window has to extrapolate from.
        keep = []
        for lo_k, hi_k in zip(kinks[::2], kinks[1::2]):
            margin = ((sqrt(alpha*hi_k) + 4.0)**2/alpha - hi_k
                      + lo_k - (sqrt(alpha*lo_k) - 4.0)**2/alpha)
            if hi_k - lo_k >= 2*margin:
                keep.extend((lo_k, hi_k))
            else:
                # the corner and the margin a window needs around it
                unsplit.append((lo_k - margin/2, hi_k + margin/2))
                if log:
                    print(f"  Not splitting at the corner spanning {lo_k:.6g} "
                          f"to {hi_k:.6g} eV: {hi_k - lo_k:.0f} eV is too "
                          f"narrow for a window, which needs about "
                          f"{2*margin:.0f} eV here; that piece will be fitted "
                          f"to corner_rtol={corner_rtol:.3g}")
        kinks = np.array(keep)
        if log and kinks.size:
            print(f"  Splitting at {kinks.size} background corners: "
                  + ", ".join(f"{k:.4g}" for k in kinks) + " eV")
    bounds = np.array(bounds)
    if kinks.size:
        # an equally spaced boundary falling inside a stepping interval would
        # only carve a sliver off it
        inside = np.zeros(bounds.size, dtype=bool)
        for lo_k, hi_k in zip(kinks[::2], kinks[1::2]):
            inside |= (bounds > lo_k) & (bounds < hi_k)
        bounds = bounds[~inside]
    bounds = np.unique(np.concatenate([bounds, kinks]))
    n_pieces = bounds.size - 1


    poles, residues, max_error = [], [], []
    atol = kwargs.get('atol', 1e-5)
    # VF piece by piece
    for i_piece in range(n_pieces):
        lo_bound, hi_bound = bounds[i_piece], bounds[i_piece + 1]
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{n_pieces} "
                  f"({lo_bound:.4g} to {hi_bound:.4g} eV)...")
        # start E of this piece, extended for Doppler broadening
        if i_piece == 0 or sqrt(alpha*lo_bound) < 4.0:
            e_start = E_min
        else:
            e_start = max(E_min, (sqrt(alpha*lo_bound) - 4.0)**2/alpha)
        # end E of this piece, extended for Doppler broadening
        e_end = min(E_max, (sqrt(alpha*hi_bound) + 4.0)**2/alpha)
        # do not extend across a corner
        if kinks.size:
            below = kinks[kinks <= lo_bound]
            above = kinks[kinks >= hi_bound]
            if below.size:
                e_start = max(e_start, below[-1])
            if above.size:
                e_end = min(e_end, above[0])
        e_start_idx = max(0, np.searchsorted(energy, e_start, side='right') - 1)
        e_end_idx = np.searchsorted(energy, e_end, side='left') + 1
        lo_idx, hi_idx = e_start_idx, min(e_end_idx, n_points - 1)
        if kinks.size:
            # the padding above is deliberately generous; do not let it reach
            # back across a corner that this piece was split at
            while lo_idx < hi_idx and energy[lo_idx] < e_start:
                lo_idx += 1
            while hi_idx > lo_idx and energy[hi_idx] > e_end:
                hi_idx -= 1
        e_idx = range(lo_idx, hi_idx + 1)

        if check_energy is None:
            c_energy = c_xs = None
        else:
            lo, hi = energy[e_idx][0], energy[e_idx][-1]
            c_mask = (check_energy >= lo) & (check_energy <= hi)
            c_energy, c_xs = check_energy[c_mask], check_xs[:, c_mask]

        # A corner that could not be split out cannot be fitted by poles at
        # any order. Exempt its neighbourhood from the tolerance rather than
        # loosening the tolerance over the whole piece, then hold that
        # neighbourhood to corner_rtol on its own.
        lo_p, hi_p = energy[e_idx][0], energy[e_idx][-1]
        relaxed = [(a, b) for a, b in unsplit if a < hi_p and b > lo_p]
        if relaxed and log:
            print(f"  holds corners that cannot be split out; holding "
                  + ", ".join(f"{a:.6g}-{b:.6g}" for a, b in relaxed)
                  + f" eV to {corner_rtol:.3g}")

        p, r = _vectfit_xs(energy[e_idx], ce_xs[:, e_idx], mts, log=log,
                           rtol=rtol, check_energy=c_energy, check_xs=c_xs,
                           relaxed=relaxed, relaxed_rtol=corner_rtol,
                           path_out=path_out, **kwargs)

        # record what this piece actually achieved against the evaluated data
        if c_energy is not None and c_energy.size:
            fit = evaluate(np.sqrt(c_energy), p, r*1j)/c_energy
            abserr = np.abs(fit - c_xs)
            with np.errstate(invalid='ignore', divide='ignore'):
                achieved = float(np.max(np.where(abserr > atol,
                                                 abserr/np.abs(c_xs), 0.0)))
        else:
            achieved = float('nan')
        max_error.append(achieved)
        if log:
            print(f"  reproduces the point-wise data to {achieved:.3%}")
        if relaxed and achieved > corner_rtol:
            raise RuntimeError(
                f"Energy range {lo_p:.3e} to {hi_p:.3e} eV holds a corner in "
                f"the ENDF background that cannot be split out, and the fit "
                f"there reaches only {achieved:.3%}, outside "
                f"corner_rtol={corner_rtol:.3g}.")

        poles.append(p)
        residues.append(r)

    # collect multipole data into a dictionary
    mp_data = {"name": nuc_ce.name,
               "AWR": nuc_ce.atomic_weight_ratio,
               "E_min": E_min,
               "E_max": E_max,
               "bounds": bounds,
               "max_error": np.array(max_error),
               "check_energy": check_energy,
               "check_xs": check_xs,
               "relaxed": np.asarray(unsplit, dtype=float).reshape(-1, 2),
               "poles": poles,
               "residues": residues}

    if log:
        worst = np.nanmax(max_error) if max_error else float('nan')
        print(f"Multipole data reproduces the point-wise data to {worst:.3%}")
        over = [i for i, e in enumerate(max_error) if e > rtol]
        if over:
            print(f"  {len(over)} of {n_pieces} pieces exceed rtol={rtol:.3g}, "
                  "each holding a corner in the background that cannot be "
                  "fitted by poles:")
            for i in over:
                print(f"    {bounds[i]:.6g} to {bounds[i+1]:.6g} eV: "
                      f"{max_error[i]:.3%}")

    # dump multipole data to file
    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not mp_filename:
            mp_filename = f"{nuc_ce.name}_mp.pickle"
        mp_filename = os.path.join(path_out, mp_filename)
        with open(mp_filename, 'wb') as f:
            pickle.dump(mp_data, f)
        if log:
            print(f"Dumped multipole data to file: {mp_filename}")

    return mp_data


def _windowing(mp_data, n_cf, rtol=1e-3, atol=1e-5, corner_rtol=5e-2,
               n_win=None, spacing=None, log=False):
    """Generate windowed multipole library from multipole data with specific
        settings of window size, curve fit order, etc.

    Parameters
    ----------
    mp_data : dict
        Multipole data
    n_cf : int
        Curve fitting order
    rtol : float, optional
        Maximum relative error tolerance
    atol : float, optional
        Minimum absolute error tolerance
    corner_rtol : float, optional
        Tolerance over the energy ranges `mp_data` records as holding a corner
        in the ENDF background, which a sum of poles and a polynomial cannot
        reproduce at any order. Defaults to 5e-2.
    n_win : int, optional
        Number of equal-in-mementum spaced energy windows
    spacing : float, optional
        Inner window spacing (sqrt energy space)
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)

    Returns
    -------
    openmc.data.WindowedMultipole
        Resonant cross sections represented in the windowed multipole
        format.

    """
    # unpack multipole data
    name = mp_data["name"]
    awr = mp_data["AWR"]
    E_min = mp_data["E_min"]
    E_max = mp_data["E_max"]
    mp_poles = mp_data["poles"]
    mp_residues = mp_data["residues"]

    n_pieces = len(mp_poles)
    # Piece boundaries are not equally spaced when extra ones were inserted at
    # corners in the background, so they are carried with the data. Older
    # multipole data has none, in which case they are equal in momentum.
    bounds = mp_data.get("bounds")
    if bounds is None:
        width = (sqrt(E_max) - sqrt(E_min)) / n_pieces
        bounds = np.array([(sqrt(E_min) + width*i)**2
                           for i in range(n_pieces + 1)])
    bounds = np.asarray(bounds, dtype=float)
    bounds_sqrt = np.sqrt(bounds)
    # Windows are judged against the evaluated cross sections when they are
    # available. Judging them against the multipole form instead only asks
    # whether the curve fit reproduces the poles it was built from, which says
    # nothing about the library's accuracy and cannot detect a window whose
    # poles come from a piece fitted somewhere else.
    check_energy = mp_data.get("check_energy")
    check_xs = mp_data.get("check_xs")
    relaxed = np.asarray(mp_data.get("relaxed", np.empty((0, 2))), dtype=float)
    piece_width = np.min(np.diff(bounds_sqrt))
    alpha = awr / (K_BOLTZMANN*TEMPERATURE_LIMIT)

    # determine window size
    if n_win is None:
        if spacing is not None:
            # ensure the windows are within the multipole energy range
            n_win = int((sqrt(E_max) - sqrt(E_min)) / spacing)
            E_max = (sqrt(E_min) + n_win*spacing)**2
        else:
            n_win = 1000
    # inner window size
    spacing = (sqrt(E_max) - sqrt(E_min)) / n_win
    # make sure inner window size is smaller than energy piece size
    if spacing > piece_width:
        raise ValueError(
            f'Window spacing {spacing:.4g} is larger than the narrowest piece '
            f'{piece_width:.4g} (both in momentum), so a window would need '
            f'poles from outside the piece it was assigned. Use at least '
            f'{int(np.ceil((sqrt(E_max) - sqrt(E_min))/piece_width))} windows.')

    if log:
        print("Windowing:")
        print(f"  config: # windows={n_win}, spacing={spacing}, CF order={n_cf}")
        print(f"  error tolerance: rtol={rtol}, atol={atol}")

    # sort poles (and residues) by the real component of the pole
    for ip in range(n_pieces):
        indices = mp_poles[ip].argsort()
        mp_poles[ip] = mp_poles[ip][indices]
        mp_residues[ip] = mp_residues[ip][:, indices]

    # initialize an array to record whether each pole is used or not
    poles_unused = [np.ones_like(p, dtype=int) for p in mp_poles]

    # optimize the windows: the goal is to find the least set of significant
    # consecutive poles and curve fit coefficients to reproduce cross section
    win_data = []
    for iw in range(n_win):
        if log >= DETAILED_LOGGING:
            print(f"Processing window {iw + 1}/{n_win}...")

        # inner window boundaries
        inbegin = sqrt(E_min) + spacing * iw
        inend = inbegin + spacing
        incenter = (inbegin + inend) / 2.0
        # extend window energy range for Doppler broadening
        if iw == 0 or sqrt(alpha)*inbegin < 4.0:
            e_start = inbegin**2
        else:
            e_start = max(E_min, (sqrt(alpha)*inbegin - 4.0)**2/alpha)
        e_end = min(E_max, (sqrt(alpha)*inend + 4.0)**2/alpha)

        # locate piece and relevant poles: the piece whose boundaries
        # bracket the centre of this window
        i_piece = int(np.clip(np.searchsorted(bounds_sqrt, incenter) - 1,
                              0, n_pieces - 1))
        poles, residues = mp_poles[i_piece], mp_residues[i_piece]
        n_poles = poles.size

        # energy points for fitting, and the reference to fit against
        in_check = None
        if check_energy is not None:
            in_check = np.flatnonzero((check_energy >= e_start) &
                                      (check_energy <= e_end))
            # thin a dense window, and fall back if the grid is too sparse
            if in_check.size > 10000:
                in_check = in_check[::int(np.ceil(in_check.size/10000))]
            if in_check.size < max(100, n_cf + 2):
                in_check = None
        if in_check is None:
            # no evaluated data here: the multipole form is all there is
            n_points = min(max(100, int((e_end - e_start)*4)), 10000)
            energy_sqrt = np.linspace(np.sqrt(e_start), np.sqrt(e_end), n_points)
            energy = energy_sqrt**2
            # note the residue terms in the multipole and vector fitting
            # representations differ by a 1j
            xs_ref = evaluate(energy_sqrt, poles, residues*1j) / energy
        else:
            energy = check_energy[in_check]
            energy_sqrt = np.sqrt(energy)
            xs_ref = check_xs[:, in_check]

        # a corner in the background is looser, as in the fitting stage
        tol = np.full(energy.size, float(rtol))
        for lo_x, hi_x in relaxed:
            tol[(energy >= lo_x) & (energy <= hi_x)] = corner_rtol

        # curve fit matrix
        matrix = np.vstack([energy**(0.5*i - 1) for i in range(n_cf + 1)]).T

        # start from 0 poles, initialize pointers to the center nearest pole
        center_pole_ind = np.argmin((np.fabs(poles.real - incenter)))
        lp = rp = center_pole_ind
        while True:
            if log >= DETAILED_LOGGING:
                print(f"Trying poles {lp} to {rp}")

            # calculate the cross sections contributed by the windowed poles
            if rp > lp:
                xs_wp = evaluate(energy_sqrt, poles[lp:rp],
                                    residues[:, lp:rp]*1j) / energy
            else:
                xs_wp = np.zeros_like(xs_ref)

            # do least square curve fit on the remains
            coefs = np.linalg.lstsq(matrix, (xs_ref - xs_wp).T, rcond=None)[0]
            xs_fit = (matrix @ coefs).T

            # assess the result
            abserr = np.abs(xs_fit + xs_wp - xs_ref)
            with np.errstate(invalid='ignore', divide='ignore'):
                relerr = abserr / xs_ref
            if not np.any(np.isnan(abserr)):
                scored = np.where(abserr > atol, relerr, 0.0)
                # every point in the window must be within its own tolerance
                if np.max(scored/tol) <= 1.0:
                    # meet tolerances
                    if log >= DETAILED_LOGGING:
                        print("Accuracy satisfied.")
                    break

            # we expect pure curvefit will succeed for the first window
            # TODO: find the energy boundary below which no poles are allowed
            if iw == 0:
                raise RuntimeError('Pure curvefit failed for the first window!')

            # try to include one more pole (next center nearest)
            if lp <= 0 and rp >= n_poles:
                # every pole of the piece is already in and the window still
                # misses; against evaluated data this is reachable, so stop
                raise RuntimeError(
                    f'Window {iw + 1} of {n_win} covering {e_start:.4g} to '
                    f'{e_end:.4g} eV cannot reach the tolerance with all '
                    f'{n_poles} poles of its piece and a curve fit of order '
                    f'{n_cf}.')
            if rp >= n_poles:
                lp -= 1
            elif lp <= 0 or poles[rp] - incenter <= incenter - poles[lp - 1]:
                rp += 1
            else:
                lp -= 1

        # save data for this window
        win_data.append((i_piece, lp, rp, coefs))

        # mark the windowed poles as used poles
        poles_unused[i_piece][lp:rp] = 0

    # flatten and shrink by removing unused poles
    data = []  # used poles and residues
    for ip in range(n_pieces):
        used = (poles_unused[ip] == 0)
        # stack poles and residues for library format
        data.append(np.vstack([mp_poles[ip][used], mp_residues[ip][:, used]]).T)
    # stack poles/residues in sequence vertically
    data = np.vstack(data)
    # new start/end pole indices
    windows = []
    curvefit = []
    for iw in range(n_win):
        ip, lp, rp, coefs = win_data[iw]
        # adjust indices and change to 1-based for the library format
        n_prev_poles = sum([poles_unused[i].size for i in range(ip)])
        n_unused = sum([(poles_unused[i] == 1).sum() for i in range(ip)]) + \
                  (poles_unused[ip][:lp] == 1).sum()
        lp += n_prev_poles - n_unused + 1
        rp += n_prev_poles - n_unused
        windows.append([lp, rp])
        curvefit.append(coefs)

    # construct the WindowedMultipole object
    wmp = WindowedMultipole(name)
    wmp.spacing = spacing
    wmp.sqrtAWR = sqrt(awr)
    wmp.E_min = E_min
    wmp.E_max = E_max
    wmp.data = data
    wmp.windows = np.asarray(windows)
    wmp.curvefit = np.asarray(curvefit)
    # TODO: check if Doppler brodening of the polynomial curvefit is negligible
    wmp.broaden_poly = np.ones((n_win,), dtype=bool)

    return wmp


class WindowedMultipole(EqualityMixin):
    """Resonant cross sections represented in the windowed multipole format.

    Parameters
    ----------
    name : str
        Name of the nuclide using the GNDS naming convention

    Attributes
    ----------
    name : str
        Name of the nuclide using the GNDS naming convention
    spacing : float
        The width of each window in sqrt(E)-space.  For example, the frst window
        will end at (sqrt(E_min) + spacing)**2 and the second window at
        (sqrt(E_min) + 2*spacing)**2.
    sqrtAWR : float
        Square root of the atomic weight ratio of the target nuclide.
    E_min : float
        Lowest energy in eV the library is valid for.
    E_max : float
        Highest energy in eV the library is valid for.
    data : np.ndarray
        A 2D array of complex poles and residues.  data[i, 0] gives the energy
        at which pole i is located.  data[i, 1:] gives the residues associated
        with the i-th pole.  There are 3 residues, one each for the scattering,
        absorption, and fission channels.
    windows : np.ndarray
        A 2D array of Integral values.  windows[i, 0] - 1 is the index of the
        first pole in window i. windows[i, 1] - 1 is the index of the last pole
        in window i.
    broaden_poly : np.ndarray
        A 1D array of boolean values indicating whether or not the polynomial
        curvefit in that window should be Doppler broadened.
    curvefit : np.ndarray
        A 3D array of Real curvefit polynomial coefficients.  curvefit[i, 0, :]
        gives coefficients for the scattering cross section in window i.
        curvefit[i, 1, :] gives absorption coefficients and curvefit[i, 2, :]
        gives fission coefficients.  The polynomial terms are increasing powers
        of sqrt(E) starting with 1/E e.g:
        a/E + b/sqrt(E) + c + d sqrt(E) + ...

    """
    def __init__(self, name):
        self.name = name
        self.spacing = None
        self.sqrtAWR = None
        self.E_min = None
        self.E_max = None
        self.data = None
        self.windows = None
        self.broaden_poly = None
        self.curvefit = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        cv.check_type('name', name, str)
        self._name = name

    @property
    def fit_order(self):
        return self.curvefit.shape[1] - 1

    @property
    def fissionable(self):
        return self.data.shape[1] == 4

    @property
    def n_poles(self):
        return self.data.shape[0]

    @property
    def n_windows(self):
        return self.windows.shape[0]

    @property
    def poles_per_window(self):
        return (self.windows[:, 1] - self.windows[:, 0] + 1).mean()

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing is not None:
            cv.check_type('spacing', spacing, Real)
            cv.check_greater_than('spacing', spacing, 0.0, equality=False)
        self._spacing = spacing

    @property
    def sqrtAWR(self):
        return self._sqrtAWR

    @sqrtAWR.setter
    def sqrtAWR(self, sqrtAWR):
        if sqrtAWR is not None:
            cv.check_type('sqrtAWR', sqrtAWR, Real)
            cv.check_greater_than('sqrtAWR', sqrtAWR, 0.0, equality=False)
        self._sqrtAWR = sqrtAWR

    @property
    def E_min(self):
        return self._E_min

    @E_min.setter
    def E_min(self, E_min):
        if E_min is not None:
            cv.check_type('E_min', E_min, Real)
            cv.check_greater_than('E_min', E_min, 0.0, equality=True)
        self._E_min = E_min

    @property
    def E_max(self):
        return self._E_max

    @E_max.setter
    def E_max(self, E_max):
        if E_max is not None:
            cv.check_type('E_max', E_max, Real)
            cv.check_greater_than('E_max', E_max, 0.0, equality=False)
        self._E_max = E_max

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is not None:
            cv.check_type('data', data, np.ndarray)
            if len(data.shape) != 2:
                raise ValueError('Multipole data arrays must be 2D')
            if data.shape[1] not in (3, 4):
                raise ValueError(
                     'data.shape[1] must be 3 or 4. One value for the pole.'
                     ' One each for the scattering and absorption residues. '
                     'Possibly one more for a fission residue.')
            if not np.issubdtype(data.dtype, np.complexfloating):
                raise TypeError('Multipole data arrays must be complex dtype')
        self._data = data

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, windows):
        if windows is not None:
            cv.check_type('windows', windows, np.ndarray)
            if len(windows.shape) != 2:
                raise ValueError('Multipole windows arrays must be 2D')
            if not np.issubdtype(windows.dtype, np.integer):
                raise TypeError('Multipole windows arrays must be integer'
                                ' dtype')
        self._windows = windows

    @property
    def broaden_poly(self):
        return self._broaden_poly

    @broaden_poly.setter
    def broaden_poly(self, broaden_poly):
        if broaden_poly is not None:
            cv.check_type('broaden_poly', broaden_poly, np.ndarray)
            if len(broaden_poly.shape) != 1:
                raise ValueError('Multipole broaden_poly arrays must be 1D')
            if not np.issubdtype(broaden_poly.dtype, np.bool_):
                raise TypeError('Multipole broaden_poly arrays must be boolean'
                                ' dtype')
        self._broaden_poly = broaden_poly

    @property
    def curvefit(self):
        return self._curvefit

    @curvefit.setter
    def curvefit(self, curvefit):
        if curvefit is not None:
            cv.check_type('curvefit', curvefit, np.ndarray)
            if len(curvefit.shape) != 3:
                raise ValueError('Multipole curvefit arrays must be 3D')
            if curvefit.shape[2] not in (2, 3):  # sig_s, sig_a (maybe sig_f)
                raise ValueError('The third dimension of multipole curvefit'
                                 ' arrays must have a length of 2 or 3')
            if not np.issubdtype(curvefit.dtype, np.floating):
                raise TypeError('Multipole curvefit arrays must be float dtype')
        self._curvefit = curvefit

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Construct a WindowedMultipole object from an HDF5 group or file.

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing multipole data. If given as a string, it is
            assumed to be the filename for the HDF5 file, and the first group is
            used to read from.

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if isinstance(group_or_filename, h5py.Group):
            group = group_or_filename
            need_to_close = False
        else:
            h5file = h5py.File(str(group_or_filename), 'r')
            need_to_close = True

            # Make sure version matches
            if 'version' in h5file.attrs:
                major, minor = h5file.attrs['version']
                if major != WMP_VERSION_MAJOR:
                    raise DataError(
                        'WMP data format uses version {}. {} whereas your '
                        'installation of the OpenMC Python API expects version '
                        '{}.x.'.format(major, minor, WMP_VERSION_MAJOR))
            else:
                raise DataError(
                    'WMP data does not indicate a version. Your installation of '
                    'the OpenMC Python API expects version {}.x data.'
                    .format(WMP_VERSION_MAJOR))

            group = list(h5file.values())[0]

        name = group.name[1:]
        out = cls(name)

        # Read scalars.

        out.spacing = group['spacing'][()]
        out.sqrtAWR = group['sqrtAWR'][()]
        out.E_min = group['E_min'][()]
        out.E_max = group['E_max'][()]

        # Read arrays.

        err = "WMP '{}' array shape is not consistent with the '{}' array shape"

        out.data = group['data'][()]

        out.windows = group['windows'][()]

        out.broaden_poly = group['broaden_poly'][...].astype(bool)
        if out.broaden_poly.shape[0] != out.windows.shape[0]:
            raise ValueError(err.format('broaden_poly', 'windows'))

        out.curvefit = group['curvefit'][()]
        if out.curvefit.shape[0] != out.windows.shape[0]:
            raise ValueError(err.format('curvefit', 'windows'))

        # _broaden_wmp_polynomials assumes the curve fit has at least 3 terms.
        if out.fit_order < 2:
            raise ValueError("Windowed multipole is only supported for "
                             "curvefits with 3 or more terms.")

        # If HDF5 file was opened here, make sure it gets closed
        if need_to_close:
            h5file.close()

        return out

    @classmethod
    def from_endf(cls, endf_file, log=False, vf_options=None, wmp_options=None):
        """Generate windowed multipole neutron data from an ENDF evaluation.

        .. versionadded:: 0.12.1

        Parameters
        ----------
        endf_file : str
            Path to ENDF evaluation
        log : bool or int, optional
            Whether to print running logs (use int for verbosity control)
        vf_options : dict, optional
            Dictionary of keyword arguments, e.g. {'njoy_error': 0.001},
            passed to :func:`openmc.data.multipole.vectfit_nuclide`
        wmp_options : dict, optional
            Dictionary of keyword arguments, e.g. {'search': True, 'rtol': 0.01},
            passed to :func:`openmc.data.WindowedMultipole.from_multipole`

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if vf_options is None:
            vf_options = {}

        if wmp_options is None:
            wmp_options = {}

        if log:
            vf_options.update(log=log)
            wmp_options.update(log=log)

        # generate multipole data from EDNF
        mp_data = vectfit_nuclide(endf_file, **vf_options)

        # windowing
        return cls.from_multipole(mp_data, **wmp_options)

    @classmethod
    def from_multipole(
        cls,
        mp_data,
        search=None,
        log=False,
        search_n_win=20,
        search_cf_orders=None,
        **kwargs,
    ):
        """Generate windowed multipole neutron data from multipole data.

        Parameters
        ----------
        mp_data : dictionary or str
            Dictionary or Path to the multipole data stored in a pickle file
        search : bool, optional
            Whether to search for optimal window size and curvefit order.
            Defaults to True if no windowing parameters are specified.
        log : bool or int, optional
            Whether to print running logs (use int for verbosity control)
        search_n_win : int, optional
            Number of window sizes to consider in the search grid when
            ``search`` is True.
        search_cf_orders : iterable of int, optional
            Curve-fit orders to consider in the search grid when ``search`` is
            True. Defaults to integers from 10 down to 2.
        **kwargs
            Keyword arguments passed to :func:`openmc.data.multipole._windowing`.

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if isinstance(mp_data, str):
            # load multipole data from file
            with open(mp_data, 'rb') as f:
                mp_data = pickle.load(f)

        if search is None:
            if 'n_cf' in kwargs and ('n_win' in kwargs or 'spacing' in kwargs):
                search = False
            else:
                search = True

        # windowing with specific options
        if not search:
            # set default value for curvefit order if not specified
            if 'n_cf' not in kwargs:
                kwargs.update(n_cf=5)
            return _windowing(mp_data, log=log, **kwargs)

        # search optimal WMP from a range of window sizes and CF orders
        if log:
            print("Start searching ...")
        if search_cf_orders is None:
            search_cf_orders = range(10, 1, -1)

        n_poles = sum([p.size for p in mp_data["poles"]])
        n_win_min = max(5, n_poles // 20)
        n_win_max = 2000 if n_poles < 2000 else 8000
        best_wmp = best_metric = None
        for n_w in np.unique(
            np.linspace(n_win_min, n_win_max, search_n_win, dtype=int)
        ):
            for n_cf in search_cf_orders:
                if log:
                    print(f"Testing N_win={n_w} N_cf={n_cf}")

                # update arguments dictionary
                kwargs.update(n_win=n_w, n_cf=n_cf)

                # windowing
                try:
                    wmp = _windowing(mp_data, log=log, **kwargs)
                except Exception as e:
                    if log:
                        print('Failed: ' + str(e))
                    break

                # select wmp library with metric:
                # - performance: average # used poles per window and CF order
                # - memory: # windows
                metric = -(wmp.poles_per_window * 10. + wmp.fit_order * 1. +
                           wmp.n_windows * 0.01)
                if best_wmp is None or metric > best_metric:
                    if log:
                        print("Best library so far.")
                    best_wmp = deepcopy(wmp)
                    best_metric = metric

        # return the best wmp library
        if log:
            print("Final library: {} poles, {} windows, {:.2g} poles per window, "
                  "{} CF order".format(best_wmp.n_poles, best_wmp.n_windows,
                   best_wmp.poles_per_window, best_wmp.fit_order))

        return best_wmp

    def _evaluate(self, E, T):
        """Compute scattering, absorption, and fission cross sections.

        Parameters
        ----------
        E : Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.

        Returns
        -------
        3-tuple of Real
            Scattering, absorption, and fission microscopic cross sections
            at the given energy and temperature.

        """

        if E < self.E_min: return (0, 0, 0)
        if E > self.E_max: return (0, 0, 0)

        # ======================================================================
        # Bookkeeping

        # Define some frequently used variables.
        sqrtkT = sqrt(K_BOLTZMANN * T)
        sqrtE = sqrt(E)
        invE = 1.0 / E

        # Locate us.  The i_window calc omits a + 1 present from the legacy
        # Fortran version of OpenMC because of the 1-based vs. 0-based
        # indexing.  Similarly startw needs to be decreased by 1.  endw does
        # not need to be decreased because range(startw, endw) does not include
        # endw.
        i_window = min(self.n_windows - 1,
                       int(np.floor((sqrtE - sqrt(self.E_min)) / self.spacing)))
        startw = self.windows[i_window, 0] - 1
        endw = self.windows[i_window, 1]

        # Initialize the ouptut cross sections.
        sig_s = 0.0
        sig_a = 0.0
        sig_f = 0.0

        # ======================================================================
        # Add the contribution from the curvefit polynomial.

        if sqrtkT != 0 and self.broaden_poly[i_window]:
            # Broaden the curvefit.
            dopp = self.sqrtAWR / sqrtkT
            broadened_polynomials = _broaden_wmp_polynomials(E, dopp,
                                                             self.fit_order + 1)
            for i_poly in range(self.fit_order + 1):
                sig_s += (self.curvefit[i_window, i_poly, _FIT_S]
                          * broadened_polynomials[i_poly])
                sig_a += (self.curvefit[i_window, i_poly, _FIT_A]
                          * broadened_polynomials[i_poly])
                if self.fissionable:
                    sig_f += (self.curvefit[i_window, i_poly, _FIT_F]
                              * broadened_polynomials[i_poly])
        else:
            temp = invE
            for i_poly in range(self.fit_order + 1):
                sig_s += self.curvefit[i_window, i_poly, _FIT_S] * temp
                sig_a += self.curvefit[i_window, i_poly, _FIT_A] * temp
                if self.fissionable:
                    sig_f += self.curvefit[i_window, i_poly, _FIT_F] * temp
                temp *= sqrtE

        # ======================================================================
        # Add the contribution from the poles in this window.

        if sqrtkT == 0.0:
            # If at 0K, use asymptotic form.
            for i_pole in range(startw, endw):
                psi_chi = -1j / (self.data[i_pole, _MP_EA] - sqrtE)
                c_temp = psi_chi / E
                sig_s += (self.data[i_pole, _MP_RS] * c_temp).real
                sig_a += (self.data[i_pole, _MP_RA] * c_temp).real
                if self.fissionable:
                    sig_f += (self.data[i_pole, _MP_RF] * c_temp).real

        else:
            # At temperature, use Faddeeva function-based form.
            dopp = self.sqrtAWR / sqrtkT
            for i_pole in range(startw, endw):
                Z = (sqrtE - self.data[i_pole, _MP_EA]) * dopp
                w_val = _faddeeva(Z) * dopp * invE * sqrt(pi)
                sig_s += (self.data[i_pole, _MP_RS] * w_val).real
                sig_a += (self.data[i_pole, _MP_RA] * w_val).real
                if self.fissionable:
                    sig_f += (self.data[i_pole, _MP_RF] * w_val).real

        return sig_s, sig_a, sig_f

    def __call__(self, E, T):
        """Compute scattering, absorption, and fission cross sections.

        Parameters
        ----------
        E : Real or Iterable of Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.

        Returns
        -------
        3-tuple of Real or 3-tuple of numpy.ndarray
            Scattering, absorption, and fission microscopic cross sections
            at the given energy and temperature.

        """

        fun = np.vectorize(lambda x: self._evaluate(x, T))
        return fun(E)

    def export_to_hdf5(self, path, mode='a', libver='earliest'):
        """Export windowed multipole data to an HDF5 file.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to
        mode : {'r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file. This is the second argument
            to the :class:`h5py.File` constructor.
        libver : {'earliest', 'latest'}
            Compatibility mode for the HDF5 file. 'latest' will produce files
            that are less backwards compatible but have performance benefits.

        """

        # Open file and write version.
        with h5py.File(str(path), mode, libver=libver) as f:
            f.attrs['filetype'] = np.bytes_('data_wmp')
            f.attrs['version'] = np.array(WMP_VERSION)

            g = f.create_group(self.name)

            # Write scalars.
            g.create_dataset('spacing', data=np.array(self.spacing))
            g.create_dataset('sqrtAWR', data=np.array(self.sqrtAWR))
            g.create_dataset('E_min', data=np.array(self.E_min))
            g.create_dataset('E_max', data=np.array(self.E_max))

            # Write arrays.
            g.create_dataset('data', data=self.data)
            g.create_dataset('windows', data=self.windows)
            g.create_dataset('broaden_poly',
                             data=self.broaden_poly.astype(np.int8))
            g.create_dataset('curvefit', data=self.curvefit)
