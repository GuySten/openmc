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
# still hold a window, and so which corners in the ENDF background are worth
# splitting pieces at; a corner left inside a piece cannot be fitted by a sum of
# poles at any order and is held to a looser tolerance instead. Lowering this
# limit shrinks the margin and so can let a nuclide reach a tolerance it cannot
# reach at 3000 K, at the cost of the temperature range the library is valid
# over. For Na-23 the margin at 1 eV is about 1 eV at 3000 K, wider than the
# piece a corner there would leave, so that corner is not split at; at 600 K it
# is about 0.4 eV, which is not.
TEMPERATURE_LIMIT = 3000

# How sharply the ENDF background must turn at a tabulated energy, as a
# fraction of the steeper of the two slopes meeting there, before that energy
# counts as a corner rather than a coarsely sampled curve. A smooth function
# tabulated on a logarithmic mesh turns by well under this at every point,
# while a background that goes flat on one side of an energy turns by all of
# it. Paired with the step threshold, which decides whether enough cross
# section is involved to be worth working around.
_KINK_SLOPE_CHANGE = 0.8

# Logging control
DETAILED_LOGGING = 2

# Distinguishes an argument left at its default from one passed as None, which
# for the checking grid means "do without one" rather than "choose for me"
_UNSET = object()

# Curve fit orders the windowing search tries, highest first
_SEARCH_CF_ORDERS = range(10, 1, -1)


# How much of the cross section one vector fit is asked to describe. Fewer
# resonances per piece is cheaper -- the search cost grows faster than the
# order, and the order grows with the resonances -- and does not cost library
# size, since the windowing stage keeps only the poles its windows use. The
# floor on points per piece keeps a piece able to determine a fit at all.
_RESONANCES_PER_PIECE = 4
_POINTS_PER_PIECE = 200
_MIN_PIECE_POINTS = 11

# How many times a search that fell short may resume with twice the iterations,
# and how few orders each resumed pass may cover. Spending the larger budget on
# every fit would multiply the cost of every nuclide to rescue the few pieces
# that fall short, and letting a resumed pass run the whole remaining order
# range would do the same to those pieces.
_EXTRA_VF_PASSES = 2
_EXTRA_VF_ORDERS = 5


class _ToleranceNotMet(RuntimeError):
    """The best fit found did not reach the tolerance, and is attached.

    Carrying it means a caller that decides to accept the fit anyway, or to
    try again with some energies exempted, does not have to run the whole
    order search a second time to get back a result already computed.

    """

    def __init__(self, message, poles, residues, max_error):
        super().__init__(message)
        self.poles = poles
        self.residues = residues
        self.max_error = max_error


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

    Raises
    ------
    _ToleranceNotMet
        If no order in the search range reaches `rtol`. The best fit found is
        attached, so a caller willing to accept it, or to try again with some
        energies exempted, need not repeat the search to recover it.

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
    if relaxed and np.all(tol > rtol) and log:
        # A piece can lie wholly inside a corner's neighbourhood where the
        # background is tabulated coarsely. Holding all of it to the looser
        # tolerance is the honest outcome; there is nothing here that a sum of
        # poles could have been held to `rtol` over.
        print(f"\tevery point is beside a corner; the whole range is held to "
              f"{relaxed_rtol:.3g}")

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
    spacing = np.median(np.diff(s))

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
    n_peaks = _count_resonances(ce_xs[0] + ce_xs[1], rtol)
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

    def fit_at(order, n_iter, poles=None):
        """Vector fit at one order, returning that order's best iteration.

        Starts from `poles` when given, so a fit can be handed back its own
        poles and carried further, and otherwise from the usual evenly spaced
        guess. The return is the best iteration's
        ``(severity, maxre, poles, residues, check_fit, relerr)``, with the
        poles ``None`` if every iteration was unusable.
        """
        if poles is None:
            poles_r = np.linspace(s[0], s[-1], order//2)
            poles = poles_r + poles_r*0.01j
            poles = np.sort(np.append(poles, np.conj(poles)))

        best = (np.inf, np.inf, None, None, None, None)
        for i_vf in range(n_iter):
            if log >= DETAILED_LOGGING:
                print(f"VF iteration {i_vf + 1}/{n_iter}")

            # call vf
            poles, residues, *_ = vectfit(f, s, poles, weight)

            # a pole on the real axis inside the range is a singularity of the
            # zero temperature cross section, so move any off it. The repair
            # keeps the number of poles, so the next iteration can carry on
            # from the poles that were actually fitted and scored.
            moved = _offaxis_poles(poles, s[0], s[-1], spacing)
            if moved is not poles:
                if log >= DETAILED_LOGGING:
                    print("  # real poles: {}".format(np.count_nonzero(
                        (np.imag(poles) == 0.)
                        & (np.real(poles) >= s[0])
                        & (np.real(poles) <= s[-1]))))
                # the residues were solved for where the poles used to be
                poles, residues, *_ = \
                      vectfit(f, s, moved, weight, skip_pole_update=True)

            # assess against the evaluated data. A fit with negative cross
            # sections or a non-finite result is unusable, whatever its error.
            check_fit = evaluate(check_s, poles, residues) / check_energy
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
                test_xs = evaluate(test_s, poles, residues) / test_energy
                if np.any(test_xs > envelope_hi) or np.any(test_xs < envelope_lo):
                    if log >= DETAILED_LOGGING:
                        print("  Rejected: fit oscillates between grid points")
                    maxre = severity = np.inf

            if log >= DETAILED_LOGGING:
                print(f"  # poles: {poles.size}")
                print(f"  Max relative error: {maxre * 100:.3f}%")

            if severity < best[0]:
                if log >= DETAILED_LOGGING:
                    print("  Best so far!")
                best = (severity, maxre, poles, residues, check_fit,
                        relerr)
            elif log >= DETAILED_LOGGING:
                print("  Discarded!")

            # every point is within tolerance, so there is nothing left to
            # gain from further iterations at this order
            if severity <= 1.0:
                break

        return best

    # perform VF with increasing orders
    best_maxre = best_severity = np.inf
    best_poles = None
    n_iter = n_vf_iter
    start, stop = 0, len(orders)
    for _ in range(_EXTRA_VF_PASSES + 1):
        improved = False
        n_discarded = 0  # for accelation, number of discarded searches
        for i in range(start, stop):
            order = orders[i]
            if log:
                print(f"Order={order}({i}/{len(orders)})")

            maxre_before = best_severity
            found = fit_at(order, n_iter)
            if found[0] < best_severity:
                (best_severity, best_maxre, best_poles, best_residues,
                 best_check_fit, best_relerr) = found
                improved = True

            # the search is done as soon as every point is within tolerance;
            # because orders are tried in increasing order, this is also the
            # smallest number of poles that achieves it
            if best_severity <= 1.0:
                if log:
                    print("Found ideal results. Stop!")
                break

            # acceleration: keep adding poles for as long as the maximum
            # relative error keeps coming down, then give up. Stopping here
            # never returns a poor fit because the tolerance check after the
            # loop raises.
            if (best_severity < maxre_before
                    or (best_poles is None and order <= 4*n_peaks)):
                # Nothing usable has been found at any order yet, so there is
                # no fit for more poles to fail to improve on, and the counter
                # below is asking a question with no answer: it would stop the
                # search ten orders in, before ever reaching an order that
                # works. A piece whose candidates are all rejected out of hand
                # -- for a negative cross section, or for swinging between the
                # fitting points -- is given the orders its resonances call
                # for before the counter is allowed to start. Beyond that the
                # search gives up as it always did, so a piece that cannot be
                # fitted at all still fails promptly.
                n_discarded = 0
            else:
                n_discarded += 1
                if n_discarded >= 10:
                    if log:
                        print("Maximum relative error stopped improving. "
                              "Stop!")
                    break

        if best_severity <= 1.0:
            break

        # The search fell short. An order whose poles have not finished moving
        # scores worse than a smaller order that has settled, so "stopped
        # improving" can mean the added poles were never given the passes to
        # settle rather than that they cannot help. Resume from the order the
        # search gave up on with twice the iterations, spending them only on
        # the pieces that need them, and stop as soon as a resumed pass
        # improves nothing, because that is the real ceiling.
        if not improved and n_iter > n_vf_iter:
            break
        start, n_iter = i, 2*n_iter
        stop = min(len(orders), start + _EXTRA_VF_ORDERS)
        if log:
            print(f"Short of tolerance; redrawing orders {orders[start]} to "
                  f"{orders[stop - 1]} with {n_iter} iterations")

    if best_poles is None:
        raise RuntimeError(
            f"Vector fitting produced no usable fit for energy range "
            f"{energy[0]:.3e} to {energy[-1]:.3e} eV with orders "
            f"{orders[0]} to {orders[-1]}. This usually means every candidate "
            f"gave negative cross sections or a non-finite result.")

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

    if best_severity > 1.0:
        raise _ToleranceNotMet(
            f"Vector fitting could not reach the maximum relative error "
            f"tolerance (rtol={rtol:.3g}) for energy range "
            f"{energy[0]:.3e} to {energy[-1]:.3e} eV with orders "
            f"{orders[0]} to {orders[-1]}. The best fit had a maximum "
            f"relative error of {best_maxre:.3%} using {best_poles.size} "
            f"poles. Either relax 'rtol' or widen the pole search range "
            f"with 'orders'.", mp_poles, mp_residues, best_maxre)

    return (mp_poles, mp_residues)

def _offaxis_poles(poles, s_lo, s_hi, spacing):
    """Move poles off the real axis within the fitted range.

    At zero temperature a pole contributes ``1/(sqrt(E) - p)``, so one sitting
    on the real axis between the first and last point of the grid is a
    singularity of the cross section at a real energy. Vector fitting works
    under no such constraint and places them freely: at the orders these fits
    need, a quarter of the poles land there on every iteration.

    Pairing them up two at a time into a conjugate pair keeps the number of
    poles the fit was asked for, so the order the search settles on is the size
    of the fit it gets. Giving each one its own pair instead would grow the fit
    by a pole every time one appeared, which is why such a repair also cannot
    be carried into the next iteration: within a few passes the fit outgrows
    the data it is fitted to.

    Parameters
    ----------
    poles : numpy.ndarray
        Complex poles, with any conjugate pairs adjacent.
    s_lo, s_hi : float
        First and last point of the fitting grid, in sqrt(eV).
    spacing : float
        Grid spacing in sqrt(eV), the narrowest resonance the data resolves.

    Returns
    -------
    numpy.ndarray
        The poles, as many as came in, none of them real within the range.

    """
    kept = []
    real_in = []
    for p in poles:
        if np.imag(p) == 0. and s_lo <= np.real(p) <= s_hi:
            real_in.append(np.real(p))
        else:
            kept.append(p)
    if not real_in:
        return poles

    # removing only real poles leaves the conjugate pairs among `kept`
    # adjacent, and each pair built below is written out adjacent too
    real_in.sort()
    out = kept
    for lower, upper in zip(real_in[::2], real_in[1::2]):
        centre = 0.5*(lower + upper)
        # a resonance narrower than the grid is one the data cannot show, so
        # there is nothing to be gained by placing the pair closer than that
        half_width = max(0.5*(upper - lower), spacing)
        out += [centre + half_width*1j, centre - half_width*1j]
    if len(real_in) % 2:
        # nothing left to pair the last one with, so reflect it across the
        # nearer end of the range, beyond which a real pole is no longer a
        # singularity of anything the fit is judged on
        leftover = real_in[-1]
        out.append(2*s_lo - leftover if leftover - s_lo <= s_hi - leftover
                   else 2*s_hi - leftover)
    return np.array(out, dtype=complex)


def _count_resonances(xs, rtol):
    """How many resonances a cross section has, for sizing the fit.

    Counting every local maximum counts the noise in the reconstructed data as
    well, which on a smooth stretch is most of them: without a threshold O-16
    is credited with 149 resonances below 500 keV, where it has none and is
    nearly constant, and the fit there starts at 298 poles rather than 2.

    A peak has to rise above its surroundings by `rtol` of the median cross
    section to be counted, since one that does not cannot decide whether a fit
    meets `rtol`. Erring low is the safe direction: this only sets where the
    order search starts, and the search climbs until the tolerance is met.

    """
    if xs.size < 3:
        return 0
    median = np.median(np.abs(xs))
    if not np.isfinite(median) or median <= 0:
        return find_peaks(xs)[0].size
    return find_peaks(xs, prominence=rtol*median)[0].size


def _background_kinks(endf_file, mts, threshold, E_min, E_max):
    """Energies where the ENDF background cross section steps sharply.

    Backgrounds in MF3 are tabulated on a coarse grid and interpolated
    linearly, so the cross section has discontinuous slope at every tabulated
    energy. A sum of poles is analytic and cannot reproduce such a corner at
    any order, so the pronounced ones are worth keeping off the middle of a
    fit. Both ends of an interval whose background steps by more than
    `threshold` are corners, along with the scale each is resolved on.

    What this finds is the corners the background *steps* across, not every
    energy it turns at. The distinction matters at a local maximum or minimum,
    where the background turns as sharply as it ever does while its value is
    stationary, so no step test can reach it. Those are left alone
    deliberately. A turn says only that the tabulation is coarse, which is
    true of any smooth curve sampled on a logarithmic mesh: measured across
    the evaluations this has been run on, treating every sign change of the
    slope as a corner would nominate over a thousand energies in the elastic
    background of Fe-56 and several hundred in Na-23, both of which fit to
    tolerance as they are. The pieces holding the sharpest turns in O-16 fit
    to within a twentieth of the tolerance. A step, by contrast, is rare and
    is a discontinuity in the value itself, which no fit can follow.

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
        One row per corner, sorted by energy and excluding the ends of the
        range: the energy of the corner, and half the tabulation spacing
        beside it, which is the scale its influence dies away over.

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
        if e.size < 2:
            continue
        # how finely the background is tabulated either side of each energy;
        # that is the scale a corner there is resolved on, and so the scale
        # its influence dies away over. It is not the width of the stepping
        # interval, which may be broad: a background that steps across a wide
        # interval is a slope, and only its ends are corners.
        width = np.diff(e)
        # A repeated energy is a jump in the background rather than a change
        # of slope. Its zero width says nothing about the scale the jump is
        # resolved on, so borrow the nearest real spacing beside it.
        real = np.flatnonzero(width > 0)
        if real.size == 0:
            continue
        degenerate = np.flatnonzero(width <= 0)
        if degenerate.size:
            j = np.searchsorted(real, degenerate)
            width = width.astype(float)
            width[degenerate] = np.minimum(
                width[real[np.minimum(j, real.size - 1)]],
                width[real[np.maximum(j - 1, 0)]])
        near = np.minimum(np.concatenate([width[:1], width]),
                          np.concatenate([width, width[-1:]]))

        # How far the background steps across each interval, and how sharply
        # it turns at each tabulated energy. Both are needed. A step alone
        # only says the grid is coarse, which is true of a smooth curve
        # sampled on a logarithmic mesh; a turn alone catches every energy
        # where a background happens to be flat on one side, however little
        # cross section is involved. A corner worth working around does both.
        scale = np.maximum(np.abs(xs[:-1]), np.abs(xs[1:]))
        with np.errstate(invalid='ignore', divide='ignore'):
            step = np.abs(np.diff(xs))/np.where(scale > 0, scale, np.inf)
            slope = np.diff(xs)/np.where(np.diff(e) > 0, np.diff(e), np.nan)
            bend = np.zeros(e.size)
            bend[1:-1] = np.abs(np.diff(slope))/np.maximum(np.abs(slope[:-1]),
                                                           np.abs(slope[1:]))
        bend = np.nan_to_num(bend, nan=0.0, posinf=np.inf)
        # a repeated energy is a jump, which turns as sharply as anything can
        for i in degenerate:
            bend[i] = bend[i+1] = np.inf

        for i in np.flatnonzero(step > threshold):
            for j in (i, i + 1):
                if bend[j] > _KINK_SLOPE_CHANGE:
                    kinks.append((e[j], near[j]/2))

    kinks = sorted(k for k in kinks if E_min < k[0] < E_max)
    merged = []
    for e_k, half in kinks:
        if merged and e_k == merged[-1][0]:
            merged[-1][1] = min(merged[-1][1], half)
        else:
            merged.append([e_k, half])
    return np.array(merged, dtype=float).reshape(-1, 2)


def _doppler_margin(e, above, alpha, E_min, E_max):
    """How far past `e` a window sitting against it is evaluated.

    Fitting and windowing both leave room for broadening up to
    `TEMPERATURE_LIMIT` by extending their energy range four Doppler widths
    beyond the energy of interest.

    """
    if above:
        return min(E_max, (sqrt(alpha*e) + 4.0)**2/alpha) - e
    return e - max(E_min, (sqrt(alpha*e) - 4.0)**2/alpha)


def _corner_band(corner):
    """The energies a corner spoils the fit over.

    A sum of poles is smooth, so it cannot follow the discontinuous slope
    where two linearly interpolated background intervals meet, but the damage
    is local: measured against point-wise data, the error from a corner dies
    away within about the spacing of the tabulation around it.

    """
    e_k, half = corner
    return e_k - half, e_k + half


def _corners_to_split(kinks, bounds, alpha, E_min, E_max, unsplit, corner_rtol,
                      energy=None, min_points=11, log=False):
    """Decide which background corners are worth making piece boundaries.

    A window is evaluated over its own width plus a margin of four Doppler
    widths on each side and takes its poles from a single piece. Splitting at
    a corner that would leave a piece narrower than that margin buys nothing:
    the piece is then mostly the extension it shares with its neighbours, and
    the windowing stage cannot place a window inside it without using an
    impractical number of them. Splitting is given up for the same reason
    where it would leave a piece holding too few points of the fitting grid to
    determine a fit at all. Those corners are left where they are, and are
    held to `corner_rtol` only if a piece turns out to need it. Dropping one
    corner widens its neighbours, so the choice is iterated until it settles.

    Parameters
    ----------
    kinks : numpy.ndarray
        Corners from :func:`_background_kinks`, as energy and half-width rows.
    bounds : numpy.ndarray
        The equally spaced piece boundaries the corners are added to.
    alpha : float
        Atomic weight ratio divided by :math:`k_B T` at `TEMPERATURE_LIMIT`.
    E_min, E_max : float
        Energy range of interest
    unsplit : list
        Appended with the energy range spoiled by each corner not split out.
    corner_rtol : float
        Tolerance those ranges may be held to; used only in the log.
    energy : numpy.ndarray, optional
        The fitting grid. When given, a split that would leave either side
        with fewer than `min_points` of it is given up as well.
    min_points : int, optional
        Fewest fitting points a piece may hold. Defaults to 11, enough to
        determine the ten poles of a mid-range fit.

    Returns
    -------
    list of float
        The energies to split pieces at.

    """
    def margin(e, above):
        return _doppler_margin(e, above, alpha, E_min, E_max)

    corners = [tuple(c) for c in np.asarray(kinks, dtype=float).reshape(-1, 2)]
    while corners:
        ends = {e for e, _ in corners}
        edges = np.unique(np.concatenate([bounds, np.fromiter(ends, float)]))
        worst = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            need = 2*((margin(lo, False) if lo in ends else 0.0)
                      + (margin(hi, True) if hi in ends else 0.0))
            short = (hi - lo) - need
            if energy is not None and (lo in ends or hi in ends):
                held = np.searchsorted(energy, hi) - np.searchsorted(energy, lo)
                if held < min_points:
                    # measured the same way, as a shortfall to be made up
                    short = min(short, (hi - lo)*(held/min_points - 1.0))
            if short < 0 and (worst is None or short < worst[0]):
                worst = (short, lo, hi, need)
        if worst is None:
            break

        # give up whichever corner bounds the worst piece; when both ends are
        # corners, the one demanding the wider margin
        _, lo, hi, need = worst
        i = max((j for j, c in enumerate(corners) if c[0] in (lo, hi)),
                key=lambda j: margin(corners[j][0], corners[j][0] == lo))
        dropped = corners.pop(i)
        unsplit.append(_corner_band(dropped))
        if log:
            print(f"  Not splitting at the corner at {dropped[0]:.6g} eV: it "
                  f"would leave the piece from {lo:.6g} to {hi:.6g} eV too "
                  f"small, against the {need:.6g} eV of Doppler margin its "
                  f"windows need and the {min_points} fitting points a fit "
                  f"needs")

    unsplit.sort()
    return [e for e, _ in corners]


def vectfit_nuclide(endf_file, njoy_error=5e-4, njoy_error_check=_UNSET,
                    vf_pieces=None, kink_threshold=0.25, rtol=1e-3,
                    corner_rtol=5e-2, min_n_win=1000, log=False,
                    path_out=None, mp_filename=None, **kwargs):
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
        points against evaluated data instead of an interpolant.

        What this tolerance buys is sampling density rather than accuracy: the
        grid's cross sections are evaluated from the resonance parameters at
        its own energies, whatever tolerance was asked for, and the fit is
        judged at those energies without interpolating between them. It has to
        be fine enough to land where the fit is worst, which is a fixed ratio
        to the tolerance being checked rather than a number of points. A
        hundredth of `rtol` finds the same maximum error on Na-23, at the same
        energy, as a grid ten times finer, using a third of the points, and
        gives B-10, Xe-135 and H-1 the same libraries as grids two to five
        times larger. That is the default, capped at a tenth of `njoy_error`
        so it stays finer than the grid it is checking. Pass a number to
        override, or None to skip the second NJOY run and judge the fit on the
        fitting grid alone.

        A window holding fewer checking points than its curve fit has
        coefficients is judged on those points interpolated, which is sound
        for the same reason the grid is usable at all: its tolerance is a
        guarantee about linear interpolation between its points. So the grid
        does not have to be dense enough to resolve every window separately.
    vf_pieces : integer, optional
        Number of equal-in-momentum spaced energy pieces for data fitting
    kink_threshold : float or None, optional
        Relative step in the ENDF background cross section above which an extra
        piece boundary is inserted. Backgrounds in MF3 are interpolated
        linearly between tabulated energies, so the cross section has a corner
        at each of them, and a sum of poles cannot reproduce a corner at any
        order. Splitting there puts the pronounced ones on piece boundaries
        rather than in the middle of a fit. It is a step the background takes
        that counts, not a turn it makes: a background turns at every local
        maximum without its value moving, and treating those as corners would
        split some evaluations at hundreds of energies that fit perfectly well
        as they are. Set to None to disable. Defaults
        to 0.25.

        Splitting is given up at a corner that would leave a piece narrower
        than the Doppler margin its windows need, which `TEMPERATURE_LIMIT`
        governs; see that constant. Such a corner is left where it is and its
        neighbourhood held to `corner_rtol` instead. A piece is extended past
        its boundaries to leave room for broadening, so the corners it is
        split at are still inside the range it is fitted over; the split
        decides where the boundary falls, not what the fit has to reproduce.
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
    min_n_win : int, optional
        Fewest windows the library may later be built with. A window takes its
        poles from a single piece and spans its own width plus four Doppler
        widths on each side, while adjacent pieces overlap only by that same
        margin, so a window straddling a boundary would fall outside both
        neighbours and be described by poles fitted somewhere else. Pieces are
        therefore widened by half a window as well, which costs about 2% more
        energy range per piece and supports any window count at or above this
        one. Coarser windowings remain unavailable. Defaults to 1000.
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
    dict
        Multipole data for the nuclide. Alongside the ``poles`` and
        ``residues`` of each piece it carries what the windowing stage needs
        to judge itself honestly: ``mts``, the reactions fitted, in the order
        their cross sections are stored; ``bounds``, the piece boundaries,
        which are
        not equally spaced once corners have been split at; ``fit_ranges``,
        the wider range each piece was actually fitted over, so a window can
        be given poles that describe it rather than poles extrapolated from
        elsewhere; ``check_energy`` and ``check_xs``, the finer NJOY
        reconstruction, so windows are judged against evaluated data rather
        than against the fit they were built from; ``relaxed``, the energy
        ranges holding a corner that could not be split out; and
        ``max_error``, the relative error each piece achieved over its full
        range, those ranges included.

    """

    # ======================================================================
    # PREPARE POINT-WISE XS

    # make 0K ACE data using njoy
    if log:
        print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

    nuc_ce = IncidentNeutron.from_njoy(endf_file, temperatures=[0.0],
             error=njoy_error, broadr=False, heatr=False, purr=False)

    # fine enough to land where the fit is worst, and never coarser than the
    # grid it is checking
    if njoy_error_check is _UNSET:
        njoy_error_check = min(0.01*rtol, 0.1*njoy_error)

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

    alpha = nuc_ce.atomic_weight_ratio/(K_BOLTZMANN*TEMPERATURE_LIMIT)

    if vf_pieces is None:
        # How much one fit is asked to describe, rather than whether the
        # nuclide is complicated enough to bother dividing at all. The order a
        # piece needs grows with the resonances in it and the cost of the
        # search grows faster than the order, so dividing further is cheaper
        # almost until it stops working: on Na-23, 76 pieces fit three times
        # faster than 38 and leave a smaller library, because the windowing
        # stage keeps fewer poles when each piece describes less. A rule that
        # divided only above a threshold left the nuclides under it with a
        # single piece spanning the whole range -- for O-16 ten decades, which
        # no windowing of it could be made accurate.
        n_peaks = _count_resonances(total_xs, rtol)
        vf_pieces = max(n_peaks//_RESONANCES_PER_PIECE,
                        n_points//_POINTS_PER_PIECE)
        # but a piece still has to hold enough of the fitting grid to
        # determine a fit, and to be wider than the Doppler margin its windows
        # reach past it -- the same requirement `_corners_to_split` puts on a
        # corner. For pieces equal in momentum that works out independent of
        # energy: a piece at E is 2*sqrt(E)*spacing wide and the margin either
        # side of it is 8*sqrt(E/alpha), so the momentum spacing must exceed
        # 16/sqrt(alpha) whatever E is.
        vf_pieces = min(vf_pieces, n_points//_MIN_PIECE_POINTS,
                        int((sqrt(E_max) - sqrt(E_min))*sqrt(alpha)/16))
        vf_pieces = max(1, vf_pieces)
        if log:
            print(f"  {n_peaks} resonances over {n_points} points "
                  f"-> {vf_pieces} pieces")
    piece_width = (sqrt(E_max) - sqrt(E_min)) / vf_pieces

    # Boundaries of the equal-in-momentum pieces, plus the pronounced corners
    # in the background that are worth splitting at.
    bounds = [(sqrt(E_min) + piece_width*i)**2 for i in range(vf_pieces + 1)]
    bounds[0], bounds[-1] = E_min, E_max
    bounds = np.array(bounds)
    kinks = np.array([])
    all_kinks = np.empty((0, 2))
    unsplit = []
    if kink_threshold is not None:
        all_kinks = _background_kinks(endf_file, mts, kink_threshold,
                                      E_min, E_max)
        orders_wanted = kwargs.get('orders')
        corners = _corners_to_split(
            all_kinks, bounds, alpha, E_min, E_max, unsplit, corner_rtol,
            energy=energy,
            min_points=(max(orders_wanted) + 1) if orders_wanted else 11,
            log=log)
        kinks = np.array(corners, dtype=float)
        if log and kinks.size:
            print(f"  Splitting at {kinks.size} background corners: "
                  + ", ".join(f"{k:.4g}" for k in kinks) + " eV")
    bounds = np.unique(np.concatenate([bounds, kinks]))
    n_pieces = bounds.size - 1


    poles, residues, max_error, fit_ranges = [], [], [], []
    # The neighbourhood of every corner, to fall back on for a piece that
    # cannot reach `rtol` across one, and the bands actually used, which the
    # windowing stage has to honour too. Nothing is exempt up front: a corner
    # is usually fitted well enough anyway, and on a nuclide whose background
    # is tabulated coarsely the bands would otherwise cover much of the
    # energy range without a single one of them being needed.
    corner_bands = [_corner_band(row) for row in all_kinks]
    used_relaxed = set()
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
        # and by half a window, so that a window straddling this boundary is
        # still covered by the fit rather than extrapolating from it. Half a
        # window of momentum spacing is sqrt(E)*spacing in energy.
        if min_n_win:
            spacing = (sqrt(E_max) - sqrt(E_min))/min_n_win
            e_start = max(E_min, e_start - sqrt(lo_bound)*spacing)
            e_end = min(E_max, e_end + sqrt(hi_bound)*spacing)
        e_start_idx = max(0, np.searchsorted(energy, e_start, side='right') - 1)
        e_end_idx = np.searchsorted(energy, e_end, side='left') + 1
        e_idx = range(e_start_idx, min(e_end_idx, n_points - 1) + 1)

        if check_energy is None:
            c_energy = c_xs = None
        else:
            lo, hi = energy[e_idx][0], energy[e_idx][-1]
            c_mask = (check_energy >= lo) & (check_energy <= hi)
            c_energy, c_xs = check_energy[c_mask], check_xs[:, c_mask]

        # A corner cannot be fitted by poles at any order. Exempt its
        # neighbourhood from the tolerance rather than loosening the tolerance
        # over the whole piece, then hold that neighbourhood to corner_rtol on
        # its own. Corners that could not be split out are exempt from the
        # start, since no fit can reach `rtol` across one. A corner this piece
        # was split at is still inside the range it is fitted over, because
        # the range is extended to leave room for broadening, but it sits in
        # that extension rather than in the middle of the piece and is usually
        # fitted well enough anyway. Exempting every one of those up front
        # would give away most of the energy range on some nuclides, so they
        # are only exempted for a piece that has actually failed without them.
        lo_p, hi_p = energy[e_idx][0], energy[e_idx][-1]
        fit_args = dict(log=log, rtol=rtol, check_energy=c_energy,
                        check_xs=c_xs, relaxed_rtol=corner_rtol,
                        path_out=path_out, **kwargs)
        try:
            p, r = _vectfit_xs(energy[e_idx], ce_xs[:, e_idx], mts, **fit_args)
            relaxed = []
        except _ToleranceNotMet as strict:
            relaxed = [(a, b) for a, b in corner_bands
                       if a < hi_p and b > lo_p]
            if relaxed and log:
                print(f"  cannot reach rtol={rtol:.3g} across the corners in "
                      f"the range it is fitted over; holding "
                      + ", ".join(f"{a:.6g}-{b:.6g}" for a, b in relaxed)
                      + f" eV to {corner_rtol:.3g}")
            attempt = strict
            if relaxed:
                try:
                    p, r = _vectfit_xs(energy[e_idx], ce_xs[:, e_idx], mts,
                                       relaxed=relaxed, **fit_args)
                    attempt = None
                except _ToleranceNotMet as exempted:
                    attempt = exempted
            if attempt is not None:
                # Poles are not the whole library: every window adds a
                # polynomial of its own, over a range narrow enough that a
                # stretch of background the poles cannot follow across a
                # whole piece is a straight line within one window. Keep the
                # best pole set the search already found -- repeating it would
                # only arrive at the same one -- and let the windowing stage,
                # which is judged against the same evaluated data and is where
                # the library's accuracy actually lives, enforce `rtol`.
                if log:
                    print(f"  poles alone reach only {attempt.max_error:.3%} "
                          f"here, over rtol={rtol:.3g}; leaving it to the "
                          f"windowing stage")
                p, r = attempt.poles, attempt.residues
        used_relaxed.update(relaxed)

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
        fit_ranges.append((lo_p, hi_p))
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
               "mts": list(mts),
               "bounds": bounds,
               "max_error": np.array(max_error),
               "fit_ranges": np.array(fit_ranges),
               "check_energy": check_energy,
               "check_xs": check_xs,
               "relaxed": np.asarray(sorted(used_relaxed),
                                     dtype=float).reshape(-1, 2),
               "corner_bands": np.asarray(sorted(corner_bands),
                                          dtype=float).reshape(-1, 2),
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
    rtol : float or Iterable of float, optional
        Maximum relative error tolerance. Several may be given, tightest
        first: each window is then held to the tightest one it can reach, and
        only the windows that cannot reach it fall back to the next. Holding
        the whole nuclide to the loosest instead would cost accuracy over
        every window that never needed it.
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
    # A window that cannot reach the tightest tolerance falls back to the next
    # rather than failing the whole library, so a single nuclide keeps its
    # accuracy everywhere it can be had. Sorted tightest first, so the first
    # one a window meets is the best it could have done.
    rtol_ladder = sorted(np.atleast_1d(np.asarray(rtol, dtype=float)).ravel())
    if not rtol_ladder:
        raise ValueError("rtol must give at least one tolerance")

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
    # Corners the fitting stage got through without help. A window is a
    # weaker approximant than a whole piece -- its curve fit is a low-order
    # polynomial over a narrow range -- so one may still need a corner
    # exempted where the piece did not, and it is given the same fallback.
    corner_bands = np.asarray(mp_data.get("corner_bands", np.empty((0, 2))),
                              dtype=float)
    fit_ranges = mp_data.get("fit_ranges")
    if fit_ranges is not None:
        fit_ranges = np.asarray(fit_ranges, dtype=float)
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
    # which tolerance each window ended up being held to, so a library
    # that had to give ground somewhere says where
    window_rtol = []
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

        # Locate the piece to take poles from. Pieces are fitted over a range
        # wider than their boundaries, by the same Doppler margin a window
        # spans plus half a window, so their ranges overlap and a window
        # straddling a boundary is still covered by the fit on either side, at
        # any window count at or above the `min_n_win` used when fitting.
        # Prefer a piece whose fit actually covers the window, since outside
        # that range its poles are an extrapolation and describe nothing.
        i_piece = None
        if fit_ranges is not None:
            covering = [ip for ip in range(n_pieces)
                        if fit_ranges[ip, 0] <= e_start and
                        e_end <= fit_ranges[ip, 1]]
            if covering:
                i_piece = min(covering, key=lambda ip: abs(
                    incenter - sqrt(0.5*(fit_ranges[ip, 0] + fit_ranges[ip, 1]))))
        if i_piece is None:
            i_piece = int(np.clip(np.searchsorted(bounds_sqrt, incenter) - 1,
                                  0, n_pieces - 1))
        poles, residues = mp_poles[i_piece], mp_residues[i_piece]
        n_poles = poles.size

        # energy points for fitting, and the reference to fit against
        in_check = None
        if check_energy is not None:
            in_check = np.flatnonzero((check_energy >= e_start) &
                                      (check_energy <= e_end))
            if in_check.size > 10000:
                in_check = in_check[::int(np.ceil(in_check.size/10000))]
            if in_check.size < 2:
                in_check = None
        if in_check is not None and in_check.size >= n_cf + 2:
            # However few, the evaluated points are used as they are. Padding
            # them out by interpolation would only add corners of the
            # interpolant's own making for the curve fit to be judged
            # against, and enough of them to determine the fit is enough to
            # test it: whether the window is sampled densely is a property of
            # `njoy_error_check`, not something to paper over here.
            energy = check_energy[in_check]
            energy_sqrt = np.sqrt(energy)
            xs_ref = check_xs[:, in_check]
        elif check_energy is not None:
            # Too few checking points fall inside this window to determine the
            # curve fit, which happens where the cross section is smooth
            # enough that NJOY needed hardly any points to describe it.
            # Interpolating the checking grid is sound for that same reason:
            # its tolerance is a guarantee about linear interpolation between
            # its points, and it is reconstructed far finer than the tolerance
            # being checked. Evaluating the poles instead would compare the
            # curve fit with what it is correcting and could never fail.
            n_points = max(100, n_cf + 2)
            energy_sqrt = np.linspace(np.sqrt(e_start), np.sqrt(e_end),
                                      n_points)
            energy = energy_sqrt**2
            xs_ref = np.vstack([np.interp(energy, check_energy, check_xs[i])
                                for i in range(check_xs.shape[0])])
        else:
            # No checking grid at all, so the multipole form is all there is.
            # This only judges the curve fit against the poles it corrects and
            # cannot fail on their account.
            n_points = min(max(100, int((e_end - e_start)*4)), 10000)
            energy_sqrt = np.linspace(np.sqrt(e_start), np.sqrt(e_end), n_points)
            energy = energy_sqrt**2
            # note the residue terms in the multipole and vector fitting
            # representations differ by a 1j
            xs_ref = evaluate(energy_sqrt, poles, residues*1j) / energy

        # curve fit matrix
        matrix = np.vstack([energy**(0.5*i - 1) for i in range(n_cf + 1)]).T

        # the pole nearest the middle of the window, which the search grows out
        # from so that a window uses the fewest poles it can
        center_pole_ind = np.argmin((np.fabs(poles.real - incenter)))

        # Work down the ladder of tolerances. A window that cannot be held to
        # the tightest one is held to the next rather than failing the whole
        # library, and because the search starts over at each tolerance the
        # window is still built from the fewest poles that reach it.
        for level in rtol_ladder:
            # a corner in the background is looser, as in the fitting stage
            tol = np.full(energy.size, float(level))
            for lo_x, hi_x in relaxed:
                tol[(energy >= lo_x) & (energy <= hi_x)] = max(corner_rtol,
                                                               level)
            # and the same again for corners the fitting stage did not need to
            # exempt, to fall back on only once every pole has been tried
            fallback_tol = tol.copy()
            for lo_x, hi_x in corner_bands:
                fallback_tol[(energy >= lo_x)
                             & (energy <= hi_x)] = max(corner_rtol, level)
            if np.array_equal(fallback_tol, tol):
                fallback_tol = None

            # start from 0 poles, at the center nearest pole
            lp = rp = center_pole_ind
            met = False
            while True:
                if log >= DETAILED_LOGGING:
                    print(f"Trying poles {lp} to {rp}")

                # calculate the cross sections contributed by the windowed
                # poles
                if rp > lp:
                    xs_wp = evaluate(energy_sqrt, poles[lp:rp],
                                     residues[:, lp:rp]*1j) / energy
                else:
                    xs_wp = np.zeros_like(xs_ref)

                # Do least square curve fit on the remains, weighted by the
                # inverse cross section so that it minimizes relative rather
                # than absolute deviation, as the tolerance below is relative.
                # A window spanning a resonance and its wing covers orders of
                # magnitude, and an unweighted fit trades error at the bottom
                # of that range for error at the top. Points below atol are
                # not scored, so they are not allowed to dominate the
                # weighting either.
                coefs = np.empty((n_cf + 1, xs_ref.shape[0]))
                for i_mt in range(xs_ref.shape[0]):
                    w = 1.0/np.maximum(np.abs(xs_ref[i_mt]), atol)
                    coefs[:, i_mt] = np.linalg.lstsq(
                        matrix*w[:, np.newaxis],
                        (xs_ref[i_mt] - xs_wp[i_mt])*w, rcond=None)[0]
                xs_fit = (matrix @ coefs).T

                # assess the result
                abserr = np.abs(xs_fit + xs_wp - xs_ref)
                with np.errstate(invalid='ignore', divide='ignore'):
                    relerr = abserr / xs_ref
                if not np.any(np.isnan(abserr)):
                    scored = np.where(abserr > atol, relerr, 0.0)
                    # every point in the window must be within its own
                    # tolerance
                    if np.max(scored/tol) <= 1.0:
                        # meet tolerances
                        if log >= DETAILED_LOGGING:
                            print("Accuracy satisfied.")
                        met = True
                        break
                    # With every pole of the piece already in, a polynomial is
                    # all that is left to add, and it cannot follow a corner
                    # either. Exempt the corners as the fitting stage does
                    # rather than abandon a windowing that is otherwise within
                    # tolerance.
                    if (fallback_tol is not None and lp <= 0 and rp >= n_poles
                            and np.max(scored/fallback_tol) <= 1.0):
                        if log >= DETAILED_LOGGING:
                            print("Accuracy satisfied outside the corners.")
                        met = True
                        break

                # try to include one more pole (next center nearest). The
                # first window is no different: a window near E_min has its
                # curve fit left unbroadened, since broadening it would reach
                # below the bottom of the library, but its poles are still
                # broadened through the Faddeeva form like any other window's,
                # and nothing in the format or the evaluation treats window
                # zero specially. Requiring it to be a pure curve fit only
                # made generation fail for nuclides whose lowest window a
                # polynomial cannot describe on its own.
                if lp <= 0 and rp >= n_poles:
                    # every pole of the piece is in and the window still
                    # misses, so this tolerance is out of reach here
                    break
                if rp >= n_poles:
                    lp -= 1
                elif lp <= 0 or poles[rp] - incenter <= incenter - poles[lp-1]:
                    rp += 1
                else:
                    lp -= 1

            if met:
                window_rtol.append(level)
                if level > rtol_ladder[0] and log >= DETAILED_LOGGING:
                    print(f"Held to {level:.3g} rather than "
                          f"{rtol_ladder[0]:.3g}.")
                break
        else:
            # not even the loosest tolerance on the ladder is reachable
            raise RuntimeError(
                f'Window {iw + 1} of {n_win} covering {e_start:.4g} to '
                f'{e_end:.4g} eV cannot reach the tolerance with all '
                f'{n_poles} poles of its piece and a curve fit of order '
                f'{n_cf}.')

        # save data for this window
        win_data.append((i_piece, lp, rp, coefs))

        # mark the windowed poles as used poles
        poles_unused[i_piece][lp:rp] = 0

    if log and len(rtol_ladder) > 1:
        for level in rtol_ladder:
            n = window_rtol.count(level)
            if n:
                print(f"  {n} of {len(window_rtol)} windows held to "
                      f"{level:.3g}")

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
            search_cf_orders = _SEARCH_CF_ORDERS

        n_poles = sum([p.size for p in mp_data["poles"]])
        n_win_min = max(5, n_poles // 20)
        n_win_max = 2000 if n_poles < 2000 else 8000
        best_wmp = best_metric = None
        failures = []
        # The ranking below charges a hundredth per window and one per curve
        # fit order, and cannot charge less than nothing for the poles, so
        # those two terms alone are the least a configuration could possibly
        # cost. Once something has been found, anything whose floor already
        # exceeds it cannot win however well it windows, and windowing it
        # would be the greater part of the work for a foregone answer.
        cheapest_cf = min(search_cf_orders)
        for n_w in np.unique(
            np.linspace(n_win_min, n_win_max, search_n_win, dtype=int)
        ):
            if (best_metric is not None
                    and -(0.01*n_w + cheapest_cf) <= best_metric):
                # window counts are tried in increasing order and the floor
                # rises with them, so nothing further along can win either
                if log:
                    print(f"Stopping at N_win={n_w}: no window count this "
                          f"large can beat what has been found.")
                break
            for n_cf in search_cf_orders:
                if (best_metric is not None
                        and -(0.01*n_w + n_cf) <= best_metric):
                    continue
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
                    # Try the remaining curve fit orders rather than giving up
                    # on this window count. Orders are tried highest first and
                    # the highest is the worst conditioned, the curve fit
                    # matrix spanning 1/E to E**4, so it is the most likely to
                    # fail and the least informative about the rest.
                    failures.append((int(n_w), int(n_cf), str(e)))
                    continue

                # Rank by cost alone: every library returned here already meets
                # the tolerance, since the windowing rejects any that does not,
                # so the candidates differ only in how expensive they are to
                # evaluate.
                # - performance: average # used poles per window and CF order
                # - memory: # windows
                metric = -(wmp.poles_per_window * 10. + wmp.fit_order * 1. +
                           wmp.n_windows * 0.01)
                if best_wmp is None or metric > best_metric:
                    if log:
                        print("Best library so far.")
                    best_wmp = deepcopy(wmp)
                    best_metric = metric

        if best_wmp is None:
            # one line per distinct reason, since a reason that recurs across
            # every curve fit order says the same thing each time
            distinct = {}
            for n, c, m in failures:
                distinct.setdefault(m, []).append((n, c))
            detail = "\n".join(
                f"    [{len(where)} of {len(failures)}] {m}"
                for m, where in sorted(distinct.items(),
                                       key=lambda kv: -len(kv[1]))[:5])
            raise RuntimeError(
                f"No windowing configuration met the tolerance: all "
                f"{len(failures)} combinations of window count and curve fit "
                f"order failed, with {len(distinct)} distinct reasons:\n"
                f"{detail}")

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
