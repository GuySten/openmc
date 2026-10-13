"""Nonequally probable step function (NEPSF) representation of scattering
angle distributions given by truncated Legendre expansions.

The method is that of L. Mao, J. P. Both, and J. C. Nimal, "Transfer Matrix
Treatments in TRIMARAN-II, Nonequally Probable Step Function Representation in
Multigroup Monte Carlo," Nucl. Sci. Eng., 130, 226 (1998). For an expansion of
odd order :math:`N`, the :math:`(N+1)/2` discrete angles that preserve the
first :math:`N` moments give the step weights, and the :math:`N+1` step
boundaries are chosen to preserve the same moments with the widest steps.

"""
from math import comb

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize

# Discrete-angle weights below this value mean that the moments lie on the
# boundary of the moment space, where the discrete angles are the only
# distribution with these moments
_DEGENERATE_WEIGHT = 1e-10

# Relative tolerance on the moments preserved by the steps
_MOMENT_TOLERANCE = 1e-10


def discrete_angles(moments):
    """Discrete angles and weights preserving Legendre moments.

    Parameters
    ----------
    moments : numpy.ndarray
        Legendre moments :math:`a_l = \\int_{-1}^1 P_l(\\mu) f(\\mu) d\\mu`,
        :math:`l = 0, \\ldots, 2n - 1`, with :math:`a_0 = 1`

    Returns
    -------
    angles : numpy.ndarray or None
        The :math:`n` angles in increasing order, or None if no nonnegative
        distribution on [-1, 1] has these moments
    weights : numpy.ndarray or None
        Probabilities of the angles

    """
    # Modified Chebyshev algorithm with monic Legendre polynomials as the
    # reference (Gautschi), which is well conditioned on [-1, 1]
    K = len(moments)
    n = K // 2
    lead = np.array([comb(2*k, k) / 2.0**k for k in range(K)])
    beta_ref = np.array([0.0] + [k*k/(4.0*k*k - 1.0) for k in range(1, K)])
    sigma_prev = np.zeros(K)
    sigma = np.asarray(moments[:K], dtype=float) / lead
    alpha = np.zeros(n)
    beta = np.zeros(n)
    alpha[0] = sigma[1] / sigma[0]
    beta[0] = sigma[0]
    for k in range(1, n):
        new = np.zeros(K)
        for l in range(k, K - k):
            new[l] = (sigma[l + 1] - alpha[k - 1]*sigma[l] -
                      beta[k - 1]*sigma_prev[l] + beta_ref[l]*sigma[l - 1])
        if not new[k] > 0.0:
            return None, None
        alpha[k] = new[k + 1]/new[k] - sigma[k]/sigma[k - 1]
        beta[k] = new[k] / sigma[k - 1]
        sigma_prev, sigma = sigma, new

    # Golub-Welsch: the angles are the eigenvalues of the Jacobi matrix
    off = np.sqrt(beta[1:])
    jacobi = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
    angles, vectors = np.linalg.eigh(jacobi)
    weights = beta[0] * vectors[0]**2
    if np.any(np.abs(angles) > 1.0):
        return None, None
    return angles, weights / weights.sum()


def _step_moments(x, p, N):
    """Moments sum_i p_i E[u^k] over uniform steps [x_2i, x_2i+1], k = 1..N,
    and their derivatives with respect to the boundaries."""
    lo, hi = x[0::2], x[1::2]
    lo_pow = np.vstack([lo**j for j in range(N + 1)])
    hi_pow = np.vstack([hi**j for j in range(N + 1)])
    m = np.zeros(N)
    jac = np.zeros((N, x.size))
    for k in range(1, N + 1):
        # (b^(k+1) - a^(k+1)) / ((k+1)(b-a)) = sum_j a^j b^(k-j) / (k+1)
        m[k - 1] = p @ sum(lo_pow[j]*hi_pow[k - j] for j in range(k + 1))
        jac[k - 1, 0::2] = p * sum(j*lo_pow[j - 1]*hi_pow[k - j]
                                   for j in range(1, k + 1))
        jac[k - 1, 1::2] = p * sum((k - j)*lo_pow[j]*hi_pow[k - j - 1]
                                   for j in range(k))
    scale = np.arange(2, N + 2)[:, None]
    return m / scale[:, 0], jac / scale


def _widest_steps(angles, weights):
    """Boundaries of the widest steps preserving the moments of the discrete
    angles, or None if none were found.

    The moments are imposed in u = (mu - c)/s, where c and s are the mean and
    standard deviation of the discrete angles. Moments of degree <= N in u are
    linear combinations of those in mu, so this is the same condition, but it
    is well conditioned for strongly peaked distributions. Among the steps
    preserving the moments, those minimizing sum_i p_i / w_i^2 (the penalty
    term of Mao et al.) are chosen.
    """
    n = angles.size
    N = 2*n - 1
    p = weights
    c = p @ angles
    s = np.sqrt(p @ (angles - c)**2)
    u = (angles - c) / s
    u_min, u_max = (-1.0 - c)/s, (1.0 - c)/s
    target = np.array([p @ u**k for k in range(1, N + 1)])
    scale = np.array([p @ np.abs(u)**k for k in range(1, N + 1)])

    # Ordering constraints: u_min <= x_0, x_j <= x_j+1, x_2n-1 <= u_max
    A = np.zeros((2*n + 1, 2*n))
    A[0, 0] = 1.0
    for j in range(2*n - 1):
        A[j + 1, j] = -1.0
        A[j + 1, j + 1] = 1.0
    A[-1, -1] = -1.0
    b = np.zeros(2*n + 1)
    b[0] = -u_min
    b[-1] = u_max

    def penalty(y):
        return p @ (y[1::2] - y[0::2])**-2.

    def objective(y):
        return np.log(penalty(y))

    def gradient(y):
        g = np.zeros_like(y)
        dw = 2.0 * p * (y[1::2] - y[0::2])**-3.
        g[0::2] = dw
        g[1::2] = -dw
        return g / penalty(y)

    cache = {}

    def moments(y):
        key = y.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = _step_moments(y, p, N)
        return cache[key]

    constraints = [
        {'type': 'eq',
         'fun': lambda y: (moments(y)[0] - target) / scale,
         'jac': lambda y: moments(y)[1] / scale[:, None]},
        {'type': 'ineq', 'fun': lambda y: A @ y + b, 'jac': lambda y: A}
    ]

    # Start from narrow steps centred on the discrete angles, which preserve
    # the moments to second order in the widths
    edges = np.concatenate(([u_min], u, [u_max]))
    room = np.minimum(np.diff(edges)[:-1], np.diff(edges)[1:])
    for fraction in (0.05, 0.005):
        half = np.maximum(fraction*room, 1e-12)
        y0 = np.ravel(np.column_stack([u - half, u + half]))
        with np.errstate(all='ignore'):
            result = minimize(objective, y0, jac=gradient,
                              constraints=constraints, method='SLSQP',
                              options={'ftol': 1e-14, 'maxiter': 1000})
        y = result.x
        if not np.all(np.isfinite(y)):
            continue
        if np.any(A @ y + b < -1e-12) or np.any(y[1::2] <= y[0::2]):
            continue
        error = np.abs(_step_moments(y, p, N)[0] - target) / scale
        if np.max(error) < _MOMENT_TOLERANCE:
            return np.clip(c + s*y, -1.0, 1.0)
    return None


def nepsf(moments, order):
    """Nonequally probable step function with given Legendre moments.

    Steps are found for the highest odd order not exceeding `order` for which
    the moments are those of a nonnegative distribution. If the moments lie on
    the boundary of the moment space, or no steps are found, the discrete
    angles (steps of zero width) of that order are returned.

    Parameters
    ----------
    moments : Iterable of float
        Legendre moments :math:`a_l`, :math:`l = 0, \\ldots, L`, normalized so
        that :math:`a_0 = 1`
    order : int
        Highest Legendre order to preserve, at most :math:`L`

    Returns
    -------
    lower : numpy.ndarray
        Lower boundaries of the :math:`(N+1)/2` steps, where :math:`N` is the
        largest odd integer not exceeding `order`. Unused steps have zero
        weight.
    upper : numpy.ndarray
        Upper boundaries of the steps
    weights : numpy.ndarray
        Probabilities of the steps
    order_used : int
        Legendre order preserved by the steps

    """
    moments = np.asarray(moments, dtype=float)
    N = order if order % 2 == 1 else order - 1
    n_steps = (N + 1) // 2
    lower = np.zeros(n_steps)
    upper = np.zeros(n_steps)
    weights = np.zeros(n_steps)

    for N_used in range(N, 0, -2):
        n = (N_used + 1) // 2
        angles, p = discrete_angles(moments[:2*n])
        if angles is None:
            # On the boundary of the moment space the distribution is a set
            # of fewer points, which reproduce all of the moments
            for m in range(n - 1, 0, -1):
                angles, p = discrete_angles(moments[:2*m])
                if angles is not None:
                    error = legendre.legvander(angles, N_used).T @ p - \
                        moments[:N_used + 1]
                    if np.max(np.abs(error)) < _MOMENT_TOLERANCE:
                        break
                    angles = None
            if angles is None:
                continue
            lower[:m] = upper[:m] = angles
            weights[:m] = p
            return lower, upper, weights, N_used
        if np.min(p) < _DEGENERATE_WEIGHT:
            x = np.repeat(angles, 2)
        elif n == 1:
            # One step centred on the mean
            half = 1.0 - abs(angles[0])
            x = np.array([angles[0] - half, angles[0] + half])
        else:
            x = _widest_steps(angles, p)
            if x is None:
                x = np.repeat(angles, 2)
        lower[:n] = x[0::2]
        upper[:n] = x[1::2]
        weights[:n] = p
        return lower, upper, weights, N_used

    # Only possible if |a_1| > 1: put all of the probability at mu = +-1
    mu = np.clip(moments[1], -1.0, 1.0)
    lower[0] = upper[0] = mu
    weights[0] = 1.0
    return lower, upper, weights, 0
