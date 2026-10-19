"""Chebyshev Rational Approximation Method module

Implements two different forms of CRAM for use in openmc.deplete.
"""

from functools import partial
import math
import numbers

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigs, spsolve, splu

from openmc.checkvalue import check_type, check_length, check_greater_than
from .abc import DepSystemSolver
from .._sparse_compat import csc_array, eye_array

__all__ = ["CRAM16", "CRAM48", "Cram16Solver", "Cram48Solver", "IPFCramSolver",
           "ShiftedIPFCramSolver", "ShiftedCram16Solver", "ShiftedCram48Solver"]


class IPFCramSolver(DepSystemSolver):
    r"""CRAM depletion solver that uses incomplete partial factorization

    Provides a :meth:`__call__` that utilizes an incomplete
    partial factorization (IPF) for the Chebyshev Rational Approximation
    Method (CRAM), as described in the following paper: M. Pusa, "`Higher-Order
    Chebyshev Rational Approximation Method and Application to Burnup Equations
    <https://doi.org/10.13182/NSE15-26>`_," Nucl. Sci. Eng., 182:3, 297-318.

    When `substeps` > 1, the time interval is split into `substeps` identical
    sub-intervals and LU factorizations are reused across them, as described
    in: A. Isotalo and M. Pusa, "`Improving the Accuracy of the Chebyshev
    Rational Approximation Method Using Substeps
    <https://doi.org/10.13182/NSE15-67>`_," Nucl. Sci. Eng., 183:1, 65-77.

    Parameters
    ----------
    alpha : numpy.ndarray
        Complex residues of poles used in the factorization. Must be a
        vector with even number of items.
    theta : numpy.ndarray
        Complex poles. Must have an equal size as ``alpha``.
    alpha0 : float
        Limit of the approximation at infinity

    Attributes
    ----------
    alpha : numpy.ndarray
        Complex residues of poles :attr:`theta` in the incomplete partial
        factorization. Denoted as :math:`\tilde{\alpha}`
    theta : numpy.ndarray
        Complex poles :math:`\theta` of the rational approximation
    alpha0 : float
        Limit of the approximation at infinity

    """

    def __init__(self, alpha, theta, alpha0):
        check_type("alpha", alpha, np.ndarray, numbers.Complex)
        check_type("theta", theta, np.ndarray, numbers.Complex)
        check_length("theta", theta, alpha.size)
        check_type("alpha0", alpha0, numbers.Real)
        self.alpha = alpha
        self.theta = theta
        self.alpha0 = alpha0

    def __call__(self, A, n0, dt, substeps=1):
        """Solve depletion equations using IPF CRAM

        Parameters
        ----------
        A : scipy.sparse.csc_array
            Sparse transmutation matrix ``A[j, i]`` desribing rates at
            which isotope ``i`` transmutes to isotope ``j``
        n0 : numpy.ndarray
            Initial compositions, typically given in number of atoms in some
            material or an atom density
        dt : float
            Time [s] of the specific interval to be solved
        substeps : int, optional
            Number of substeps per depletion interval.

        Returns
        -------
        numpy.ndarray
            Final compositions after ``dt``

        """
        check_type("substeps", substeps, numbers.Integral)
        check_greater_than("substeps", substeps, 0)

        step_dt = dt if substeps == 1 else dt / substeps
        A = step_dt * csc_array(A, dtype=np.float64)
        ident = eye_array(A.shape[0], format='csc')

        if substeps == 1:
            solvers = [partial(spsolve, A - theta * ident) for theta in self.theta]
        else:
            # Pre-compute LU factorizations and reuse them across substeps.
            solvers = [splu(A - theta * ident).solve for theta in self.theta]

        y = n0.copy()
        for _ in range(substeps):
            for alpha, solve in zip(self.alpha, solvers):
                y += 2 * np.real(alpha * solve(y))
            y *= self.alpha0
        return y


# Coefficients for IPF Cram 16
c16_alpha = np.array([
    +5.464930576870210e+3 - 3.797983575308356e+4j,
    +9.045112476907548e+1 - 1.115537522430261e+3j,
    +2.344818070467641e+2 - 4.228020157070496e+2j,
    +9.453304067358312e+1 - 2.951294291446048e+2j,
    +7.283792954673409e+2 - 1.205646080220011e+5j,
    +3.648229059594851e+1 - 1.155509621409682e+2j,
    +2.547321630156819e+1 - 2.639500283021502e+1j,
    +2.394538338734709e+1 - 5.650522971778156e+0j],
    dtype=np.complex128)

c16_theta = np.array([
    +3.509103608414918 + 8.436198985884374j,
    +5.948152268951177 + 3.587457362018322j,
    -5.264971343442647 + 16.22022147316793j,
    +1.419375897185666 + 10.92536348449672j,
    +6.416177699099435 + 1.194122393370139j,
    +4.993174737717997 + 5.996881713603942j,
    -1.413928462488886 + 13.49772569889275j,
    -10.84391707869699 + 19.27744616718165j],
    dtype=np.complex128)

c16_alpha0 = 2.124853710495224e-16
Cram16Solver = IPFCramSolver(c16_alpha, c16_theta, c16_alpha0)
CRAM16 = Cram16Solver.__call__

del c16_alpha, c16_alpha0, c16_theta

# Coefficients for 48th order IPF Cram

theta_r = np.array([
    -4.465731934165702e+1, -5.284616241568964e+0,
    -8.867715667624458e+0, +3.493013124279215e+0,
    +1.564102508858634e+1, +1.742097597385893e+1,
    -2.834466755180654e+1, +1.661569367939544e+1,
    +8.011836167974721e+0, -2.056267541998229e+0,
    +1.449208170441839e+1, +1.853807176907916e+1,
    +9.932562704505182e+0, -2.244223871767187e+1,
    +8.590014121680897e-1, -1.286192925744479e+1,
    +1.164596909542055e+1, +1.806076684783089e+1,
    +5.870672154659249e+0, -3.542938819659747e+1,
    +1.901323489060250e+1, +1.885508331552577e+1,
    -1.734689708174982e+1, +1.316284237125190e+1])

theta_i = np.array([
    +6.233225190695437e+1, +4.057499381311059e+1,
    +4.325515754166724e+1, +3.281615453173585e+1,
    +1.558061616372237e+1, +1.076629305714420e+1,
    +5.492841024648724e+1, +1.316994930024688e+1,
    +2.780232111309410e+1, +3.794824788914354e+1,
    +1.799988210051809e+1, +5.974332563100539e+0,
    +2.532823409972962e+1, +5.179633600312162e+1,
    +3.536456194294350e+1, +4.600304902833652e+1,
    +2.287153304140217e+1, +8.368200580099821e+0,
    +3.029700159040121e+1, +5.834381701800013e+1,
    +1.194282058271408e+0, +3.583428564427879e+0,
    +4.883941101108207e+1, +2.042951874827759e+1])

c48_theta = np.array(theta_r + theta_i * 1j, dtype=np.complex128)

alpha_r = np.array([
    +6.387380733878774e+2, +1.909896179065730e+2,
    +4.236195226571914e+2, +4.645770595258726e+2,
    +7.765163276752433e+2, +1.907115136768522e+3,
    +2.909892685603256e+3, +1.944772206620450e+2,
    +1.382799786972332e+5, +5.628442079602433e+3,
    +2.151681283794220e+2, +1.324720240514420e+3,
    +1.617548476343347e+4, +1.112729040439685e+2,
    +1.074624783191125e+2, +8.835727765158191e+1,
    +9.354078136054179e+1, +9.418142823531573e+1,
    +1.040012390717851e+2, +6.861882624343235e+1,
    +8.766654491283722e+1, +1.056007619389650e+2,
    +7.738987569039419e+1, +1.041366366475571e+2])

alpha_i = np.array([
    -6.743912502859256e+2, -3.973203432721332e+2,
    -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4,
    -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4,
    -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1,
    -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2,
    -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3,
    -4.311715386228984e+1, -2.777743732451969e+2])

c48_alpha = np.array(alpha_r + alpha_i * 1j, dtype=np.complex128)

c48_alpha0 = 2.258038182743983e-47

Cram48Solver = IPFCramSolver(c48_alpha, c48_theta, c48_alpha0)

del c48_alpha, c48_alpha0, c48_theta, alpha_r, alpha_i, theta_r, theta_i

CRAM48 = Cram48Solver.__call__


class ShiftedIPFCramSolver(IPFCramSolver):
    r"""IPF CRAM with a spectral shift for each nuclide

    CRAM's error is absolute: about :math:`\alpha_0` times the atoms a
    nuclide can draw on, so a nuclide whose atoms decay by more than about
    :math:`\alpha_0` within one interval is lost in the approximation error.
    For any :math:`\mu`,

    .. math::

        e^{\Delta t A} = e^{-\mu \Delta t}\, e^{\Delta t (A + \mu I)},

    and CRAM may be applied to the second factor as long as the spectrum of
    :math:`A + \mu I` stays on the non-positive real axis. A nuclide depends
    only on itself and its ancestors, so it is solved with :math:`\mu` equal
    to the slowest decay rate among them (the Perron root of each strongly
    connected block, such as a capture loop). Nuclides are grouped in levels
    of :math:`e^{-L}`, each solved once over its ancestors, which makes the
    error relative for every nuclide down to the bottom of the double range.

    Parameters
    ----------
    alpha : numpy.ndarray
        Complex residues of poles used in the factorization.
    theta : numpy.ndarray
        Complex poles. Must have an equal size as ``alpha``.
    alpha0 : float
        Limit of the approximation at infinity
    level : float, optional
        Width :math:`L` of a shift level: a nuclide drops by at most
        :math:`e^{-L}` relative to its shift.

    """

    def __init__(self, alpha, theta, alpha0, level=50.0):
        super().__init__(alpha, theta, alpha0)
        check_type("level", level, numbers.Real)
        check_greater_than("level", level, 0.0)
        self.level = float(level)

    def __call__(self, A, n0, dt, substeps=1):
        """Solve depletion equations using IPF CRAM with spectral shifts

        Parameters
        ----------
        A : scipy.sparse.csc_array
            Sparse transmutation matrix ``A[j, i]`` desribing rates at
            which isotope ``i`` transmutes to isotope ``j``
        n0 : numpy.ndarray
            Initial compositions, typically given in number of atoms in some
            material or an atom density
        dt : float
            Time [s] of the specific interval to be solved
        substeps : int, optional
            Number of substeps per depletion interval.

        Returns
        -------
        numpy.ndarray
            Final compositions after ``dt``

        """
        check_type("substeps", substeps, numbers.Integral)
        check_greater_than("substeps", substeps, 0)

        A = csc_array(A, dtype=np.float64)
        step_dt = dt / substeps
        levels, graph, node_rate = _shift_levels(A, n0, step_dt, self.level)
        threshold = -math.log(self.alpha0) + 10.0
        groups = []
        position = np.empty(A.shape[0], dtype=int)
        for shift, members, subset in levels:
            dropped = _negligible_sources(graph, node_rate, members, subset,
                                          shift, step_dt, threshold)
            # Subsets come in topological order, so the shifted matrices are
            # block lower triangular. Factoring them as they are, with
            # diagonal pivots, makes each solve a forward substitution that
            # keeps the round-off of every nuclide relative to its own
            # value; partial pivoting would route a nuclide through the rows
            # of its descendants and leak their round-off into it.
            sub = csc_array(step_dt * A[subset][:, subset])
            ident = eye_array(subset.size, format='csc')
            thetas = [theta - shift * step_dt for theta in self.theta]
            solvers = [splu(csc_array(sub - th * ident), permc_spec='NATURAL',
                            diag_pivot_thresh=0.0).solve for th in thetas]
            # exp(-shift dt), with the rounding error of the product kept
            product = shift * step_dt
            factor = math.exp(-product) * (1.0 - math.fma(shift, step_dt, -product))
            position[subset] = np.arange(subset.size)
            local = position[members]
            groups.append((subset, local, members, solvers, factor,
                           position[dropped]))

        y = np.asarray(n0, dtype=np.float64).copy()
        for _ in range(substeps):
            out = np.zeros_like(y)
            for subset, local, members, solvers, factor, dropped in groups:
                u = y[subset]
                u[dropped] = 0.0
                for alpha, solve in zip(self.alpha, solvers):
                    u = u + 2 * np.real(alpha * solve(u))
                u *= self.alpha0
                out[members] = u[local] * factor
            y = out
        return y


def _shift_levels(A, n0, dt, width):
    r"""Shift levels of a transmutation matrix for one interval.

    Returns a list of ``(shift, members, subset)``: the shift :math:`\mu`,
    the nuclides that take their result from it, and the ancestor-closed set
    of nuclides to solve, in topological order (parents first). The first
    entry, with zero shift, covers every nuclide not assigned elsewhere;
    nuclides that must have underflowed are left out and come back as zero.
    """
    n = A.shape[0]
    n0 = np.asarray(n0, dtype=np.float64)
    coo = A.tocoo()
    off = coo.row != coo.col
    # graph[j, i] != 0 when nuclide j produces nuclide i
    graph = csc_array((np.ones(off.sum()), (coo.col[off], coo.row[off])),
                      shape=(n, n)).tocsr()
    everyone = np.arange(n)

    # Nuclides that hold or receive atoms
    active = n0 != 0.0
    frontier = active.copy()
    while frontier.any():
        reached = (graph.T @ frontier.astype(np.float64)) != 0.0
        frontier = reached & ~active
        active |= frontier
    largest = np.abs(n0).max() if n0.size else 0.0

    # Strongly connected blocks, and every nuclide's rank in a topological
    # order of the blocks
    nblocks, label = connected_components(graph, directed=True,
                                          connection='strong')
    parent_blocks = [set() for _ in range(nblocks)]
    rows, cols = graph.nonzero()          # edge rows -> cols
    for j, i in zip(rows, cols):
        if label[j] != label[i]:
            parent_blocks[label[i]].add(label[j])
    order = _topological(nblocks, parent_blocks)
    block_rank = np.empty(nblocks, dtype=int)
    block_rank[order] = np.arange(nblocks)
    rank = block_rank[label]
    by_rank = np.argsort(rank, kind='stable')
    if not active.any() or largest == 0.0:
        return [(0.0, everyone, by_rank)], graph, -A.diagonal()

    # The decay rate of each block's slowest mode
    diag = A.diagonal()
    block_rate = np.empty(nblocks)
    members_of = [[] for _ in range(nblocks)]
    for i in range(n):
        members_of[label[i]].append(i)
    for b, members in enumerate(members_of):
        if len(members) == 1:
            block_rate[b] = -diag[members[0]]
        else:
            idx = np.array(members)
            block_rate[b] = -_perron_root(A[idx][:, idx])

    # Propagate the slowest rate from ancestors that hold or receive atoms,
    # in topological order
    rate = np.full(nblocks, np.inf)
    block_active = np.zeros(nblocks, dtype=bool)
    block_active[label[active]] = True
    for b in order:
        if not block_active[b]:
            continue
        slowest = block_rate[b]
        for p in parent_blocks[b]:
            slowest = min(slowest, rate[p])     # inactive parents are inf
        rate[b] = max(0.0, slowest)

    decay = rate[label] * dt
    cutoff = math.log(largest) + 745.0 + 69.0
    level = np.where(active, np.floor(decay / width), 0.0)
    negligible = active & (decay > cutoff)
    level[negligible] = -1

    result = []
    base = everyone[level == 0]
    result.append((0.0, base, by_rank))
    for lv in np.unique(level[level > 0]):
        members = np.flatnonzero(level == lv)
        # ancestors of the members that hold or receive atoms
        closure = np.zeros(n, dtype=bool)
        closure[members] = True
        frontier = closure.copy()
        while frontier.any():
            reached = (graph @ frontier.astype(np.float64)) != 0.0
            frontier = reached & ~closure & active
            closure |= frontier
        subset = np.flatnonzero(closure)
        subset = subset[np.argsort(rank[subset], kind='stable')]
        result.append((lv * width / dt, members, subset))
    return result, graph, block_rate[label]


def _negligible_sources(graph, node_rate, members, subset, shift, dt,
                        threshold):
    """Nuclides of a solve whose atoms can only reach its members through
    decay faster than ``shift`` by more than ``threshold`` / dt.

    Their true contribution, at most exp(-threshold) of their atoms on the
    shifted scale, is below CRAM's own noise for them (alpha0 times their
    atoms, which r(z) leaves behind as z -> -inf and passes on to their
    descendants), so leaving their atoms out of the solve loses less than
    keeping them.
    """
    in_subset = np.zeros(graph.shape[0], dtype=bool)
    in_subset[subset] = True
    is_member = np.zeros(graph.shape[0], dtype=bool)
    is_member[members] = True
    # slowest rate on any path from a nuclide to a member, children first
    slowest = np.full(graph.shape[0], np.inf)
    indptr, indices = graph.indptr, graph.indices
    for j in subset[::-1]:
        best = np.inf
        for c in indices[indptr[j]:indptr[j + 1]]:
            if in_subset[c] and slowest[c] < np.inf:
                best = min(best, slowest[c])
        if is_member[j] or best < np.inf:
            slowest[j] = min(node_rate[j], best)
    return subset[(slowest[subset] - shift) * dt > threshold]


def _topological(nblocks, parent_blocks):
    """Blocks in an order where every parent precedes its children."""
    children = [[] for _ in range(nblocks)]
    indegree = np.zeros(nblocks, dtype=int)
    for b, parents in enumerate(parent_blocks):
        indegree[b] = len(parents)
        for p in parents:
            children[p].append(b)
    ready = [b for b in range(nblocks) if indegree[b] == 0]
    order = []
    while ready:
        b = ready.pop()
        order.append(b)
        for c in children[b]:
            indegree[c] -= 1
            if indegree[c] == 0:
                ready.append(c)
    return order


def _perron_root(block):
    """Rightmost (real) eigenvalue of a Metzler matrix block.

    Small blocks: bisection with the M-matrix test, which keeps the upper end
    so that a shift built from it never moves an eigenvalue right of zero.
    Large blocks: the largest eigenvalue of (-block)^-1 with ARPACK. Any
    failure gives 0, i.e. no shift.
    """
    block = csc_array(block)
    m = block.shape[0]
    if m <= 64:
        a = block.toarray()
        lo = np.max(np.diag(a))
        hi = np.max(np.diag(a) + np.abs(a - np.diag(np.diag(a))).sum(axis=0))
        hi = max(hi, lo)

        def left_of(s):
            z = s * np.eye(m) - a
            for k in range(m):
                pivot = z[k, k]
                if not pivot > 0.0:
                    return False
                z[k + 1:, k:] -= np.outer(z[k + 1:, k] / pivot, z[k, k:])
            return True

        while not left_of(hi):
            hi += max(1.0, abs(hi))
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if left_of(mid):
                hi = mid
            else:
                lo = mid
        return hi
    try:
        lu = splu(csc_array(-block))
        op = LinearOperator((m, m), matvec=lu.solve, dtype=np.float64)
        value = eigs(op, k=1, which='LM', return_eigenvectors=False)[0]
        root = -1.0 / value.real
        # A small safety margin towards zero keeps the shift conservative.
        return min(0.0, root * (1.0 - 1e-10))
    except Exception:
        return 0.0


ShiftedCram16Solver = ShiftedIPFCramSolver(
    Cram16Solver.alpha, Cram16Solver.theta, Cram16Solver.alpha0)
ShiftedCram48Solver = ShiftedIPFCramSolver(
    Cram48Solver.alpha, Cram48Solver.theta, Cram48Solver.alpha0)
