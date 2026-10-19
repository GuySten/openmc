"""Tests for cram.py.

Compares a few Mathematica matrix exponentials to CRAM16/CRAM48.
Tests substep accuracy against self-converged reference solutions.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pytest import approx
from openmc.deplete.cram import (CRAM16, CRAM48, Cram16Solver, Cram48Solver,
                                  IPFCramSolver, ShiftedCram16Solver,
                                  ShiftedCram48Solver)


def test_CRAM16():
    """Test 16-term CRAM."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 0.1

    z = CRAM16(mat, x, dt)

    # Solution from mathematica
    z0 = np.array((0.904837418035960, 0.576799023327476))

    assert z == approx(z0)


def test_CRAM48():
    """Test 48-term CRAM."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 0.1

    z = CRAM48(mat, x, dt)

    # Solution from mathematica
    z0 = np.array((0.904837418035960, 0.576799023327476))

    assert z == approx(z0)


def test_substeps1_matches_original():
    """substeps=1 must be bitwise identical to original spsolve path."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 0.1

    z_orig = CRAM48(mat, x, dt)
    z_sub1 = CRAM48(mat, x, dt, substeps=1)

    np.testing.assert_array_equal(z_sub1, z_orig)


def test_substeps2_matches_two_half_steps():
    """substeps=2 must match two independent CRAM calls with dt/2."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    dt = 1.0

    # Two manual half-steps using original spsolve path
    z_half = CRAM48(mat, x, dt / 2)
    z_two = CRAM48(mat, z_half, dt / 2)

    # Single call with substeps=2
    z_sub2 = CRAM48(mat, x, dt, substeps=2)

    assert z_sub2 == approx(z_two, rel=1e-12)


@pytest.mark.parametrize("substeps", [0, -1])
def test_invalid_substeps(substeps):
    """substeps must be a positive integer at call time."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])

    with pytest.raises(ValueError, match="substeps"):
        CRAM48(mat, x, 0.1, substeps=substeps)


def test_substeps_self_convergence():
    """Increasing substeps converges toward reference solution.

    Uses CRAM16 (alpha0 ~ 2e-16) where substep convergence is visible.
    CRAM48 (alpha0 ~ 2e-47) is already near machine precision for small
    systems; its correctness is verified by the other substep tests.
    """
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    x = np.array([1.0, 1.0])
    dt = 50  # lambda*dt = 50 and 150, stresses CRAM16

    n_ref = CRAM16(mat, x, dt, substeps=128)

    prev_err = np.inf
    for s in [1, 2, 4, 8, 16]:
        n_s = CRAM16(mat, x, dt, substeps=s)
        err = np.linalg.norm(n_s - n_ref) / np.linalg.norm(n_ref)
        assert err < prev_err, \
            f"substeps={s} error {err:.2e} not less than previous {prev_err:.2e}"
        prev_err = err


def _chain_matrix(rates, branches):
    """Transmutation matrix from removal rates and (parent, daughter, rate)."""
    n = len(rates)
    mat = sp.lil_matrix((n, n))
    for i, rate in enumerate(rates):
        mat[i, i] = -rate
    for parent, daughter, rate in branches:
        mat[daughter, parent] += rate
    return sp.csc_matrix(mat)


@pytest.mark.parametrize("decay", [10.0, 150.0, 300.0, 700.0])
def test_shifted_deep_single_decay(decay):
    """A nuclide followed far below CRAM's absolute error, to 1e-304."""
    lam = np.log(2) / 30.0
    mat = _chain_matrix([lam], [])
    dt = decay / lam
    z = ShiftedCram48Solver(mat, np.array([1.0]), dt)
    assert z[0] == approx(np.exp(-lam * dt), rel=1e-12, abs=0)
    # plain CRAM48 loses it below ~1e-31
    if decay > 100.0:
        plain = CRAM48(mat, np.array([1.0]), dt)[0]
        assert abs(plain / np.exp(-lam * dt) - 1.0) > 1e-3


def test_shifted_deep_chain():
    """A(1 s) -> B(10 s) -> stable C: B to 1e-90, A underflows to zero."""
    la, lb = np.log(2), np.log(2) / 10.0
    mat = _chain_matrix([la, lb, 0.0], [(0, 1, la), (1, 2, lb)])
    for dt in (100.0, 500.0, 3000.0):
        z = ShiftedCram48Solver(mat, np.array([1.0, 0.0, 0.0]), dt)
        a = np.exp(-la * dt)
        b = la / (la - lb) * (np.exp(-lb * dt) - a)
        assert z[1] == approx(b, rel=1e-12, abs=0)
        assert z[2] == approx(1.0 - a - b, rel=1e-12, abs=0)
        if a > 1e-300:
            assert z[0] == approx(a, rel=1e-12, abs=0)
        else:
            assert z[0] == 0.0


def test_shifted_capture_loop():
    """A <-> B loop leaking to stable C: the slowest mode sets the shift."""
    a, b, c = 1.0, 2.0, 0.5
    mat = _chain_matrix([a, b + c, 0.0], [(0, 1, a), (1, 0, b), (1, 2, c)])
    tr, det = -(a + b + c), a * c
    l1 = 0.5 * (tr + np.sqrt(tr * tr - 4.0 * det))
    l2 = 0.5 * (tr - np.sqrt(tr * tr - 4.0 * det))
    c1 = -(a + l2) / (b * (l1 - l2))
    for dt in (200.0, 1500.0, 4000.0):
        z = ShiftedCram48Solver(mat, np.array([1.0, 0.0, 0.0]), dt)
        slow = c1 * np.exp(l1 * dt)
        assert z[0] == approx(slow * b, rel=1e-11, abs=0)
        assert z[1] == approx(slow * (a + l1), rel=1e-11, abs=0)


@pytest.mark.parametrize("substeps", [1, 3])
def test_shifted_matches_plain_without_deep_decay(substeps):
    """Nothing decays by more than exp(-50): the shift changes nothing."""
    x = np.array([1.0, 1.0])
    mat = sp.csr_matrix([[-1.0, 0.0], [-2.0, -3.0]])
    for solver, plain in ((ShiftedCram16Solver, Cram16Solver),
                          (ShiftedCram48Solver, Cram48Solver)):
        z = solver(mat, x, 0.1, substeps=substeps)
        assert z == approx(plain(mat, x, 0.1, substeps=substeps), rel=1e-14, abs=0)
