"""Tests for cram.py.

Compares a few Mathematica matrix exponentials to CRAM16/CRAM48.
Tests substep accuracy against self-converged reference solutions.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pytest import approx
from openmc.deplete.cram import (CRAM16, CRAM48, Cram16Solver, Cram48Solver,
                                  IPFCramSolver)


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


# A 12-nuclide system in which nothing produces nuclide 4 (its row holds only its
# own removal) while it feeds nuclides 1 and 3; the initial atoms are in
# nuclide 11.  LU with partial pivoting (spsolve) leaves round-off atoms in
# nuclide 4 (about 5e-16); without pivoting it stays exactly zero.
UNREACHABLE_ENTRIES = [
    (0, 0, -110.49047411784434), (0, 2, 5.941427590531732),
    (1, 0, 2.444343635192702), (1, 1, -903.00316411997),
    (1, 4, 321.17115810617685), (2, 2, -9.521191516501158),
    (2, 9, 6309.812856475182), (3, 0, 18.50507909204667),
    (3, 1, 903.00316411997), (3, 3, -2.6669638421677104e-06),
    (3, 4, 1477.125360990396), (4, 4, -1798.2965190965729),
    (5, 5, -7743.487807940799), (5, 7, 6.247622151826492e-08),
    (6, 6, -92.96750722704776), (6, 10, 21.916132886563467),
    (7, 6, 69.21644859190125), (7, 7, -1.1949882754298622e-06),
    (8, 8, -3.964229735163267), (8, 11, 12.775373755454929),
    (9, 5, 4892.249437058718), (9, 9, -6309.812856475182),
    (10, 8, 0.6968949912340413), (10, 10, -110.80628145389011),
    (11, 6, 23.751058635146503), (11, 11, -459.06971478397),
]


@pytest.mark.parametrize("solver", [CRAM16, CRAM48])
@pytest.mark.parametrize("substeps", [1, 4])
def test_unreachable_nuclide_stays_zero(solver, substeps):
    """A nuclide that no path reaches from the initial composition stays zero."""
    rows, cols, vals = zip(*UNREACHABLE_ENTRIES)
    mat = sp.csc_matrix((vals, (rows, cols)), shape=(12, 12))
    x = np.zeros(12)
    x[11] = 3.5922317192179515e+19

    z = solver(mat, x, 1.0e5, substeps=substeps)

    assert z[4] == 0.0


def test_substeps1_matches_original():
    """substeps=1 must be bitwise identical to the default call."""
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

    # Two manual half-steps
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
