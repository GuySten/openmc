import numpy as np
import pytest

from openmc.data import atomic_mass
from openmc.data.optical_model import (
    AMU_MEV, NEUTRON_MASS_MEV, OpticalPotential,
    koning_delaroche, relativistic_kinematics, solve)


def _pb208(energy):
    mass = atomic_mass('Pb208')*AMU_MEV
    return solve(koning_delaroche(82, 208, energy), energy, mass)


def test_no_potential():
    # With no potential, every S-matrix element is one and nothing scatters
    pot = OpticalPotential(0., 0., 0., 0., 0., 7., 0.6, 7., 0.6, 7., 0.6,
                           6., 0.6)
    res = solve(pot, 50.0, atomic_mass('Pb208')*AMU_MEV)
    assert res.s_plus == pytest.approx(1.0, abs=1e-5)
    assert res.s_minus == pytest.approx(1.0, abs=1e-5)
    assert res.total == pytest.approx(0.0, abs=1e-5)


def test_real_potential_conserves_flux():
    pot = koning_delaroche(26, 56, 50.0)
    pot.wv = pot.wd = pot.wso = 0.0
    res = solve(pot, 50.0, atomic_mass('Fe56')*AMU_MEV)
    assert np.abs(res.s_plus) == pytest.approx(1.0, abs=1e-6)
    assert np.abs(res.s_minus) == pytest.approx(1.0, abs=1e-6)
    assert res.reaction == pytest.approx(0.0, abs=1e-6)
    assert res.elastic == pytest.approx(res.total)


def test_elastic_integral():
    # Integrating the differential elastic cross section gives sigma_el
    res = _pb208(80.0)
    mu, w = np.polynomial.legendre.leggauss(200)
    integral = 2*np.pi*np.sum(w*res.differential_elastic(mu))
    assert integral == pytest.approx(res.elastic, rel=1e-6)


def test_optical_theorem():
    # Im f(0) = k sigma_tot / (4 pi)
    res = _pb208(80.0)
    f0, _ = res.amplitudes(1.0)
    assert f0.imag*0.01 == pytest.approx(res.k*res.total/(4*np.pi), rel=1e-10)


@pytest.mark.parametrize('energy, total', [
    # Finlay et al., Phys. Rev. C 47, 237 (1993), EXFOR 13569018
    (20.0, 5.793), (50.0, 4.291), (100.0, 4.600), (150.0, 3.572)
])
def test_pb208_total(energy, total):
    assert _pb208(energy).total == pytest.approx(total, rel=0.03)


def test_kinematics_nonrelativistic_limit():
    mass = atomic_mass('Fe56')*AMU_MEV
    energy = 1e-3
    k, mu = relativistic_kinematics(energy, mass)
    mu_nr = NEUTRON_MASS_MEV*mass/(NEUTRON_MASS_MEV + mass)
    assert mu == pytest.approx(mu_nr, rel=1e-5)
    assert k*197.3269804 == pytest.approx(np.sqrt(2*mu_nr*energy*mass/(
        NEUTRON_MASS_MEV + mass)), rel=1e-5)
