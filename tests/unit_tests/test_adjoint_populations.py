import h5py
import numpy as np
import pytest

import openmc
from openmc.adjoint_populations import (
    CLASS_FISSION, CLASS_DELAYED, CLASS_PHOTONEUTRON)


def test_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    s.adjoint_populations = {'n_generation': 7, 'photoneutrons': True}
    s.export_to_xml()
    elem = openmc.Settings.from_xml().adjoint_populations
    assert elem == {'n_generation': 7, 'photoneutrons': True}


def test_settings_checks():
    s = openmc.Settings()
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 0}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'photoneutrons': True}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3, 'depth': 3}
    with pytest.raises(TypeError):
        s.adjoint_populations = {'n_generation': 3, 'photoneutrons': 1}


def _write(path, weight, weight_t0, photoneutrons=True):
    n_batches, n_class, n_tag, n_depth = weight.shape
    with h5py.File(path, 'w') as f:
        g = f.create_group('adjoint_populations')
        g['n_generation'] = n_depth - 1
        g['n_batches'] = n_batches
        g['n_class'] = n_class
        g['n_tag'] = n_tag
        g['photoneutrons'] = int(photoneutrons)
        g['weight'] = weight.ravel()
        g['weight_t0'] = weight_t0.ravel()
        g['site_weight'] = np.ones(n_class)
        g['n_roots_last_generation'] = np.zeros(n_class, dtype=np.int64)
        g['raw_weight_last_generation'] = np.zeros(n_class)
        g['n_histories'] = 0


def _read(path, keff):
    with h5py.File(path, 'r') as f:
        return openmc.AdjointPopulations(f['adjoint_populations'], keff)


def test_estimators(run_in_tmpdir):
    """Every estimator against its formula, on exact (noiseless) sums"""
    L, k = 3, 1.2
    w = np.zeros((4, 3, 9, L + 1))
    wt = np.zeros_like(w)
    # Fission roots: prompt, and groups 1 and 2
    w[:, CLASS_FISSION, 0, :] = 100.0
    w[:, CLASS_FISSION, 1, :] = 0.2
    w[:, CLASS_FISSION, 2, :] = 0.4
    wt[:, CLASS_FISSION, 0, 1:] = 100.0 * 2e-8
    # Delayed roots
    w[:, CLASS_DELAYED, 1, :] = 0.25
    w[:, CLASS_DELAYED, 2, :] = 0.45
    # Photoneutrons: prompt, and group 2 from photofission
    w[:, CLASS_PHOTONEUTRON, 0, :] = 0.05
    w[:, CLASS_PHOTONEUTRON, 2, :] = 0.01
    wt[:, CLASS_PHOTONEUTRON, 0, 1:] = 0.05 * 5e-8
    _write('ap.h5', w, wt)
    ap = _read('ap.h5', k)

    I_F, T_F = 100.6, 2e-6
    I_P, T_P = 0.06, 0.05 * 5e-8
    assert ap.n_delayed_groups == 2
    assert ap.beta_eff().n == pytest.approx(0.70 / I_F)
    assert ap.beta_eff(2).n == pytest.approx(0.45 / I_F)
    assert ap.beta_eff(forced=False).n == pytest.approx(0.6 / I_F)
    assert ap.generation_time().n == pytest.approx(T_F / (k * I_F))
    assert ap.generation_time().s == pytest.approx(0.0)

    drho = I_P / I_F
    kp = k / (1 - k * drho)
    assert ap.photoneutron_reactivity().n == pytest.approx(drho)
    assert ap.keff_with_photoneutrons().n == pytest.approx(kp)
    assert 1 / k - 1 / ap.keff_with_photoneutrons().n == pytest.approx(drho)
    assert ap.beta_eff_with_photoneutrons().n == \
        pytest.approx((0.70 + 0.01) / (I_F + I_P))
    assert ap.delta_beta_eff(1).n == \
        pytest.approx(0.25 / (I_F + I_P) - 0.25 / I_F)
    lam_p = (T_F + k * T_P) / (kp * (I_F + I_P))
    assert ap.generation_time_with_photoneutrons().n == pytest.approx(lam_p)
    assert ap.delta_generation_time().n == \
        pytest.approx(lam_p - T_F / (k * I_F))
    assert len(ap.depth_curve('beta_eff', group=1)) == L

    with pytest.raises(ValueError):
        ap.beta_eff(depth=0)
    with pytest.raises(ValueError):
        ap.beta_eff(group=9)


def test_uncertainty_matches_delta_method(run_in_tmpdir):
    """The reported sigma of a ratio is the delta-method sigma"""
    rng = np.random.default_rng(1)
    n = 200
    w = np.zeros((n, 3, 9, 2))
    num = 0.7 + 0.05 * rng.standard_normal(n)
    den = 100.0 + 2.0 * rng.standard_normal(n) + 10.0 * (num - 0.7)
    w[:, CLASS_DELAYED, 1, 1] = num
    w[:, CLASS_FISSION, 0, 1] = den
    _write('ap.h5', w, np.zeros_like(w), photoneutrons=False)
    ap = _read('ap.h5', 1.0)

    r = num.mean() / den.mean()
    z = (num - num.mean()) / den.mean() - r * (den - den.mean()) / den.mean()
    b = ap.beta_eff()
    assert b.n == pytest.approx(r)
    assert b.s == pytest.approx(z.std(ddof=1) / np.sqrt(n), rel=1e-4)
    with pytest.raises(ValueError):
        ap.photoneutron_reactivity()
