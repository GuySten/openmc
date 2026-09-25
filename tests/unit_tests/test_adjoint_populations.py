import h5py
import numpy as np
import pytest

import openmc
from openmc.adjoint_populations import (
    CLASS_FISSION, CLASS_DELAYED, CLASS_PHOTONEUTRON, CLASS_FISSION_BRANCH,
    CLASS_DELAYED_BRANCH)


def test_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    value = {'n_generation': 7, 'photoneutrons': True,
             'perturbed_importance': True}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value


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


def _write(path, weight, weight_t0, photoneutrons=True, perturbed=True):
    n_batches, n_class, n_tag, n_depth = weight.shape
    with h5py.File(path, 'w') as f:
        g = f.create_group('adjoint_populations')
        g['n_generation'] = n_depth - 1
        g['n_batches'] = n_batches
        g['n_class'] = n_class
        g['n_tag'] = n_tag
        g['photoneutrons'] = int(photoneutrons)
        g['perturbed_importance'] = int(perturbed)
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
    w = np.zeros((4, 5, 9, L + 1))
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
    # Photoneutron branches of the fission- and delayed-root trees
    w[:, CLASS_FISSION_BRANCH, 0, 1:] = 0.3
    wt[:, CLASS_FISSION_BRANCH, 0, 1:] = 0.3 * 4e-8
    w[:, CLASS_DELAYED_BRANCH, 1, 1:] = 0.002
    w[:, CLASS_DELAYED_BRANCH, 2, 1:] = 0.003
    _write('ap.h5', w, wt)
    ap = _read('ap.h5', k)

    I_F, T_F = 100.6, 2e-6
    I_P, T_P = 0.06, 0.05 * 5e-8
    B_F, BT_F = 0.3, 0.3 * 4e-8
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
    g = 1 + (L - 1) * k * drho
    den = (I_F + B_F) / g + k * I_P
    assert ap.beta_eff_with_photoneutrons().n == \
        pytest.approx(((0.70 + 0.005) / g + k * 0.01) / den)
    assert ap.delta_beta_eff(1).n == \
        pytest.approx((0.25 + 0.002) / g / den - 0.25 / I_F)
    lam_p = ((T_F + BT_F) / g + k * T_P) / (kp * den)
    assert ap.generation_time_with_photoneutrons().n == pytest.approx(lam_p)
    assert ap.delta_generation_time().n == \
        pytest.approx(lam_p - T_F / (k * I_F))
    assert len(ap.depth_curve('beta_eff', group=1)) == L

    with pytest.raises(ValueError):
        ap.beta_eff(depth=0)
    with pytest.raises(ValueError):
        ap.beta_eff(group=9)

    # Without perturbed importance, the changes are refused
    _write('ap2.h5', w, wt, perturbed=False)
    ap2 = _read('ap2.h5', k)
    assert ap2.photoneutron_reactivity().n == pytest.approx(drho)
    with pytest.raises(ValueError):
        ap2.delta_beta_eff()
    with pytest.raises(ValueError):
        ap2.delta_generation_time()


def test_uncertainty_matches_delta_method(run_in_tmpdir):
    """The reported sigma of a ratio is the delta-method sigma"""
    rng = np.random.default_rng(1)
    n = 200
    w = np.zeros((n, 5, 9, 2))
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


def test_energy_group_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    value = {'n_generation': 5, 'photoneutrons': True,
             'photoneutron_energy_bins': [2.2246e6, 3.0e6, 2.0e7],
             'photoneutron_energy_variable': 'photoneutron'}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value


def test_energy_group_settings_checks():
    s = openmc.Settings()
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_energy_bins': [3.0e6]}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_energy_bins': [3.0e6, 3.0e6]}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_energy_variable': 'gamma'}


def test_energy_group_estimators(run_in_tmpdir):
    """Grouped photoneutron sums: their reactivity contributions add up to
    the ungrouped one, and each group's importance per weight is its own
    growth over the fission roots'."""
    L, k, n_b = 2, 1.0, 5
    rng = np.random.default_rng(1)
    w = np.zeros((n_b, 5, 9, L + 1))
    w[:, CLASS_FISSION, 0] = [100.0, 90.0, 80.0]
    # Two groups of photoneutron roots, growing differently with depth
    ew = np.zeros((n_b, 2, L + 1))
    ew[:, 0] = [1.0, 0.5, 0.25]
    ew[:, 1] = [2.0, 2.0, 2.0]
    ew *= rng.uniform(0.9, 1.1, size=(n_b, 1, 1))
    w[:, CLASS_PHOTONEUTRON, 0] = ew.sum(axis=1)
    _write('ap.h5', w, np.zeros_like(w))
    with h5py.File('ap.h5', 'a') as f:
        g = f['adjoint_populations']
        g['photoneutron_energy_bins'] = [2.2246e6, 3.0e6, 2.0e7]
        g['photoneutron_energy_variable'] = 'photon_birth'
        g['photoneutron_energy_weight'] = ew.ravel()
    ap = _read('ap.h5', k)
    assert ap.photoneutron_energy_variable == 'photon_birth'
    by_e = ap.photoneutron_reactivity_by_energy(depth=2)
    total = ap.photoneutron_reactivity(depth=2)
    assert sum(x.n for x in by_e) == pytest.approx(total.n)
    imp = ap.photoneutron_importance_by_energy(depth=2)
    fis = 80.0 / 100.0
    assert imp[0].n == pytest.approx((0.25 / 1.0) / fis, rel=1e-9)
    assert imp[1].n == pytest.approx(1.0 / fis, rel=1e-9)


def test_energy_group_estimators_need_the_tally(run_in_tmpdir):
    w = np.ones((3, 5, 9, 3))
    _write('ap.h5', w, np.zeros_like(w))
    ap = _read('ap.h5', 1.0)
    with pytest.raises(ValueError):
        ap.photoneutron_reactivity_by_energy()


def test_fission_nuclide_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    value = {'n_generation': 5, 'photoneutrons': True,
             'photoneutron_energy_bins': [2.2246e6, 3.0e6, 2.0e7],
             'photoneutron_fission_nuclides': ['U235', 'U238']}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value


def test_fission_nuclide_settings_checks():
    s = openmc.Settings()
    with pytest.raises(TypeError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_fission_nuclides': 'U235'}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_fission_nuclides': []}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_fission_nuclides':
                                 ['U235', 'U235']}


def test_fission_nuclide_estimators(run_in_tmpdir):
    """Photoneutron sums by fissioning nuclide and energy group: they add up
    to the energy-group sums and to the ungrouped reactivity, and each bin's
    importance is its own growth over the fission roots'."""
    L, k, n_b = 2, 1.0, 5
    rng = np.random.default_rng(2)
    w = np.zeros((n_b, 5, 9, L + 1))
    w[:, CLASS_FISSION, 0] = [100.0, 90.0, 80.0]
    # [batch, nuclide bin (U235, U238, other), energy group, depth]
    nw = np.zeros((n_b, 3, 2, L + 1))
    nw[:, 0, 0] = [1.0, 0.5, 0.25]
    nw[:, 0, 1] = [2.0, 2.0, 2.0]
    nw[:, 1, 0] = [0.5, 0.4, 0.3]
    nw[:, 1, 1] = [0.2, 0.1, 0.05]
    nw *= rng.uniform(0.9, 1.1, size=(n_b, 1, 1, 1))
    ew = nw.sum(axis=1)
    w[:, CLASS_PHOTONEUTRON, 0] = ew.sum(axis=1)
    _write('ap.h5', w, np.zeros_like(w))
    with h5py.File('ap.h5', 'a') as f:
        g = f['adjoint_populations']
        g['photoneutron_energy_bins'] = [2.2246e6, 3.0e6, 2.0e7]
        g['photoneutron_energy_variable'] = 'photon_birth'
        g['photoneutron_energy_weight'] = ew.ravel()
        g['photoneutron_fission_nuclides'] = 'U235 U238 other'
        g['photoneutron_nuclide_weight'] = nw.ravel()
    ap = _read('ap.h5', k)
    assert ap.photoneutron_fission_nuclides == ['U235', 'U238', 'other']
    by_n = ap.photoneutron_reactivity_by_nuclide(depth=2)
    by_e = ap.photoneutron_reactivity_by_energy(depth=2)
    for e in range(2):
        assert sum(by_n[j][e].n for j in by_n) == pytest.approx(by_e[e].n)
    total = ap.photoneutron_reactivity(depth=2)
    assert sum(x.n for row in by_n.values() for x in row) == \
        pytest.approx(total.n)
    imp = ap.photoneutron_importance_by_nuclide(depth=2)
    fis = 80.0 / 100.0
    assert imp['U238'][0].n == pytest.approx((0.3 / 0.5) / fis, rel=1e-9)
    assert np.isnan(imp['other'][0].n)


def test_fission_nuclide_estimators_need_the_tally(run_in_tmpdir):
    w = np.ones((3, 5, 9, 3))
    _write('ap.h5', w, np.zeros_like(w))
    ap = _read('ap.h5', 1.0)
    with pytest.raises(ValueError):
        ap.photoneutron_reactivity_by_nuclide()


def test_probe_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    lines = list(openmc.probe_line_energies(2.2246e6, 12.6e6, 7))
    value = {'n_generation': 5, 'photoneutrons': True,
             'photoneutron_fission_nuclides': ['U235', 'U238'],
             'photoneutron_probe_energies': lines,
             'photoneutron_probe_fraction': 0.5}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value


def test_probe_settings_checks():
    s = openmc.Settings()
    for bad in ([], [3.0e6, 2.5e6], [0.0, 3.0e6], [3.0e6, 3.0e6]):
        with pytest.raises(ValueError):
            s.adjoint_populations = {'n_generation': 3,
                                     'photoneutron_probe_energies': bad}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_probe_fraction': 0.0}


def test_probe_line_energies():
    e = openmc.probe_line_energies(2.2246e6, 12.6e6, 11, first=1.0e3)
    assert e[0] == pytest.approx(2.2246e6 + 1.0e3)
    assert e[-1] == pytest.approx(12.6e6)
    # Evenly spaced in photoneutron lethargy
    du = np.diff(np.log(e - 2.2246e6))
    assert np.allclose(du, du[0])


def _resonant_target():
    """A target with a smooth production cross section and a narrow
    resonance, on a grid that resolves it."""
    thr = 1.6647e6
    e = np.unique(np.concatenate([
        np.linspace(thr, 12.6e6, 400),
        2.431e6 + np.linspace(-5e3, 5e3, 201)]))
    smooth = 1e-3 * np.clip(e - thr, 0.0, None) / 1e6
    res = 0.08 / (1.0 + ((e - 2.431e6) / 390.0) ** 2)
    sigma = np.where(e > thr, smooth + res, 0.0)
    removal = 0.08 * (e / 2e6) ** -0.5 + sigma
    return openmc.PhotoneutronTarget(e, sigma, removal)


def _smooth_parts(target, E):
    """Direct and scattered importance whose smooth factors A, B are linear
    in photoneutron lethargy, so the rebuilding is exact."""
    u = np.log(E - target.threshold)
    a = 2.0 + 0.1 * u
    b = 1e-4 * (u - np.log(1e3)) + 1e-6
    s, r = target.sigma_at(E), target.sigma_removal_at(E)
    return s * a / r, b / r


def test_probe_importance_rebuilds_resonance():
    """The direct part follows the resonance, the scattered part does not:
    rebuilt from lines that miss the resonance, the importance matches the
    true function at the resonance, and the refined table interpolates it
    within the tolerance."""
    t = _resonant_target()
    lines = openmc.probe_line_energies(t.threshold, 12.6e6, 30)
    d, s = _smooth_parts(t, lines)
    n_b = 4
    imp = openmc.ProbeImportance(lines, np.tile(d, (n_b, 1)),
                                 np.tile(s, (n_b, 1)), np.ones(n_b), t,
                                 rtol=0.01)
    E = np.concatenate([np.linspace(lines[0], lines[-1], 3000),
                        2.431e6 + np.linspace(-2e3, 2e3, 101)])
    dt, st = _smooth_parts(t, E)
    assert np.allclose(imp.rebuilt(E), dt + st, rtol=1e-9)
    assert imp.interpolation_error(n_points=20) < 0.01
    assert imp.interpolation_error(energies=E) < 0.01
    # The resonance peak is in the refined grid's reach
    peak = imp(np.array([2.431e6]))[0]
    assert peak == pytest.approx((dt + st)[np.argmin(abs(E - 2.431e6))],
                                 rel=0.01)
    # Folding lines and a continuum uses the rebuilt function
    y = np.array([1e-3, 2e-3])
    el = np.array([2.4308e6, 5.0e6])
    dl, sl = _smooth_parts(t, el)
    assert imp.fold(el, y).n == pytest.approx((y * (dl + sl)).sum(), rel=1e-9)
    ce = np.linspace(3.0e6, 4.0e6, 11)
    cp = np.full_like(ce, 1e-9)
    x = np.unique(np.concatenate([ce, t.energy[(t.energy >= 3e6) &
                                               (t.energy <= 4e6)]]))
    dx, sx = _smooth_parts(t, x)
    ref = np.trapezoid(1e-9 * (dx + sx), x)
    assert imp.fold(continuum=(ce, cp)).n == pytest.approx(ref, rel=1e-9)


def test_probe_importance_statistics():
    """Batch statistics of the table and of a fold are the ratio of batch
    means with a delta-method sigma, like every other estimator."""
    t = _resonant_target()
    lines = openmc.probe_line_energies(t.threshold, 12.6e6, 10)
    d, s = _smooth_parts(t, lines)
    rng = np.random.default_rng(4)
    f = rng.uniform(0.9, 1.1, 6)
    noise = rng.uniform(0.8, 1.2, (6, 1))
    imp = openmc.ProbeImportance(lines, d * noise, s * noise, f, t)
    k = 3
    ref = openmc.adjoint_populations._ratio((d[k] + s[k]) * noise[:, 0], f)
    got = imp.fold([lines[k]], [1.0])
    assert got.n == pytest.approx(ref.n, rel=1e-12)
    assert got.s == pytest.approx(ref.s, rel=1e-9)


def test_probe_estimators(run_in_tmpdir):
    L, n_b, K = 2, 3, 4
    w = np.zeros((n_b, 5, 9, L + 1))
    w[:, CLASS_FISSION, 0] = [100.0, 90.0, 80.0]
    pw = np.zeros((n_b, 2, K, 2, L + 1))
    pw[:, 0, :, 0, 0] = 2.0
    pw[:, 0, :, 1, 0] = 1.0
    pw[:, 0, :, 0, 2] = 0.5
    pw[:, 0, :, 1, 2] = 0.25
    pf = np.zeros((n_b, 2))
    pf[:, 0] = 40.0
    _write('ap.h5', w, np.zeros_like(w))
    with h5py.File('ap.h5', 'a') as f:
        g = f['adjoint_populations']
        g['probe_energies'] = np.linspace(2.3e6, 5e6, K)
        g['probe_fission_nuclides'] = 'U235 other'
        g['probe_weight'] = pw.ravel()
        g['probe_fission_weight'] = pf.ravel()
    ap = _read('ap.h5', 1.0)
    assert ap.probe_fission_nuclides == ['U235', 'other']
    d, s = ap.probe_photoneutron_yield('U235')
    assert d[0].n == pytest.approx(2.0 / 40.0)
    assert s[3].n == pytest.approx(1.0 / 40.0)
    d, s = ap.probe_reactivity('U235', depth=2)
    assert d[1].n == pytest.approx(0.5 / 80.0)
    assert s[1].n == pytest.approx(0.25 / 80.0)
    with pytest.raises(ValueError):
        ap.probe_reactivity()          # two bins: one must be named
    w2 = np.ones((3, 5, 9, 3))
    _write('ap2.h5', w2, np.zeros_like(w2))
    with pytest.raises(ValueError):
        _read('ap2.h5', 1.0).probe_reactivity()


def test_probe_importance_empty_bin():
    """A nuclide bin no probe reached rebuilds to zero, with no error."""
    t = _resonant_target()
    lines = openmc.probe_line_energies(t.threshold, 12.6e6, 10)
    z = np.zeros((3, lines.size))
    imp = openmc.ProbeImportance(lines, z, z, np.ones(3), t)
    assert not imp.mean.any()
    assert imp.interpolation_error() == 0.0
    assert imp.fold([3.0e6], [1.0]).n == 0.0


def test_probe_labels_and_rays(run_in_tmpdir):
    """Three probe labels, with rays: direct = rays + coherent-only probes,
    scattered = energy-changed probes; the depth-0 yield rescales the rays by
    their own fission weight."""
    L, n_b, K, M = 2, 3, 4, 5
    w = np.zeros((n_b, 5, 9, L + 1))
    w[:, CLASS_FISSION, 0] = [100.0, 90.0, 80.0]
    pw = np.zeros((n_b, 1, K, 3, L + 1))
    pw[:, 0, :, 1, :] = 0.1          # coherent only
    pw[:, 0, :, 2, :] = 0.3          # energy-changed
    rw = np.zeros((n_b, 1, K, L + 1))
    rw[:, 0, :, :] = 2.0
    pf = np.full((n_b, 1), 40.0)
    rf = np.full((n_b, 1), 20.0)
    _write('ap.h5', w, np.zeros_like(w))
    with h5py.File('ap.h5', 'a') as f:
        g = f['adjoint_populations']
        g['probe_energies'] = np.linspace(2.3e6, 5e6, K)
        g['probe_fission_nuclides'] = 'all'
        g['probe_n_labels'] = 3
        g['probe_weight'] = pw.ravel()
        g['probe_fission_weight'] = pf.ravel()
        g['ray_neutron_energies'] = np.geomspace(1e3, 6e6, M)
        g['ray_weight'] = rw.ravel()
        g['ray_comb_weight'] = np.zeros((n_b, 1, M, L + 1)).ravel()
        g['ray_fission_weight'] = rf.ravel()
    ap = _read('ap.h5', 1.0)
    assert ap.probe_n_labels == 3
    d, s = ap.probe_reactivity(depth=2)
    assert d[0].n == pytest.approx((2.0 + 0.1) / 80.0)
    assert s[0].n == pytest.approx(0.3 / 80.0)
    d, s = ap.probe_photoneutron_yield()
    # rays per their own fission weight, coherent probes per the probes'
    assert d[0].n == pytest.approx(2.0 / 20.0 + 0.1 / 40.0)
    assert s[0].n == pytest.approx(0.3 / 40.0)


def test_ray_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings()
    value = {'n_generation': 5, 'photoneutrons': True,
             'photoneutron_probe_energies': [2.3e6, 3e6, 5e6],
             'photoneutron_ray_neutron_energies': [1e3, 1e5, 6e6],
             'photoneutron_ray_fraction': 0.5,
             'photoneutron_ray_allocation': 'fission'}
    s.adjoint_populations = value
    s.export_to_xml()
    assert openmc.Settings.from_xml().adjoint_populations == value
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_ray_neutron_energies': [1e3]}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_ray_fraction': -1.0}
    with pytest.raises(ValueError):
        s.adjoint_populations = {'n_generation': 3,
                                 'photoneutron_ray_allocation': 'rank'}


def test_probe_importances_per_nuclide(run_in_tmpdir):
    """With rays and labels, every nuclide bin's table is linearly
    interpolable within the tolerance, including at a narrow resonance."""
    t = _resonant_target()
    K, M, L, n_b = 30, 6, 2, 4
    lines = openmc.probe_line_energies(t.threshold, 12.6e6, K)
    d, sc = _smooth_parts(t, lines)
    rng = np.random.default_rng(7)
    w = np.zeros((n_b, 5, 9, L + 1))
    w[:, CLASS_FISSION, 0] = 100.0
    pw = np.zeros((n_b, 2, K, 3, L + 1))
    rw = np.zeros((n_b, 2, K, L + 1))
    noise = rng.uniform(0.9, 1.1, (n_b, K))
    for j, scale in enumerate((1.0, 0.3)):
        rw[:, j, :, L] = 100.0 * scale * d * noise
        pw[:, j, :, 2, L] = 100.0 * scale * sc * noise
    _write('ap.h5', w, np.zeros_like(w))
    with h5py.File('ap.h5', 'a') as f:
        g = f['adjoint_populations']
        g['probe_energies'] = lines
        g['probe_fission_nuclides'] = 'U235 other'
        g['probe_n_labels'] = 3
        g['probe_weight'] = pw.ravel()
        g['probe_fission_weight'] = np.ones((n_b, 2)).ravel()
        g['ray_neutron_energies'] = np.geomspace(1e3, 6e6, M)
        g['ray_weight'] = rw.ravel()
        g['ray_comb_weight'] = np.zeros((n_b, 2, M, L + 1)).ravel()
        g['ray_fission_weight'] = np.ones((n_b, 2)).ravel()
    ap = _read('ap.h5', 1.0)
    tables = ap.probe_importances(t, depth=L, rtol=0.01)
    assert set(tables) == {'U235', 'other'}
    E = 2.431e6 + np.linspace(-3e3, 3e3, 301)
    for name, imp in tables.items():
        assert imp.interpolation_error(n_points=20, energies=E) < 0.01
        assert np.all(np.diff(imp.energy) > 0)
        assert imp.mean.shape == imp.energy.shape == imp.std_dev.shape
    # the U235 bin is 1/0.3 of the other at every energy
    x = np.linspace(lines[0], lines[-1], 50)
    assert np.allclose(tables['other'](x) * (1 / 0.3), tables['U235'](x),
                       rtol=0.03, atol=1e-12)
