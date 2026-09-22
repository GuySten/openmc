"""Tests for local perturbation worths (Branched Exact Perturbation).

Two halves, deliberately in one file because they exercise the same API:

* Module-level tests are pure Python -- construction, XML round trips, the
  results algebra, and the statepoint reader driven by synthetic HDF5. They
  need no nuclear data and run in well under a second.

* The tests after the divider run OpenMC. Every assertion there is an
  invariant that holds whatever the nuclear data says, so none of it needs a
  stored reference. They need a cross-section library and take about half a
  minute.
"""

from pathlib import Path

import h5py
import lxml.etree as ET
import numpy as np
import pytest

import openmc
from openmc.exceptions import DataError
from uncertainties import UFloat, correlated_values


@pytest.fixture(autouse=True)
def reset_perturbation_ids():
    """IDs are class-global, so keep tests from leaking into each other."""
    openmc.LocalPerturbation.used_ids.clear()
    openmc.LocalPerturbation.next_id = 1
    yield
    openmc.LocalPerturbation.used_ids.clear()
    openmc.LocalPerturbation.next_id = 1


@pytest.fixture
def cells_and_materials():
    water = openmc.Material(material_id=91)
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    steel = openmc.Material(material_id=92)
    steel.add_nuclide('Fe56', 1.0)
    steel.set_density('g/cm3', 7.9)

    inner = openmc.Sphere(r=0.5)
    outer = openmc.Sphere(r=5.0, boundary_type='vacuum')
    sample = openmc.Cell(cell_id=71, fill=water, region=-inner)
    other = openmc.Cell(cell_id=72, fill=water, region=+inner & -outer)
    return sample, other, water, steel


# ----------------------------------------------------------------------------
# LocalPerturbation
# ----------------------------------------------------------------------------

def test_substitutions_from_ids():
    p = openmc.LocalPerturbation({71: 92})
    assert p.substitutions == {71: 92}
    assert p.cells == [71]


def test_substitutions_from_objects(cells_and_materials):
    sample, _, _, steel = cells_and_materials
    p = openmc.LocalPerturbation({sample: steel})
    assert p.substitutions == {sample.id: steel.id}


def test_substitutions_mixed_ids_and_objects(cells_and_materials):
    sample, other, water, steel = cells_and_materials
    p = openmc.LocalPerturbation({sample: steel.id, other.id: water})
    assert p.substitutions == {sample.id: steel.id, other.id: water.id}


def test_substitutions_from_pairs():
    p = openmc.LocalPerturbation([(71, 92), (72, 91)])
    assert p.substitutions == {71: 92, 72: 91}


def test_void_substitution_is_material_zero():
    # A void perturbation is how the 'sample against nothing' check is set up,
    # so None has to survive as material 0 rather than raising.
    p = openmc.LocalPerturbation({71: None})
    assert p.substitutions == {71: 0}


def test_substitutions_reject_bad_types():
    with pytest.raises(TypeError):
        openmc.LocalPerturbation({'not a cell': 92})
    with pytest.raises(TypeError):
        openmc.LocalPerturbation({71: 'not a material'})


def test_ids_are_assigned_and_unique():
    a = openmc.LocalPerturbation({71: 92})
    b = openmc.LocalPerturbation({71: 91})
    assert a.id != b.id

    c = openmc.LocalPerturbation({71: 92}, perturbation_id=4321)
    assert c.id == 4321
    with pytest.warns(openmc.IDWarning):
        openmc.LocalPerturbation({71: 92}, perturbation_id=4321)


def test_name():
    p = openmc.LocalPerturbation({71: 92}, name='steel sample')
    assert p.name == 'steel sample'
    assert openmc.LocalPerturbation({71: 92}).name == ''
    with pytest.raises(TypeError):
        openmc.LocalPerturbation({71: 92}, name=3)


def test_repr_without_results_omits_worth():
    p = openmc.LocalPerturbation({71: 92}, name='steel')
    assert p.rho is None
    assert 'Worth' not in repr(p)
    p.rho, = correlated_values([-40.0], [[4.0]])
    assert 'Worth' in repr(p)
    assert p.rho.std_dev == pytest.approx(2.0)
    assert p.rho.nominal_value == pytest.approx(-40.0)


# ----------------------------------------------------------------------------
# XML round trip
# ----------------------------------------------------------------------------

def test_perturbation_xml_roundtrip():
    p = openmc.LocalPerturbation({71: 92}, perturbation_id=7, name='steel')
    q = openmc.LocalPerturbation.from_xml_element(p.to_xml_element())
    assert q.id == 7
    assert q.name == 'steel'
    assert q.substitutions == {71: 92}


def test_multi_substitution_xml_roundtrip():
    # This is the displacement shape: one sliver reverts, one takes the sample.
    p = openmc.LocalPerturbation({71: 91, 72: 92}, perturbation_id=8)
    q = openmc.LocalPerturbation.from_xml_element(p.to_xml_element())
    assert q.substitutions == {71: 91, 72: 92}


def test_bare_cell_material_shorthand_is_accepted():
    # The C++ reader accepts <cell>/<material> directly on the element as
    # shorthand for the one-cell case; the Python side must parse it too.
    elem = ET.fromstring(
        b'<local_perturbation id="3">'
        b'<cell>71</cell><material>92</material>'
        b'</local_perturbation>')
    p = openmc.LocalPerturbation.from_xml_element(elem)
    assert p.substitutions == {71: 92}


def test_collection_xml_roundtrip():
    ps = openmc.Perturbations([
        openmc.LocalPerturbation({71: 92}, perturbation_id=1, name='steel'),
        openmc.LocalPerturbation({71: 91, 72: 92}, perturbation_id=2),
    ])

    qs = openmc.Perturbations.from_xml_element(ps.to_xml_element())
    assert qs.ids == [1, 2]
    assert qs.by_id(1).name == 'steel'
    assert qs.by_id(2).substitutions == {71: 91, 72: 92}


def test_collection_file_roundtrip(run_in_tmpdir):
    ps = openmc.Perturbations([
        openmc.LocalPerturbation({71: 92}, perturbation_id=1),
    ])
    ps.export_to_xml()
    assert Path('perturbations.xml').is_file()

    qs = openmc.Perturbations.from_xml('perturbations.xml')
    assert qs.by_id(1).substitutions == {71: 92}


def test_xml_element_name_matches_cpp_reader():
    """Tag names the C++ reader depends on.

    bep::read_perturbations_xml() looks for <perturbations> with
    <local_perturbation> children. If these drift the C++ silently reads
    nothing, so pin them down.
    """
    elem = openmc.Perturbations(
        [openmc.LocalPerturbation({71: 92})]).to_xml_element()
    assert elem.tag == 'perturbations'
    # The depth is parsed from settings.xml, not from here -- settings.xml is
    # read first, so anything needing it early can see it.
    assert elem.find('n_generation') is None
    assert len(elem.findall('local_perturbation')) == 1
    sub = elem.find('local_perturbation').find('substitution')
    assert sub.find('cell').text == '71'
    assert sub.find('material').text == '92'


# ----------------------------------------------------------------------------
# Perturbations collection
# ----------------------------------------------------------------------------

def test_collection_type_checking():
    ps = openmc.Perturbations()
    with pytest.raises(TypeError):
        ps.append('not a perturbation')


def test_n_generation_lives_on_settings():
    """L is a property of the RUN, not of a perturbation.

    Every shadow tree is compared against the same reference trees at the
    same depths, so one scalar governs the whole run -- the same reasoning
    that puts superhistory_n_generation and ifp_n_generation on Settings.
    Perturbations.n_generation is output only: it says what depth a set of
    RESULTS came from.
    """
    settings = openmc.Settings()
    # The estimator is a finite difference in depth, so L < 2 is meaningless.
    with pytest.raises(ValueError):
        settings.perturbation_n_generation = 1
    with pytest.raises(TypeError):
        settings.perturbation_n_generation = 10.5
    settings.perturbation_n_generation = 2
    assert settings.perturbation_n_generation == 2

    # and it round trips through settings.xml, not perturbations.xml
    settings.perturbation_n_generation = 7
    assert openmc.Settings.from_xml_element(
        settings.to_xml_element()).perturbation_n_generation == 7

    # a fresh collection has no depth: nothing has been run yet
    assert openmc.Perturbations().n_generation is None
    with pytest.raises(AttributeError):
        openmc.Perturbations().n_generation = 8


def test_indexing_by_position_and_id():
    a = openmc.LocalPerturbation({71: 92}, perturbation_id=101)
    b = openmc.LocalPerturbation({71: 91}, perturbation_id=102)
    ps = openmc.Perturbations([a, b])
    assert ps.ids == [101, 102]
    assert ps[0] is a
    assert ps.by_id(102) is b
    assert ps[101] is a          # id lookup when it is not a valid position


# ----------------------------------------------------------------------------
# Results algebra
# ----------------------------------------------------------------------------

@pytest.fixture
def results():
    """Two strongly correlated worths, as co-located perturbations give."""
    ps = openmc.Perturbations([
        openmc.LocalPerturbation({71: 92}, perturbation_id=1),
        openmc.LocalPerturbation({71: 93}, perturbation_id=2),
    ])
    cov = np.array([[4.0, 3.96], [3.96, 4.0]])
    for p, rho in zip(ps, correlated_values([-40.0, -40.5], cov)):
        p.rho = rho
    return ps


def test_rho_is_a_correlated_ufloat(results):
    for p in results:
        assert isinstance(p.rho, UFloat)
    assert results.by_id(1).rho.nominal_value == pytest.approx(-40.0)
    assert results.by_id(1).rho.std_dev == pytest.approx(2.0)


def test_arbitrary_combination_matches_hand_algebra(results):
    a, b = results.by_id(1).rho, results.by_id(2).rho
    combined = 0.25 * a + 0.75 * b
    w = np.array([0.25, 0.75])
    rho = np.array([-40.0, -40.5])
    assert combined.nominal_value == pytest.approx(float(w @ rho))
    assert combined.std_dev == pytest.approx(
        float(np.sqrt(w @ results.covariance @ w)))


def test_difference_beats_independent_propagation(results):
    """The whole point of running perturbations together.

    Treating two co-located worths as independent gives sqrt(2)*sigma for
    their difference. Because rho carries its correlations, plain subtraction
    must do far better -- if it does not, correlated_values was handed the
    wrong matrix.
    """
    diff = results.by_id(2).rho - results.by_id(1).rho
    assert diff.nominal_value == pytest.approx(-0.5)
    assert diff.std_dev == pytest.approx(np.sqrt(2 * (4.0 - 3.96)))
    assert diff.std_dev < np.hypot(2.0, 2.0) / 5


def test_derivative_scales_by_dz(results):
    diff = results.by_id(2).rho - results.by_id(1).rho
    for dz in (0.5, -0.5):
        deriv = diff / dz
        assert deriv.nominal_value == pytest.approx(diff.nominal_value / dz)
        # A negative step must not flip the sign of the uncertainty
        assert deriv.std_dev == pytest.approx(diff.std_dev / abs(dz))


def test_covariance_round_trips(results):
    """The reported covariance must agree with what the rho arithmetic does."""
    assert np.allclose(results.covariance,
                       np.array([[4.0, 3.96], [3.96, 4.0]]))
    corr = results.correlation()
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(3.96 / 4.0)


def test_results_accessors_without_results():
    ps = openmc.Perturbations([openmc.LocalPerturbation({71: 92},
                                                        perturbation_id=1)])
    assert ps.by_id(1).rho is None
    assert ps.covariance is None
    with pytest.raises(ValueError):
        ps.correlation()
    with pytest.raises(ValueError):
        ps.depth_convergence(1)


def _levels(curve, sigma=1.0, n_batches=30, seed=7, pooled_offset=0.0):
    """A Perturbations whose per-batch levels follow ``curve`` in depth.

    The pooled level defaults to the mean of the draws, which is what a run
    with no ratio-of-means bias gives; ``pooled_offset`` moves it away in
    units of the standard error, to exercise the check that detects that.
    """
    rng = np.random.default_rng(seed)
    curve = np.asarray(curve, dtype=float)
    nd = curve.size
    draws = curve[None, :] + rng.normal(0.0, sigma, (n_batches, nd))
    level_sum = draws.sum(0)[None, :]
    level_cross = (draws * draws).sum(0)[None, None, :]
    pooled = draws.mean(0)
    if pooled_offset:
        pooled = pooled + pooled_offset * draws.std(0, ddof=1) / np.sqrt(
            n_batches)
    ps = openmc.Perturbations(
        [openmc.LocalPerturbation({71: 92}, perturbation_id=1)])
    ps._set_results(level_sum, level_cross, pooled[None, :], n_batches)
    return ps


def test_depth_convergence_is_flat_without_a_transient():
    """The one diagnostic the level estimator needs.

    A level is flat in depth once the perturbed fundamental mode has
    established itself. A sub-dominant mode shows up directly as drift, with
    no fit, no window and nothing to choose.
    """
    d = np.arange(13)
    flat = _levels(np.full(13, -300e-5), sigma=2e-5)
    curve = flat.depth_convergence(1)
    values = np.array([curve[i][0] for i in d])
    assert abs(values[1] - values[-1]) < 5.0     # pcm

    drifting = _levels(-300e-5 + 50e-5 * 0.7**d, sigma=2e-5)
    curve = drifting.depth_convergence(1)
    values = np.array([curve[i][0] for i in d])
    assert values[1] - values[-1] > 20.0


def test_confidence_interval_uses_student_t():
    """A t-interval on n_batches - 1, not a normal quantile.

    With the twenty or so batches a perturbation run has, the difference is
    several per cent and always in the direction of under-stating it.
    """
    ps = _levels(np.full(3, -300e-5), sigma=1e-5, n_batches=10)
    rho = ps.by_id(1).rho
    lo, hi = ps.confidence_interval(1)
    half = 0.5 * (hi - lo)
    assert lo < rho.nominal_value < hi
    assert half > 1.96 * rho.std_dev           # wider than the normal one
    assert half == pytest.approx(2.262 * rho.std_dev, rel=1e-3)  # t(9)


def test_pooled_agrees_with_the_batch_mean():
    """The one assumption batch statistics make here, checked directly.

    The metric is the gap between the pooled level and the mean of the
    per-batch levels, in units of the worth's own sigma. Zero when there is
    no ratio-of-means bias, and it must report the offset faithfully when
    there is one.
    """
    ps = _levels(np.full(3, -300e-5), sigma=1e-5, n_batches=40)
    assert abs(ps.pooled_vs_batch(1)) < 1e-9

    shifted = _levels(np.full(3, -300e-5), sigma=1e-5, n_batches=40,
                      pooled_offset=0.7)
    assert shifted.pooled_vs_batch(1) == pytest.approx(0.7, rel=1e-6)


def test_too_few_batches_raises():
    with pytest.raises(DataError):
        _levels(np.full(3, -300e-5), n_batches=1)


def test_level_diagnostics_need_results():
    ps = openmc.Perturbations([openmc.LocalPerturbation({71: 92},
                                                        perturbation_id=1)])
    with pytest.raises(ValueError):
        ps.depth_convergence(1)
    with pytest.raises(ValueError):
        ps.confidence_interval(1)
    with pytest.raises(ValueError):
        ps.pooled_vs_batch(1)


# ----------------------------------------------------------------------------
# Model integration
# ----------------------------------------------------------------------------

def test_model_accepts_list_and_collection():
    model = openmc.Model()
    assert isinstance(model.perturbations, openmc.Perturbations)
    assert len(model.perturbations) == 0

    p = openmc.LocalPerturbation({71: 92})
    model.perturbations = [p]
    assert list(model.perturbations) == [p]

    ps = openmc.Perturbations([p])
    model.perturbations = ps
    assert model.perturbations is ps

    with pytest.raises(TypeError):
        model.perturbations = ['not a perturbation']


def test_model_export_and_reimport(run_in_tmpdir, cells_and_materials):
    sample, other, water, steel = cells_and_materials

    model = openmc.Model()
    model.geometry = openmc.Geometry([sample, other])
    model.materials = openmc.Materials([water, steel])
    model.settings.particles = 100
    model.settings.batches = 5
    model.settings.inactive = 1
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({sample: steel}, perturbation_id=1,
                                 name='steel'),
    ])
    model.settings.perturbation_n_generation = 12

    model.export_to_xml()
    assert Path('perturbations.xml').is_file()

    reloaded = openmc.Model.from_xml()
    assert reloaded.settings.perturbation_n_generation == 12
    assert reloaded.perturbations.by_id(1).substitutions == \
        {sample.id: steel.id}
    assert reloaded.perturbations.by_id(1).name == 'steel'


def test_model_xml_single_file_roundtrip(run_in_tmpdir, cells_and_materials):
    sample, other, water, steel = cells_and_materials

    model = openmc.Model()
    model.geometry = openmc.Geometry([sample, other])
    model.materials = openmc.Materials([water, steel])
    model.settings.particles = 100
    model.settings.batches = 5
    model.settings.inactive = 1
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({sample: steel}, perturbation_id=1),
        openmc.LocalPerturbation({sample: water, other: steel},
                                 perturbation_id=2),
    ])
    model.settings.perturbation_n_generation = 12

    model.export_to_model_xml()
    root = ET.parse('model.xml').getroot()
    assert root.find('perturbations') is not None

    reloaded = openmc.Model.from_model_xml()
    assert reloaded.settings.perturbation_n_generation == 12
    assert reloaded.perturbations.ids == [1, 2]
    assert reloaded.perturbations.by_id(2).substitutions == \
        {sample.id: water.id, other.id: steel.id}


def test_no_perturbations_writes_no_file(run_in_tmpdir, cells_and_materials):
    sample, other, water, steel = cells_and_materials

    model = openmc.Model()
    model.geometry = openmc.Geometry([sample, other])
    model.materials = openmc.Materials([water, steel])
    model.settings.particles = 100
    model.settings.batches = 5
    model.settings.inactive = 1
    model.export_to_xml()
    assert not Path('perturbations.xml').exists()

    model.export_to_model_xml()
    assert ET.parse('model.xml').getroot().find('perturbations') is None


# ----------------------------------------------------------------------------
# StatePoint parsing
# ----------------------------------------------------------------------------

def _write_statepoint(path, tau, ids, n_generation, keff=1.0):
    """Minimal statepoint carrying only a local_perturbation group.

    ``tau`` is [batch][tree][depth] with five trees per perturbation, in the
    C++'s class order: D, F+, F-, L+, L-. The level and its batch statistics
    are formed here exactly as ``bep::finalize_batch`` forms them, so the
    reader is tested against an independent implementation of the same
    arithmetic rather than against itself.
    """
    tau = np.asarray(tau, dtype=float)
    n_batch, n_trees, nd = tau.shape
    assert nd == n_generation + 1
    assert n_trees == 5 * len(ids)

    def level(t):
        d_w, fp, fn, lp, ln = t
        n_f, n_l = fp - fn, lp - ln
        den = keff * d_w + n_f
        return np.where(den == 0.0, 0.0, (n_f / keff + n_l) / np.where(
            den == 0.0, 1.0, den))

    ell = np.stack(
        [np.stack([level(tau[b, 5 * i:5 * i + 5]) for i in range(len(ids))])
         for b in range(n_batch)])                     # [batch][pert][depth]
    pooled = np.stack([level(tau[:, 5 * i:5 * i + 5].sum(0))
                       for i in range(len(ids))])

    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('statepoint')
        f.attrs['version'] = [18, 0]
        f.create_dataset('run_mode', data=np.bytes_('eigenvalue'))
        f.create_dataset('k_combined', data=np.array([keff, 1.0e-5]))
        g = f.create_group('local_perturbation')
        g.create_dataset('n_generation', data=n_generation)
        g.create_dataset('n_generations_recorded', data=n_batch)
        g.create_dataset('n_batches', data=n_batch)
        g.create_dataset('n_trees', data=n_trees)
        g.create_dataset('n_tracks', data=123456)
        g.create_dataset('n_roots', data=123456)
        g.create_dataset('n_perturbations', data=len(ids))
        g.create_dataset('keff', data=keff)
        g.create_dataset('ids', data=np.asarray(ids, dtype=np.int32))
        g.create_dataset('level_sum', data=ell.sum(0).ravel())
        g.create_dataset('level_cross',
                         data=np.einsum('bid,bjd->ijd', ell, ell).ravel())
        g.create_dataset('level_pooled', data=pooled.ravel())
        g.create_dataset('tau_pooled', data=tau.sum(0).ravel())
        for i, pid in enumerate(ids):
            pg = g.create_group(f'perturbation {pid}')
            pg.create_dataset('index', data=i)
            pg.create_dataset('trees',
                              data=np.arange(5 * i, 5 * i + 5, dtype=np.int32))
            pg.create_dataset('cells', data=np.array([71], dtype=np.int32))
            pg.create_dataset('materials',
                              data=np.array([92 + i], dtype=np.int32))


def _branching_tau(rng, k, rho, n_root, n_batch, L, keff=None):
    """Simulate the five source-rooted forests for a known worth.

    The denominator D is ``n_root`` roots grown as a branching process, and
    the removal population L- is built to carry exactly the overlap that
    makes the level come out at ``rho``: with no fission-production change
    the level is ``N_L / (k D)``, so ``L- = -rho * k * D`` reproduces it at
    every depth. The point of the exercise is the reader's arithmetic and its
    statistics, not the transport.
    """
    keff = k if keff is None else keff
    D = np.zeros((n_batch, L + 1))
    for b in range(n_batch):
        alive = np.full(n_root, 1.0)
        D[b, 0] = n_root
        for d in range(1, L + 1):
            alive = rng.poisson(alive).astype(float)   # banking divides by k
            D[b, d] = alive.sum()
    zero = np.zeros_like(D)
    return D, zero, zero, zero, -rho * keff * D


def test_statepoint_parsing(run_in_tmpdir):
    """Recover a known worth from simulated source-rooted forests."""
    L, n_batch, k, rho = 10, 60, 2.2, -300e-5
    rng = np.random.default_rng(20240829)
    tau = np.stack(_branching_tau(rng, k, rho, 400, n_batch, L), axis=1)
    _write_statepoint('sp.h5', tau, [1], L, keff=k)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        ps = sp.perturbations
        assert isinstance(ps, openmc.Perturbations)
        assert ps.ids == [1]
        assert ps.n_generation == L
        assert ps.n_batches == n_batch

        got = ps.by_id(1).rho
        assert got.nominal_value == pytest.approx(1e5 * rho, rel=1e-9), \
            'the level is exact by construction here; nothing should move it'
        assert got.std_dev == pytest.approx(0.0, abs=1e-9)

        assert ps.by_id(1).substitutions == {71: 92}
        assert len(ps.by_id(1).depth_curve) == L + 1
        assert np.isfinite(ps.by_id(1).depth_curve).all()
        assert ps.by_id(1).rho_pooled == pytest.approx(1e5 * rho, rel=1e-9)


def test_level_inversion_is_exact_at_large_worth(run_in_tmpdir):
    """The closed form, against the k' it is supposed to reproduce.

    A single Newton step -- drho ~ R(1-R)/k -- agrees only to O(R^2) and is
    badly wrong for a large worth. Here the level is built from a KNOWN k',
    so anything but the exact inversion shows up immediately.
    """
    L, k = 4, 1.0
    for k_pert in (1.0003, 1.2, 1.5, 2.7):
        # N_F/k' + N_L = (1/k - 1/k') <phi'|F psi>, with D = <phi'|F psi>/k
        d_w = np.full(L + 1, 1000.0)
        n_f = np.full(L + 1, 200.0)
        drho = 1.0 / k - 1.0 / k_pert
        n_l = drho * k * d_w - n_f / k_pert
        tau = np.stack([d_w, n_f, np.zeros(L + 1), np.maximum(n_l, 0.0),
                        np.maximum(-n_l, 0.0)])[None, ...]
        tau = np.repeat(tau, 4, axis=0) / 4.0
        _write_statepoint('sp.h5', tau, [1], L, keff=k)
        with openmc.StatePoint('sp.h5', autolink=False) as sp:
            got = sp.perturbations.by_id(1).rho.nominal_value
        assert got == pytest.approx(1e5 * drho, rel=1e-9), \
            f'k\'/k = {k_pert}: got {got:.3f} pcm, expected {1e5*drho:.3f}'


def test_statepoint_survives_extinct_generations(run_in_tmpdir):
    """A batch whose whole forest dies must not poison the run.

    The level is a ratio of summed positive populations, so an extinct
    forest contributes a finite zero rather than the log(0) the slope
    estimator could hit. An all-zero batch gives level 0 by construction --
    noise, not a nan.
    """
    L, n_batch = 8, 60
    rng = np.random.default_rng(7)
    tau = np.stack(_branching_tau(rng, 2.2, -300e-5, 3, n_batch, L), axis=1)
    tau[::7] = 0.0

    _write_statepoint('sp.h5', tau, [1], L, keff=2.2)
    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        ps = sp.perturbations
        got = ps.by_id(1).rho
        assert np.isfinite(got.nominal_value)
        assert np.isfinite(got.std_dev)
        assert np.isfinite(ps.covariance).all()


def test_statepoint_covariance_is_symmetric(run_in_tmpdir):
    L, n_batch = 8, 60
    rng = np.random.default_rng(99)
    t1 = _branching_tau(rng, 2.2, -300e-5, 40, n_batch, L)
    t2 = _branching_tau(rng, 2.2, -150e-5, 40, n_batch, L)
    tau = np.stack(list(t1) + list(t2), axis=1)
    # Give the second one noise of its own, or both levels are exact and the
    # covariance is identically zero.
    tau[:, 9, :] *= rng.normal(1.0, 0.05, (n_batch, L + 1))
    tau[:, 4, :] *= rng.normal(1.0, 0.05, (n_batch, L + 1))
    _write_statepoint('sp.h5', tau, [1, 2], L, keff=2.2)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        ps = sp.perturbations
        cov = ps.covariance
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)
        assert (np.diag(cov) > 0).all()
        assert np.all(np.abs(ps.correlation()) <= 1.0 + 1e-9)
        diff = ps.by_id(2).rho - ps.by_id(1).rho
        assert np.isfinite([diff.nominal_value, diff.std_dev]).all()


def test_statepoint_absent_group_returns_none(run_in_tmpdir):
    with h5py.File('sp.h5', 'w') as f:
        f.attrs['filetype'] = np.bytes_('statepoint')
        f.attrs['version'] = [18, 0]
    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        assert sp.perturbations is None


def test_statepoint_result_is_cached(run_in_tmpdir):
    L, n_batch = 8, 60
    rng = np.random.default_rng(1)
    tau = np.stack(_branching_tau(rng, 2.2, -100e-5, 30, n_batch, L), axis=1)
    _write_statepoint('sp.h5', tau, [1], L, keff=2.2)
    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        assert sp.perturbations is sp.perturbations


def test_amplification_sees_a_cancellation_the_level_cannot(run_in_tmpdir):
    """Padding L+ and L- equally leaves the level alone and is the whole risk.

    The level is a difference of populations. Adding the same amount to both
    removal populations changes nothing about the answer and everything about
    how hard the answer is to get: a transient that is a fixed fraction of
    the populations is a hundred times more of the answer once the
    populations are a hundred times the answer. The depth curve is blind to
    this -- it is identical in both runs below -- so the diagnostic that
    decides whether a depth is enough has to measure it separately.
    """
    L, n_batch, k, rho = 8, 40, 2.2, -300e-5
    rng = np.random.default_rng(7)
    D, zero, _, lp, ln = _branching_tau(rng, k, rho, 300, n_batch, L)

    # Pad both removal populations by 49x the surviving difference, so the
    # cancellation is 100:1 instead of 2:1 while N_L = L+ - L- is untouched.
    pad = 49.0 * abs(rho) * k * D
    padded = np.stack([D, zero, zero, lp + pad, ln + pad], axis=1)
    plain = np.stack([D, zero, zero, lp, ln], axis=1)

    _write_statepoint('plain.h5', plain, [1], L, keff=k)
    _write_statepoint('padded.h5', padded, [1], L, keff=k)

    with openmc.StatePoint('plain.h5', autolink=False) as sp:
        a_plain = sp.perturbations.amplification(1)
        curve_plain = np.asarray(sp.perturbations.by_id(1).depth_curve)
    with openmc.StatePoint('padded.h5', autolink=False) as sp:
        a_padded = sp.perturbations.amplification(1)
        curve_padded = np.asarray(sp.perturbations.by_id(1).depth_curve)

    assert curve_padded == pytest.approx(curve_plain, rel=1e-12), \
        'the padding must not move the answer, or the test proves nothing'
    assert a_plain[1:] == pytest.approx(2.0, rel=1e-9)
    assert a_padded[1:] == pytest.approx(100.0, rel=1e-9)


def test_convergence_recovers_a_planted_transient(run_in_tmpdir):
    """A + B r**d, planted in the removal population and read back.

    ``convergence`` exists because 'the curve looks flat' and 'the drift is
    inside the error bars' both passed a level that was 6.5% wrong. Here the
    asymptote, the decay rate and what is left to go at L are all known, so
    the fit can be held to them.
    """
    L, n_batch, k = 24, 40, 2.2
    A, B, r = -300e-5, -900e-5, 0.8         # level runs -1200 pcm -> -300 pcm
    rng = np.random.default_rng(11)
    D, zero, *_ = _branching_tau(rng, k, 0.0, 400, n_batch, L)

    d = np.arange(L + 1, dtype=float)
    ell = A + B * r ** d                     # the level wanted at each depth
    ln = -ell * k * D                        # N_L = -L- makes level = ell
    tau = np.stack([D, zero, zero, np.zeros_like(D), ln], axis=1)
    _write_statepoint('sp.h5', tau, [1], L, keff=k)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        c = sp.perturbations.convergence(1, target=0.05)

    assert c['rate'] == pytest.approx(r, abs=2e-3)
    assert c['asymptote'] == pytest.approx(1.0e5 * A, rel=1e-3)
    assert c['remaining'] == pytest.approx(1.0e5 * B * r ** L, rel=5e-2)

    # Two roots below the answer at L=24: converged on the drift alone.
    assert abs(c['remaining']) < 0.05 * abs(c['asymptote'])
    # ...but the required depth is set by the cancellation, not by the drift,
    # and at amplification 2 with target 0.05 that is ln(40)/ln(1.25) = 16.5.
    assert c['amplification'] == pytest.approx(2.0, rel=1e-6)
    assert c['required_depth'] == pytest.approx(
        np.log(2.0 / 0.05) / np.log(1.0 / c['rate']), rel=1e-9)
    assert c['converged'] is True

    # The same curve read at a depth short of that is not converged, even
    # though its drift there is still small against its own error bars.
    short = 10
    _write_statepoint('short.h5', tau[:, :, :short + 1], [1], short, keff=k)
    with openmc.StatePoint('short.h5', autolink=False) as sp:
        assert sp.perturbations.convergence(1, target=0.05)['converged'] \
            is False


def test_convergence_weights_the_fit_by_the_depth_errors(run_in_tmpdir):
    """The deep end of a depth curve is the noisy end, and must not lead.

    A depth-d population is a d-long branching chain, so its error grows with
    depth -- measured 3.17 pcm at d=1 against 10.02 at d=30 on one run, and
    16 against 234 on another. An unweighted fit lets exactly the least
    trustworthy points set the asymptote.

    Here the curve is exact and precise out to d=10 and then wanders 60 pcm
    away with a hundredfold larger error. The weighted fit must stay with the
    precise points; an unweighted one is dragged most of the way to the
    wandering ones.
    """
    L, n_batch, k = 20, 60, 2.2
    A, B, r = -300e-5, -900e-5, 0.75
    rng = np.random.default_rng(31)
    D, zero, *_ = _branching_tau(rng, k, 0.0, 400, n_batch, L)

    d = np.arange(L + 1, dtype=float)
    ell = A + B * r ** d
    ell[11:] += 60e-5                       # the deep end wanders off
    # Per-batch scatter: tiny up to d=10, large past it, so depth_sigma
    # carries the difference the fit is supposed to respect.
    scale = np.where(d <= 10, 0.02e-5, 20e-5)
    eps = rng.normal(0.0, 1.0, size=(n_batch, L + 1)) * scale
    eps -= eps.mean(axis=0)                 # keep the planted mean exact
    ln = -(ell + eps) * k * D
    tau = np.stack([D, zero, zero, np.zeros_like(D), ln], axis=1)
    _write_statepoint('w.h5', tau, [1], L, keff=k)

    with openmc.StatePoint('w.h5', autolink=False) as sp:
        P = sp.perturbations
        sig = np.asarray(P.by_id(1).depth_sigma)
        assert sig[20] > 100 * sig[5], 'the deep end must really be noisier'
        c = P.convergence(1, target=0.05)

    # The weighted fit stays with the precise points.
    assert c['asymptote'] == pytest.approx(1.0e5 * A, abs=3.0)
    assert c['rate'] == pytest.approx(r, abs=0.05)

    # An unweighted fit of the same curve is dragged far off, which is what
    # this weighting exists to prevent.
    y = np.asarray(P.by_id(1).depth_curve)[1:]
    x = np.arange(1, L + 1, dtype=float)
    best = min(
        (float(((y - np.vstack([np.ones_like(x), rr ** x]).T @ np.linalg.lstsq(
            np.vstack([np.ones_like(x), rr ** x]).T, y, rcond=None)[0]) ** 2
            ).sum()),
         np.linalg.lstsq(np.vstack([np.ones_like(x), rr ** x]).T, y,
                         rcond=None)[0][0])
        for rr in np.linspace(0.05, 0.999, 400))
    assert abs(best[1] - 1.0e5 * A) > 20.0, \
        'if the unweighted fit were fine here the test proves nothing'


def test_convergence_reports_a_railed_fit_instead_of_inventing_a_rate(
        run_in_tmpdir):
    """A flat curve has no decay in it, and the fit must say so.

    Fitting ``A + B r**d`` to a curve that never turns over drives ``r`` to
    the slowest decay on the grid, and the fit then reports a rate near one
    and a required depth in the thousands. Both are artifacts of the grid
    edge. Read literally they say 'run deeper', which is exactly the wrong
    advice for a level that is barely moving because it is wrong.

    The curve here is the measured water->B10 one idealised: -285 pcm
    creeping by a quarter of a pcm per generation, with no turnover
    anywhere. An exactly flat curve is a different case and not this one --
    it is fitted by ``B = 0`` at any rate, and it is genuinely converged.
    """
    L, n_batch, k = 20, 40, 2.2
    rng = np.random.default_rng(5)
    D, zero, *_ = _branching_tau(rng, k, 0.0, 400, n_batch, L)

    d = np.arange(L + 1, dtype=float)
    creep = -285e-5 - 0.25e-5 * d
    tau = np.stack([D, zero, zero, np.zeros_like(D), -creep * k * D], axis=1)
    _write_statepoint('creep.h5', tau, [1], L, keff=k)

    with openmc.StatePoint('creep.h5', autolink=False) as sp:
        c = sp.perturbations.convergence(1, target=0.05)

    assert c['railed'] is True
    assert np.isnan(c['rate']), 'a railed fit has no rate to report'
    assert np.isnan(c['remaining'])
    assert c['required_depth'] == np.inf
    assert c['converged'] is False
    assert c['asymptote'] == pytest.approx(-290.0, rel=1e-9), \
        'with no decay visible the best statement is the level at L itself'

    # The exactly flat curve, for contrast: nothing left to go, and the fit
    # must not call that a rail.
    flat = np.full(L + 1, -285e-5)
    tau = np.stack([D, zero, zero, np.zeros_like(D), -flat * k * D], axis=1)
    _write_statepoint('flat.h5', tau, [1], L, keff=k)
    with openmc.StatePoint('flat.h5', autolink=False) as sp:
        c = sp.perturbations.convergence(1, target=0.05)
    assert c['railed'] is False
    assert c['remaining'] == pytest.approx(0.0, abs=1e-9)
    assert c['asymptote'] == pytest.approx(-285.0, rel=1e-9)


def test_convergence_needs_the_populations(run_in_tmpdir):
    """Without tau_pooled the amplification is unknowable -- say so."""
    L, n_batch = 8, 40
    rng = np.random.default_rng(3)
    tau = np.stack(_branching_tau(rng, 2.2, -100e-5, 30, n_batch, L), axis=1)
    _write_statepoint('sp.h5', tau, [1], L, keff=2.2)
    with h5py.File('sp.h5', 'a') as f:
        del f['local_perturbation']['tau_pooled']

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        with pytest.raises(ValueError, match='population totals'):
            sp.perturbations.amplification(1)


# ----------------------------------------------------------------------------
# Tests that run OpenMC
#
# Self-asserting: every check below is an invariant that holds whatever the
# nuclear data says, so none of them needs a stored reference.
# ----------------------------------------------------------------------------

@pytest.fixture
def model():
    """Reflected fuel sphere with a small off-centre sample cavity.

    The boundary is reflective on purpose. A leaky system makes the shadow
    trees decay as k**d, so by depth L there is almost no descendant weight
    left and every assertion below drowns in noise. The sample sits off
    centre and away from the source so that only a small fraction of
    histories branch, which is what keeps the shadow pass affordable.
    """
    fuel = openmc.Material(material_id=1)
    fuel.add_nuclide('U235', 1.0)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.0)

    water = openmc.Material(material_id=2)
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    absorber = openmc.Material(material_id=3)
    absorber.add_nuclide('B10', 1.0)
    absorber.set_density('g/cm3', 2.5)

    # The sample is deliberately large. Precision on a worth scales as
    # 1/sqrt(branch sites), i.e. with the sample's surface area, while the
    # worth itself grows faster than that, so a pinhead sample needs an
    # impractical number of histories before its sign is even resolved. A
    # 1.5 cm sample makes these assertions decidable in seconds.
    sample_surf = openmc.Sphere(x0=4.0, r=2.5)
    outer = openmc.Sphere(r=10.0, boundary_type='reflective')

    sample = openmc.Cell(cell_id=10, fill=water, region=-sample_surf)
    bulk = openmc.Cell(cell_id=11, fill=fuel, region=+sample_surf & -outer)

    model = openmc.Model()
    model.geometry = openmc.Geometry([sample, bulk])
    model.materials = openmc.Materials([fuel, water, absorber])
    # A shadow tree is a branching process and can go extinct: at k ~ 2 a
    # single tree dies about 16% of the time. Generations where EVERY tree of
    # one kind dies are dropped, which biases the worth, so the particle count
    # has to be high enough that many branch sites are recorded each
    # generation. Check n_skipped in the statepoint if results look off.
    model.settings.particles = 5000
    model.settings.batches = 50
    model.settings.inactive = 10
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())
    model.settings.seed = 1
    return model


def _sample_cell(model):
    return model.geometry.get_all_cells()[10]


def _water_and_absorber(model):
    mats = {m.id: m for m in model.materials}
    return mats[2], mats[3]     # water, absorber


def test_null_perturbation_is_exactly_zero(run_in_tmpdir, model):
    """Substituting the material already in place must give exactly zero.

    Common random numbers plus branching at the entry point make the two
    shadow trees the same tree, history for history, so the only thing that
    can separate them is floating-point rounding: score_site() accumulates
    with an atomic add, and the interleaving across threads differs between
    the reference and perturbed slots. The tolerance below is that rounding
    and nothing else -- it is some eight orders of magnitude tighter than
    any real worth, so a genuine break in the branch pairing or the seeding
    cannot hide under it.

    This is the test to run first. Every stream of the shadow root has to
    be seeded from the shared branch id; seeding only STREAM_TRACKING
    leaves STREAM_URR_PTABLE picking up stack garbage, which decorrelates
    the trees wherever a nuclide has unresolved resonances and turns this
    into noise.
    """
    water, _ = _water_and_absorber(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): water},
                                 perturbation_id=1, name='null'),
    ])
    model.settings.perturbation_n_generation = 6

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        p = sp.perturbations.by_id(1)
        # rho is an uncertainties value; compare its parts, not the object.
        # Ordering and abs() on AffineScalarFunc are deprecated, and numpy
        # ufuncs reject it outright.
        assert abs(p.rho.nominal_value) < 1.0e-6, \
            f'null perturbation gave {p.rho} pcm; common random ' \
            'numbers are not holding between the trees'
        assert p.rho.std_dev < 1.0e-6
        assert np.allclose(p.depth_curve, 0.0, atol=1.0e-12)


def test_survival_biasing_preserves_the_null_perturbation(run_in_tmpdir,
                                                         model):
    """Survival biasing must not separate a tree from its reference.

    Shadow trees are not exempt from survival biasing: neither absorption()
    nor apply_russian_roulette() is gated on super_gen or bep_tree, so
    whatever the driver does, they do. That makes the roulette's SCALE a
    shadow-tree concern -- it measures a particle against weight_cutoff *
    wgt_born, and run_one_tree() sets wgt_born to the weight a typical
    particle in the tree is born at rather than the tree's root weight.

    Whatever that scale is, it has to be the SAME for a perturbation's tree
    and for the reference tree it is scored against, or the common random
    numbers that make a null perturbation exactly zero stop holding and
    every worth picks up the difference as noise. A null perturbation with
    survival biasing on is the sharpest available check that they have not
    drifted apart.
    """
    water, _ = _water_and_absorber(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): water},
                                 perturbation_id=1, name='null'),
    ])
    model.settings.perturbation_n_generation = 6
    model.settings.survival_biasing = True

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        p = sp.perturbations.by_id(1)
        assert abs(p.rho.nominal_value) < 1.0e-6, (
            f'null perturbation gave {p.rho} pcm under survival biasing; the '
            'roulette is treating the perturbed tree differently from its '
            'reference')
        assert p.rho.std_dev < 1.0e-6
        assert np.allclose(p.depth_curve, 0.0, atol=1.0e-12)


def test_survival_biasing_does_not_leak_into_the_driver(run_in_tmpdir,
                                                       model):
    """The shadow-tree roulette override must stop at the trunk.

    apply_russian_roulette() forces the NORMALIZED branch for any particle in
    a shadow tree, whatever survival_normalization is set to, because the
    absolute branch would roulette a perturbation's whole population away.
    The override is keyed on bep_tree() != BEP_TRUNK, so a driver particle
    must never take it: with survival biasing on and normalization off, the
    driver has to get exactly the absolute form it asked for, and adding
    perturbations to a model must not change its fission source.

    Bit-exact on the source for the reason test_driver_is_unperturbed gives;
    k only to rounding, since its accumulators are order-dependent.
    """
    model.settings.survival_biasing = True
    model.settings.survival_normalization = False
    last_batch = model.settings.batches
    model.settings.sourcepoint = {
        'batches': [last_batch], 'separate': False, 'write': True}

    sp_reference = model.run(cwd='reference')
    with openmc.StatePoint(sp_reference) as sp:
        k_reference = sp.k_generation[:]
        source_reference = sp.source if sp.source_present else None

    _, absorber = _water_and_absorber(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ])
    model.settings.perturbation_n_generation = 6

    sp_perturbed = model.run(cwd='perturbed')
    with openmc.StatePoint(sp_perturbed) as sp:
        k_perturbed = sp.k_generation[:]
        source_perturbed = sp.source if sp.source_present else None

    assert source_reference is not None
    for field in ('r', 'u', 'E', 'wgt'):
        assert np.array_equal(source_reference[field],
                              source_perturbed[field]), (
            f'the fission source moved in {field}: the shadow-tree roulette '
            'override is reaching driver particles')
    assert np.allclose(k_reference, k_perturbed, rtol=1.0e-12)


def test_driver_is_unperturbed(run_in_tmpdir, model):
    """The driver must be an ordinary eigenvalue calculation.

    The invariant that actually matters, and the one that is bit-exact, is
    the FISSION SOURCE: sort_bank() orders the bank by parent and progeny
    id precisely so it does not depend on thread scheduling. If a shadow
    particle reaches the real fission bank, or clobbers a per-source array
    such as progeny_per_particle, the source moves and this catches it.

    k is checked only to rounding. The k tallies are accumulated with
    `#pragma omp atomic` float adds in event_death(), and floating-point
    addition is not associative, so their last bits depend on the order the
    threads finish. BEP shifts that timing simply by doing extra work,
    which means k is NOT bit-reproducible even when the driver is
    untouched. The tolerance below is nine orders of magnitude tighter than
    the contamination it is meant to catch -- a shadow particle
    contributing to k-eff moves it by O(1/N), around 1e-3 here, not 1e-12.
    """
    # Make sure the final source bank lands in the statepoint
    last_batch = model.settings.batches
    model.settings.sourcepoint = {
        'batches': [last_batch], 'separate': False, 'write': True}

    sp_reference = model.run(cwd='reference')
    with openmc.StatePoint(sp_reference) as sp:
        k_reference = sp.k_generation[:]
        source_reference = sp.source if sp.source_present else None
        assert sp.perturbations is None

    water, absorber = _water_and_absorber(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
        openmc.LocalPerturbation({_sample_cell(model): water},
                                 perturbation_id=2),
    ])
    model.settings.perturbation_n_generation = 6

    sp_perturbed = model.run(cwd='perturbed')
    with openmc.StatePoint(sp_perturbed) as sp:
        k_perturbed = sp.k_generation[:]
        source_perturbed = sp.source if sp.source_present else None
        assert sp.perturbations is not None

    assert source_reference is not None, \
        'no source bank in the statepoint; cannot check the real invariant'
    for field in ('r', 'u', 'E', 'wgt'):
        assert np.array_equal(source_reference[field],
                              source_perturbed[field]), \
            f'BEP moved the fission source: {field} differs'

    assert len(k_reference) == len(k_perturbed)
    assert np.allclose(k_reference, k_perturbed, rtol=1e-12, atol=0.0), \
        'BEP changed k beyond rounding: a shadow particle reaches k-eff'


def test_absorber_worth_is_negative(run_in_tmpdir, model):
    """Sign convention, and that the estimator produces a real number.

    Replacing water with B10 in a central cavity must be worth less than
    nothing. Loose tolerance: this pins the sign, not the value.
    """
    _, absorber = _water_and_absorber(model)
    # Precision on a worth scales as 1/sqrt(branch sites). The fixture's
    # sample is already large; this buys the rest of the margin needed to
    # resolve the sign well clear of 3 sigma.
    model.settings.particles = 10000
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1, name='B10'),
    ])
    model.settings.perturbation_n_generation = 8

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        p = sp.perturbations.by_id(1)
        assert np.isfinite(p.rho.nominal_value), \
            'non-finite worth: a shadow tree went extinct and log1p(-1) ' \
            'leaked into the accumulator'
        assert np.isfinite(p.rho.std_dev)
        assert p.rho.std_dev > 0.0
        assert p.rho.nominal_value < -3.0 * p.rho.std_dev, (
            f'B10 sample worth {p.rho:.0f} pcm is not '
            'resolvably negative. If the sign is right but the error '
            'bar is too large, this is statistics, not correctness: '
            'precision scales as 1/sqrt(branch sites), so raise particles '
            'or enlarge the sample rather than loosening the assertion.')


def test_depth_curve_is_flat_not_linear(run_in_tmpdir, model):
    """The estimator is a LEVEL, so its depth curve must plateau.

    This is the inverse of what the slope estimator needed. Depth 0 is the
    source counted with no propagation at all -- uniform weighting, which is
    not an importance and is wildly wrong -- and every depth after it is the
    same ratio, flat once the perturbed fundamental mode has established
    itself. Drift at the deep end means n_generation is too small; that is
    the whole convergence diagnostic, and there is no fit window to pick.
    """
    _, absorber = _water_and_absorber(model)
    # Shadow trees cost ~k**d histories each, so trade particles for depth
    model.settings.particles = 2000
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ])
    model.settings.perturbation_n_generation = 10

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        p = ps.by_id(1)
        curve, sigma = p.depth_curve, p.depth_sigma
        assert len(curve) == 11
        assert ps.depth_convergence(1)[10] == (curve[10], sigma[10])

        # Depth 0 is the un-propagated source: it must NOT look like the
        # answer, or the propagation is not happening at all.
        assert abs(curve[0] - curve[10]) > 3.0 * sigma[10]

        # Every propagated depth is the same level. Compared against the
        # deepest one's own uncertainty, since they are strongly correlated
        # and it is drift, not scatter, that matters.
        for d in range(2, 11):
            assert abs(curve[d] - curve[10]) < 3.0 * sigma[10], (
                f'level at depth {d} is {curve[d]:.1f} pcm against '
                f'{curve[10]:.1f} +/- {sigma[10]:.1f} at depth 10; a level '
                'that still drifts has not converged')


def test_covariance_is_symmetric_and_correlated(run_in_tmpdir,
                                                model):
    """Perturbations sharing a source must come out correlated.

    Two substitutions into the same cell see the same driver segments and,
    for every nuclide they both change, emit the same source particle from
    the same seed -- differing only in its weight. Their estimator noise is
    then largely common and subtracting them must beat treating them as
    independent. A seed keyed on the perturbation index would destroy that,
    and would show up here.

    The pair that has to be strongly correlated is two absorbers at nearly
    the same density: they change the same three nuclides by nearly the same
    amounts, so almost all of their source coincides. Absorber against void
    is NOT such a pair -- they share only the water's removal, while the
    absorber's worth is dominated by boron the void does not have -- and
    asserting a positive correlation there is a coin flip on a
    finite-batch estimate.
    """
    water, absorber = _water_and_absorber(model)
    lighter = openmc.Material(material_id=4)
    lighter.add_nuclide('B10', 1.0)
    lighter.set_density('g/cm3', 2.4)
    model.materials.append(lighter)

    cell = _sample_cell(model)
    # No null perturbation here. A null has exactly zero worth AND exactly
    # zero variance, so its row and column of the covariance are zero, and
    # its correlation with itself is 0/0 -- correct, but it makes a
    # correlation-matrix assertion meaningless.
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: None}, perturbation_id=1),
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=2),
        openmc.LocalPerturbation({cell: lighter}, perturbation_id=3),
    ])
    model.settings.perturbation_n_generation = 6

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        cov = ps.covariance
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)
        assert (np.diag(cov) > 0.0).all(), \
            'a perturbation has zero variance; is one of them a null?'

        corr = ps.correlation()
        assert np.allclose(np.diag(corr), 1.0)
        assert np.all(np.abs(corr) <= 1.0 + 1e-9)

        # No absolute threshold on a raw correlation: it is estimated from
        # n_batches realizations and carries about 1/sqrt(n) of its own
        # noise. The check below tests the same property and is robust,
        # because it compares two numbers from the same covariance rather
        # than one against a bar.
        diff = ps.by_id(3).rho - ps.by_id(2).rho
        independent = np.hypot(ps.by_id(2).rho.std_dev,
                               ps.by_id(3).rho.std_dev)
        # The bound is 0.85, and it is measured rather than chosen. If the
        # two perturbations shared nothing this ratio would be 1.0 by
        # construction, so the test still catches a broken common source --
        # but the ratio is itself a noisy statistic and 0.5 was far tighter
        # than the quantity supports. Over eight seeds
        # (tools/bep_checks/sharecheck.py) it measures
        #
        #     0.600 +/- 0.049, range 0.354 to 0.750
        #
        # so a 0.5 bar fails about seven times in eight. It had been passing
        # on this fixture's particular seed by luck, and the seed-reuse fix
        # -- which changes every random stream in the feature -- moved it to
        # a different draw. That the fix was NOT the cause is the point of
        # the measurement: the same eight seeds on the pre-fix code give
        # 0.600 +/- 0.049 and on the fixed code 0.582 +/- 0.045, identical
        # within error.
        assert diff.std_dev < 0.85 * independent, (
            f'two absorbers 4% apart in density gave a difference of '
            f'{diff:.1f} pcm against {independent:.1f} for independent '
            'worths; their shared source is not being sampled in common')


def test_displacement_across_a_symmetry_plane_is_zero(run_in_tmpdir):
    """A multi-cell substitution with an exactly known answer.

    A displacement is a SET of substitutions applied together: the trailing
    sliver reverting to what was there and the leading sliver taking the
    sample. Displacing the sample across a plane of symmetry moves it
    between two positions of identical importance, so the worth is exactly
    zero -- not by construction the way a null substitution is (this one
    really does emit a source, a negative population in one slice and a
    positive one in the other), but as a property of the geometry. That
    makes it the sharpest available check that the two halves of a two-cell
    substitution are weighted correctly against each other.

    The earlier version of this test compared the displacement against the
    difference of two whole-sample perturbations. That comparison was not
    well posed: the reference geometry held water in every slice, so the
    ``trailing sliver reverts to water`` half of the displacement was a null
    substitution and the displacement was just the leading half. The two
    routes were never estimating the same thing.
    """
    fuel = openmc.Material(material_id=1)
    fuel.add_nuclide('U235', 1.0)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.0)

    water = openmc.Material(material_id=2)
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    absorber = openmc.Material(material_id=3)
    absorber.add_nuclide('B10', 1.0)
    absorber.set_density('g/cm3', 2.5)

    # Axially sliced channel through a reflected fuel sphere, with the
    # slice boundaries placed symmetrically about z = 0 so that slices 1 and
    # 2 are mirror images. Reflective for the same reason as the fixture
    # above: a leaky system starves the shadow trees.
    outer = openmc.Sphere(r=10.0, boundary_type='reflective')
    channel = openmc.ZCylinder(x0=4.0, r=1.2)
    planes = [openmc.ZPlane(z0=z) for z in np.linspace(-3.0, 3.0, 5)]

    slices = []
    for i in range(len(planes) - 1):
        slices.append(openmc.Cell(
            cell_id=100 + i, fill=water,
            region=-channel & +planes[i] & -planes[i + 1]))
    # The reference geometry HOLDS the sample, at slice 1; the displacement
    # moves it to slice 2, its mirror image in z.
    slices[1].fill = absorber

    bulk_region = -outer & ~openmc.Union([c.region for c in slices])
    bulk = openmc.Cell(cell_id=200, fill=fuel, region=bulk_region)

    model = openmc.Model()
    model.geometry = openmc.Geometry(slices + [bulk])
    model.materials = openmc.Materials([fuel, water, absorber])
    model.settings.particles = 5000
    model.settings.batches = 80
    model.settings.inactive = 10
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())
    model.settings.seed = 1

    ids = [c.id for c in slices]
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({ids[1]: water, ids[2]: absorber},
                                 perturbation_id=1, name='moved 1->2'),
    ])
    model.settings.perturbation_n_generation = 8

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        rho = sp.perturbations.by_id(1).rho
        assert rho.std_dev > 0.0, 'a displacement does emit a source'
        assert abs(rho.nominal_value) < 3.0 * rho.std_dev, (
            f'displacing the sample across a plane of symmetry gave '
            f'{rho:.3f} pcm; it must be zero')


def test_multigroup_is_refused_clearly(run_in_tmpdir):
    """Multigroup is not supported yet, and must say so rather than guess.

    The perturbation source needs the group-to-group transfer matrices of
    both the reference and the substituted material to form dSigma_s, and
    those are not wired up. An absorption-only source would run and would be
    quietly wrong for any substitution that changes scattering, which is
    almost all of them -- so the run stops instead.
    """
    groups = openmc.mgxs.EnergyGroups([0.0, 1.0e5, 20.0e6])

    def xsdata(name, absorption, scatter, fissile):
        x = openmc.XSdata(name, groups)
        x.order = 0
        x.set_total([1.0, 2.0])
        x.set_absorption(absorption)
        x.set_scatter_matrix(scatter)
        fission = list(np.array(absorption) * 0.4) if fissile else [0.0, 0.0]
        x.set_fission(fission)
        x.set_nu_fission(absorption if fissile else [0.0, 0.0])
        x.set_chi([1.0, 0.0])
        return x

    fuel_scatter = np.array([[[0.60], [0.38]], [[0.00], [1.80]]])
    abs_scatter = np.array([[[0.57], [0.38]], [[0.00], [1.00]]])

    library = openmc.MGXSLibrary(groups)
    library.add_xsdatas([
        xsdata('fuel', [0.02, 0.20], fuel_scatter, True),
        xsdata('sample', [0.02, 0.20], fuel_scatter, True),
        xsdata('absorber', [0.05, 1.00], abs_scatter, False),
    ])
    library.export_to_hdf5('mgxs.h5')

    def macro(name):
        m = openmc.Material(name=name)
        m.set_density('macro', 1.0)
        m.add_macroscopic(name)
        return m

    fuel, sample, absorber = macro('fuel'), macro('sample'), macro('absorber')
    inner = openmc.Sphere(r=2.0)
    outer = openmc.Sphere(r=10.0, boundary_type='reflective')

    model = openmc.Model()
    model.geometry = openmc.Geometry([
        openmc.Cell(cell_id=10, fill=sample, region=-inner),
        openmc.Cell(cell_id=11, fill=fuel, region=+inner & -outer),
    ])
    model.materials = openmc.Materials([fuel, sample, absorber])
    model.materials.cross_sections = 'mgxs.h5'
    model.settings.energy_mode = 'multi-group'
    model.settings.particles = 200
    model.settings.batches = 15
    model.settings.inactive = 5
    model.settings.seed = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    cell = model.geometry.get_all_cells()[10]
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
    ])
    model.settings.perturbation_n_generation = 4

    with pytest.raises(RuntimeError, match='continuous-energy'):
        model.run()


def test_site_splitting_tunes_the_denominator_without_moving_the_worth(
        run_in_tmpdir, model):
    """The denominator is a tunable now, and it must still be unbiased.

    This test used to assert the opposite -- that the rule leaves the
    denominator at exactly 1.0 and its populations come out BIT-IDENTICAL
    with splitting on and off. That was right for the rule as it stood and
    is wrong for the rule as derived. Section 4b of ``bep_autotune.md``
    shows the denominator enters ``relvar(rho)`` with ``G_D = 1`` like any
    other population, and that ``N_D`` sits far above its own optimum when
    left untuned, so it belongs in the optimisation. The old assertion
    encoded a restriction, not an invariant.

    What survives is what the old test was really protecting, and it is
    checked here directly rather than as a side effect:

    * **unbiasedness** -- splitting changes how finely the populations are
      banked, never what they are worth, since ``E[N] = nu`` exactly at any
      site weight. So the two worths must agree statistically even though
      the populations no longer agree at all.
    * **reproducibility** -- two runs of one configuration must be
      bit-identical. That is the part that earned its keep: it caught the
      denominator roots being keyed on their index in the fission bank,
      which ``thread_safe_append()`` fills in whatever order the threads
      finish, so two runs of the same seed differed by over a per cent.
    """
    _, absorber = _water_and_absorber(model)
    model.settings.particles = 2000
    model.settings.perturbation_n_generation = 6
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ])

    def run_once(splitting):
        model.settings.perturbation_site_splitting = splitting
        path = model.run()
        with h5py.File(path, 'r') as f:
            g = f['local_perturbation']
            nd = int(g['n_generation'][()]) + 1
            tau = np.array(g['tau_pooled'][()]).reshape(
                int(g['n_trees'][()]), nd)
            trees = np.array(g['perturbation 1']['trees'][()])
            d = tau[trees[0]]
        with openmc.StatePoint(path) as sp:
            rho = sp.perturbations.by_id(1).rho
        return d, rho

    d_on, rho_on = run_once(True)     # the default: every population tuned
    d_on2, rho_on2 = run_once(True)   # same configuration, again
    d_off, rho_off = run_once(False)  # unit-weight sites throughout

    # Reproducibility, bit for bit.
    assert np.array_equal(d_on, d_on2), \
        'two runs of one configuration disagree; the shadow pass is not ' \
        'reproducible'
    assert rho_on.nominal_value == rho_on2.nominal_value

    # With splitting off every site weight is 1, so the denominator is the
    # raw banked population and the tuned run must differ from it -- that is
    # the whole point of making it a tunable.
    assert not np.array_equal(d_on, d_off), \
        'the denominator population is identical with splitting on and ' \
        'off, so the rule is still refusing to tune it'

    # ...but the answer must not move. Unbiased at any site weight.
    sigma = float(np.hypot(rho_on.std_dev, rho_off.std_dev))
    assert sigma > 0.0
    assert abs(rho_on.nominal_value - rho_off.nominal_value) < 4.0 * sigma, (
        f'site splitting moved the worth: {rho_on} with splitting against '
        f'{rho_off} without, which is more than 4 sigma apart')


def test_site_splitting_xml_roundtrip():
    s = openmc.Settings()
    assert s.perturbation_site_splitting is None

    s.perturbation_site_splitting = False
    elem = s.to_xml_element()
    assert elem.find('perturbation_site_splitting').text == 'false'
    assert openmc.Settings.from_xml_element(
        elem).perturbation_site_splitting is False

    s.perturbation_site_splitting = True
    assert openmc.Settings.from_xml_element(
        s.to_xml_element()).perturbation_site_splitting is True

    with pytest.raises(TypeError):
        s.perturbation_site_splitting = 0.1


def test_rejects_non_material_cell(run_in_tmpdir, model):
    """The swap replaces a material, so a lattice fill must fail.

    Silently doing nothing here would produce a plausible-looking zero
    worth.
    """
    water, absorber = _water_and_absorber(model)
    universe = openmc.Universe(cells=[openmc.Cell(
        fill=water, region=-openmc.Sphere(r=0.3))])
    cell = _sample_cell(model)
    cell.fill = universe

    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
    ])
    model.settings.perturbation_n_generation = 6

    with pytest.raises(RuntimeError, match='filled with a material'):
        model.run()


def test_rejects_adjoint_tally_combination(run_in_tmpdir, model):
    """BEP and adjoint tallies both drive the revival loop, incompatibly.

    `adjoint` has no setter on the Python Tally class in this branch, so
    the attribute is written into tallies.xml directly and the executable
    invoked without re-exporting.
    """
    _, absorber = _water_and_absorber(model)
    tally = openmc.Tally(tally_id=1)
    tally.scores = ['flux']
    model.tallies = openmc.Tallies([tally])
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ])
    model.settings.perturbation_n_generation = 6
    model.export_to_xml()

    tree = ET.parse('tallies.xml')
    tree.getroot().find('tally').set('adjoint', 'true')
    tree.write('tallies.xml')

    with pytest.raises(RuntimeError, match='revival loop'):
        openmc.run()
