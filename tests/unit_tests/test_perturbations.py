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
        ps.linearity(1)


def _transient_tau(amplitude, dominance_ratio, slope=-300e-5, n_gen=400,
                   L=10, noise=2e-3, seed=7):
    """Shadow weights whose log ratio is ``c + slope*d + A*r**d``.

    The last term is a sub-dominant mode that has not died out. It biases the
    fitted slope, always in the same direction, which is exactly the failure
    the diagnostics below have to catch.
    """
    rng = np.random.default_rng(seed)
    d = np.arange(L + 1)
    ell = slope * d + amplitude * dominance_ratio**d
    den = np.exp(np.outer(np.ones(n_gen), 4.0 + 0.3 * d))
    num = den * np.exp(ell) * np.exp(rng.normal(0, noise, (n_gen, L + 1)))
    return num[None, ...], den[None, ...]


def _fitted(amplitude, dominance_ratio, k_ref=1.38):
    ps = openmc.Perturbations(
        [openmc.LocalPerturbation({71: 92}, perturbation_id=1)])
    ps._set_results(*_transient_tau(amplitude, dominance_ratio), k_ref=k_ref)
    return ps


def test_linearity_is_a_real_chi_square():
    """Near 1 without a transient, far above it with one.

    The old version divided the residuals by the curve's own magnitude rather
    than by their uncertainty, so it read ~0 for any curve with a linear
    trend -- including one whose slope was biased by hundreds of pcm.
    """
    clean = _fitted(0.0, 0.7)
    assert 0.2 < clean.linearity(1) < 5.0

    contaminated = _fitted(0.05, 0.7)
    assert contaminated.linearity(1) > 20.0


def test_worth_by_fit_start_exposes_a_transient():
    """The worth must stop moving as the fit window shrinks.

    This is the diagnostic that answers "is n_generation large enough". A
    sub-dominant mode makes the fitted worth drift monotonically with the
    starting depth; without one it sits still.
    """
    def drift(ps):
        curve = ps.worth_by_fit_start(1)
        starts = sorted(curve)
        return curve[starts[1]][0] - curve[starts[-1]][0]

    assert abs(drift(_fitted(0.0, 0.7))) < 20.0
    assert abs(drift(_fitted(0.05, 0.7))) > 100.0


def test_fit_diagnostics_need_results():
    ps = openmc.Perturbations([openmc.LocalPerturbation({71: 92},
                                                        perturbation_id=1)])
    with pytest.raises(ValueError):
        ps.linearity(1)
    with pytest.raises(ValueError):
        ps.worth_by_fit_start(1)


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

    ``tau`` is [generation][tree][depth]. Tree 0 is the shared reference; tree
    ``i + 1`` belongs to perturbation ``ids[i]``.

    ``run_mode`` and ``k_combined`` are written because the reader needs
    k-effective to turn the fitted slope (dk/k) into a reactivity. The default
    of 1.0 makes that conversion the identity, so tests that check a slope
    read unchanged; pass a realistic k to exercise the conversion.
    """
    tau = np.asarray(tau, dtype=float)
    n_rec, n_trees, nd = tau.shape
    assert nd == n_generation + 1
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('statepoint')
        f.attrs['version'] = [18, 0]
        f.create_dataset('run_mode', data=np.bytes_('eigenvalue'))
        f.create_dataset('k_combined', data=np.array([keff, 1.0e-5]))
        g = f.create_group('local_perturbation')
        g.create_dataset('n_generation', data=n_generation)
        g.create_dataset('n_generations_recorded', data=n_rec)
        g.create_dataset('n_trees', data=n_trees)
        g.create_dataset('n_branch', data=123456)
        g.create_dataset('w_branch', data=1.0)
        g.create_dataset('n_perturbations', data=len(ids))
        g.create_dataset('ids', data=np.asarray(ids, dtype=np.int32))
        g.create_dataset('tau', data=tau.ravel())
        for i, pid in enumerate(ids):
            pg = g.create_group(f'perturbation {pid}')
            pg.create_dataset('index', data=i)
            pg.create_dataset('tree', data=i + 1)
            pg.create_dataset('ref_trees',
                              data=np.array([0], dtype=np.int32))
            pg.create_dataset('cells', data=np.array([71], dtype=np.int32))
            pg.create_dataset('materials',
                              data=np.array([92 + i], dtype=np.int32))


def _branching_tau(rng, k, rho, n_branch, n_gen, L):
    """Simulate shadow forests: reference and perturbed weight per depth."""
    ref = np.zeros((n_gen, L + 1))
    pert = np.zeros((n_gen, L + 1))
    for g in range(n_gen):
        for out, kk in ((ref, k), (pert, k * (1.0 + rho))):
            alive = np.full(n_branch, 1.0)
            out[g, 0] = n_branch
            for d in range(1, L + 1):
                alive = rng.poisson(kk * alive).astype(float)
                out[g, d] = alive.sum()
    return ref, pert


def test_statepoint_parsing(run_in_tmpdir):
    """Recover a known worth from simulated shadow forests.

    The estimator is the slope of ln(tau_p / R_p), so feeding it branching
    processes with a known ratio of mean offspring must return that ratio.
    """
    L, n_gen, k, rho = 10, 200, 2.2, -300e-5
    rng = np.random.default_rng(20240829)
    ref, pert = _branching_tau(rng, k, rho, 60, n_gen, L)

    tau = np.stack([ref, pert], axis=1)          # [gen][tree][depth]
    _write_statepoint('sp.h5', tau, [1], L)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        ps = sp.perturbations
        assert isinstance(ps, openmc.Perturbations)
        assert ps.ids == [1]
        assert ps.n_generation == L
        assert ps.n_blocks >= 2

        got = ps.by_id(1).rho
        assert np.isfinite(got.nominal_value) and got.std_dev > 0.0
        assert abs(got.nominal_value - 1e5 * rho) < 4.0 * got.std_dev, \
            f'recovered {got:.1f} pcm, expected {1e5 * rho:.1f}'

        assert ps.by_id(1).substitutions == {71: 92}
        assert len(ps.by_id(1).depth_curve) == L + 1
        assert np.isfinite(ps.by_id(1).depth_curve).all()


def test_slope_is_converted_to_a_reactivity(run_in_tmpdir):
    """The fitted slope is dk/k; the reported worth must be a reactivity.

    The slope of ln(tau_p/R_p) is ln(k_p/k_ref). The reactivity difference is
    1/k_ref - 1/k_p, smaller by a factor of k. Reporting the slope directly
    overstates every worth by that factor -- which is exactly what the 7x7
    benchmark showed before this conversion existed.
    """
    L, n_gen, k = 8, 120, 1.38
    rng = np.random.default_rng(4242)
    ref, pert = _branching_tau(rng, k, -300e-5, 40, n_gen, L)

    def worth(k_ref):
        ps = openmc.Perturbations(
            [openmc.LocalPerturbation({71: 92}, perturbation_id=1)])
        ps._set_results(np.array([pert]), np.array([ref]), k_ref=k_ref)
        return ps.by_id(1).rho

    raw = worth(1.0)          # dk/k
    converted = worth(k)      # reactivity

    assert converted.nominal_value == pytest.approx(
        raw.nominal_value / k, rel=1e-6), \
        'the slope was not divided by k'
    assert converted.std_dev == pytest.approx(raw.std_dev / k, rel=1e-6), \
        'the uncertainty was not transformed with the same Jacobian'
    # exact form, not just the first-order 1/k
    slope = -np.log1p(-raw.nominal_value / 1e5)
    assert converted.nominal_value == pytest.approx(
        1e5 * (1.0 - np.exp(-slope)) / k, rel=1e-9)


def test_statepoint_applies_the_reactivity_conversion(run_in_tmpdir):
    """The reader must divide the fitted slope by the run's k-effective.

    `test_slope_is_converted_to_a_reactivity` checks the conversion itself;
    this checks that StatePoint actually supplies k rather than leaving the
    worth in dk/k. That omission is invisible to every internal consistency
    check -- it took an independent method on the OECD benchmark to catch it.
    """
    L, n_gen, k = 8, 120, 1.38
    rng = np.random.default_rng(31337)
    ref, pert = _branching_tau(rng, k, -300e-5, 40, n_gen, L)
    tau = np.stack([ref, pert], axis=1)

    _write_statepoint('slope.h5', tau, [1], L, keff=1.0)
    _write_statepoint('rho.h5', tau, [1], L, keff=k)

    with openmc.StatePoint('slope.h5', autolink=False) as sp:
        slope = sp.perturbations.by_id(1).rho
    with openmc.StatePoint('rho.h5', autolink=False) as sp:
        rho = sp.perturbations.by_id(1).rho

    assert rho.nominal_value == pytest.approx(slope.nominal_value / k,
                                              rel=1e-6)
    assert rho.std_dev == pytest.approx(slope.std_dev / k, rel=1e-6)


def test_statepoint_survives_extinct_generations(run_in_tmpdir):
    """Generations where a whole shadow forest dies must not poison the run.

    This is the failure the per-generation form had: log(0) is -inf and one
    bad generation turned the whole worth into nan. Summing over every
    progenitor before dividing is what makes it harmless.
    """
    L, n_gen = 8, 120
    rng = np.random.default_rng(7)
    ref, pert = _branching_tau(rng, 2.2, -300e-5, 3, n_gen, L)

    # Force some generations to have a completely extinct perturbed forest
    pert[::7, 1:] = 0.0
    assert (pert[:, 1:].sum(1) == 0).any()

    tau = np.stack([ref, pert], axis=1)
    _write_statepoint('sp.h5', tau, [1], L)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        ps = sp.perturbations
        got = ps.by_id(1).rho
        assert np.isfinite(got.nominal_value), \
            'extinct generations produced a nan worth'
        assert np.isfinite(got.std_dev)
        assert np.isfinite(ps.covariance).all()


def test_statepoint_zero_depth_raises(run_in_tmpdir):
    """No weight past depth 0 is a build fault, and must be named as one.

    It means no shadow tree ever produced a fission site -- the symptom of a
    chain-bounding gate reading settings::super_n_generation instead of
    bep::generation_limit(). Reporting it as thin statistics would send
    someone off adding particles for no reason.
    """
    L, n_gen = 8, 40
    ref = np.ones((n_gen, L + 1))
    pert = np.zeros((n_gen, L + 1))
    pert[:, 0] = 1.0                     # roots exist, nothing descends
    tau = np.stack([ref, pert], axis=1)
    _write_statepoint('sp.h5', tau, [1], L)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        with pytest.raises(DataError, match='fission site'):
            sp.perturbations


def test_statepoint_degenerate_jackknife_raises(run_in_tmpdir):
    """If a leave-one-out replicate has nothing left, fail loudly.

    Weight past depth 0 exists, so this is not the build fault above, but it
    is concentrated in a single generation: drop that block and the replicate
    has no surviving tree at any depth. Returning nan there would be worse
    than an error.
    """
    L, n_gen = 8, 40
    ref = np.ones((n_gen, L + 1))
    pert = np.zeros((n_gen, L + 1))
    pert[:, 0] = 1.0
    pert[0, :] = 1.0                     # only the first generation survives
    tau = np.stack([ref, pert], axis=1)
    _write_statepoint('sp.h5', tau, [1], L)

    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        with pytest.raises(DataError, match='extinct'):
            sp.perturbations


def test_statepoint_covariance_is_symmetric(run_in_tmpdir):
    L, n_gen = 8, 150
    rng = np.random.default_rng(99)
    ref, p1 = _branching_tau(rng, 2.2, -300e-5, 40, n_gen, L)
    _, p2 = _branching_tau(rng, 2.2, -150e-5, 40, n_gen, L)
    tau = np.stack([ref, p1, p2], axis=1)
    _write_statepoint('sp.h5', tau, [1, 2], L)

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
    L, n_gen = 8, 60
    rng = np.random.default_rng(1)
    ref, pert = _branching_tau(rng, 2.2, -100e-5, 30, n_gen, L)
    _write_statepoint('sp.h5', np.stack([ref, pert], axis=1), [1], L)
    with openmc.StatePoint('sp.h5', autolink=False) as sp:
        assert sp.perturbations is sp.perturbations


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


def test_depth_curve_is_linear_not_flat(run_in_tmpdir, model):
    """The estimator is the slope of l(d), and there is no plateau.

    A curve that comes out flat would mean the perturbation is not reaching
    the shadow trees at all. Check both that it varies and that a straight
    line describes the asymptotic part.
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
        curve = ps.by_id(1).depth_curve
        assert len(curve) == 11

        # It must actually move with depth. A flat curve would mean the
        # perturbation never reached the shadow trees at all.
        assert abs(curve[-1] - curve[ps.n_generation // 2]) > 0.0

        # And be describable by a straight line over the fitted range
        assert ps.linearity(1) < 0.5


def test_covariance_is_symmetric_and_correlated(run_in_tmpdir,
                                                model):
    """Co-located perturbations must come out correlated.

    They share branch sites and seeds by construction, so a near-zero
    off-diagonal would mean the shared-seed path is broken and every
    difference would carry a needlessly large error bar.
    """
    water, absorber = _water_and_absorber(model)
    fuel = model.materials[0]
    cell = _sample_cell(model)
    # No null perturbation here. A null has exactly zero worth AND exactly
    # zero variance, so its row and column of the covariance are zero,
    # and its correlation with itself is 0/0 -- correct, but it makes a
    # correlation-matrix assertion meaningless. Three genuinely different
    # substitutions instead: absorber and void are negative, fuel positive.
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
        openmc.LocalPerturbation({cell: fuel}, perturbation_id=2),
        openmc.LocalPerturbation({cell: None}, perturbation_id=3),
    ])
    model.settings.perturbation_n_generation = 6

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        cov = ps.covariance
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)
        assert (np.diag(cov) >= 0.0).all()

        corr = ps.correlation()
        assert np.allclose(np.diag(corr), 1.0)
        assert np.all(np.abs(corr) <= 1.0 + 1e-9)

        # Absorber and void act on the same cell through the same branch
        # sites, so they must be positively correlated.
        assert corr[0, 2] > 0.1
        assert (np.diag(cov) > 0.0).all(), \
            'a perturbation has zero variance; is one of them a null?'

        # Subtracting the correlated rho values must beat treating them as
        # independent -- that is what correlated_values buys.
        diff = ps.by_id(3).rho - ps.by_id(1).rho
        independent = np.hypot(ps.by_id(1).rho.std_dev,
                               ps.by_id(3).rho.std_dev)
        assert diff.std_dev < independent


def test_displacement_matches_difference_of_positions(run_in_tmpdir):
    """Two routes to the same quantity must agree.

    A sliver-only displacement perturbation and the difference of two
    whole-sample perturbations one slice apart estimate the same thing. The
    displacement form should also be the tighter of the two, since it forms
    the difference inside the correlated sample instead of between two
    larger worths.
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

    # Axially sliced channel through a reflected fuel sphere.
    # Reflective for the same reason as the fixture above: a leaky system
    # starves the shadow trees, and this test needs two worths precise
    # enough to difference.
    outer = openmc.Sphere(r=10.0, boundary_type='reflective')
    channel = openmc.ZCylinder(x0=4.0, r=1.2)
    planes = [openmc.ZPlane(z0=z) for z in np.linspace(-3.0, 3.0, 5)]

    slices = []
    for i in range(len(planes) - 1):
        slices.append(openmc.Cell(
            cell_id=100 + i, fill=water,
            region=-channel & +planes[i] & -planes[i + 1]))

    bulk_region = -outer & ~openmc.Union(
        [c.region for c in slices])
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
    at_1 = {c: water for c in ids}
    at_1[ids[1]] = absorber
    at_2 = {c: water for c in ids}
    at_2[ids[2]] = absorber
    # Only the symmetric difference: slice 1 reverts, 2 takes the
    # sample
    moved = {ids[1]: water, ids[2]: absorber}

    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation(at_1, perturbation_id=1,
                                 name='at slice 1'),
        openmc.LocalPerturbation(at_2, perturbation_id=2,
                                 name='at slice 2'),
        openmc.LocalPerturbation(moved, perturbation_id=3,
                                 name='moved 1->2'),
    ])
    model.settings.perturbation_n_generation = 8

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        by_diff = ps.by_id(2).rho - ps.by_id(1).rho
        by_displacement = ps.by_id(3).rho

        # Both routes share the same run, so subtract them as correlated
        # quantities too: the residual carries the right uncertainty.
        residual = by_displacement - by_diff
        assert abs(residual.nominal_value) < 4.0 * residual.std_dev, (
            f'displacement {by_displacement:.3f} disagrees with '
            f'difference {by_diff:.3f} (residual {residual:.3f})')


def test_multigroup_null_is_exactly_zero(run_in_tmpdir):
    """BEP must work in multigroup mode, not just continuous energy.

    SourceSite.E carries a GROUP INDEX in multigroup mode -- from_source()
    does ``g() = int(src->E)`` there, while create_secondary() and split()
    both write ``run_CE ? E() : g()``. Recording a shadow root's energy
    without that distinction fed an energy in as a group index, which runs
    off the end of every group-indexed array.

    The null test catches it because a shadow root launched at a nonsense
    group does not reproduce the reference tree.
    """
    groups = openmc.mgxs.EnergyGroups([0.0, 1.0e5, 20.0e6])

    def xsdata(name, absorption, scatter, fissile):
        """Two-group data that balances against the totals.

        For the fissile ones nu_fission is set EQUAL to absorption in both
        groups, so k_inf = sum(nu_f phi) / sum(abs phi) = 1 for ANY
        spectrum. The test cannot drift supercritical however the flux
        settles, which matters because shadow trees grow as k**L: an earlier
        version of this data gave k = 3.3 and the trees ran away.
        """
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

    # absorption + total scatter out == total, group by group
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
    model.settings.particles = 2000
    model.settings.batches = 25
    model.settings.inactive = 5
    model.settings.seed = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    cell = model.geometry.get_all_cells()[10]
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
        openmc.LocalPerturbation({cell: sample}, perturbation_id=2,
                                 name='null'),
    ])
    model.settings.perturbation_n_generation = 6

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        assert abs(ps.by_id(2).rho.nominal_value) < 1.0e-6, \
            'multigroup null perturbation is not zero'
        assert ps.by_id(2).rho.std_dev < 1.0e-6
        # and the real perturbation has to produce something finite
        worth = ps.by_id(1).rho
        assert np.isfinite(worth.nominal_value)
        assert worth.std_dev > 0.0


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
