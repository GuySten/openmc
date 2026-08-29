from pathlib import Path

import h5py
import lxml.etree as ET
import numpy as np
import pytest

import openmc
from openmc.exceptions import DataError


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
    assert 'Worth' not in repr(p)
    p.rho, p.std_dev = -40.0, 2.0
    assert 'Worth' in repr(p)


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
    ], n_generation=13)

    qs = openmc.Perturbations.from_xml_element(ps.to_xml_element())
    assert qs.n_generation == 13
    assert qs.ids == [1, 2]
    assert qs.by_id(1).name == 'steel'
    assert qs.by_id(2).substitutions == {71: 91, 72: 92}


def test_collection_file_roundtrip(run_in_tmpdir):
    ps = openmc.Perturbations([
        openmc.LocalPerturbation({71: 92}, perturbation_id=1),
    ], n_generation=11)
    ps.export_to_xml()
    assert Path('perturbations.xml').is_file()

    qs = openmc.Perturbations.from_xml('perturbations.xml')
    assert qs.n_generation == 11
    assert qs.by_id(1).substitutions == {71: 92}


def test_xml_element_name_matches_cpp_reader():
    # bep::read_perturbations_xml() looks for <perturbations> with
    # <n_generation> and <local_perturbation> children. If these tags drift the
    # C++ silently reads nothing, so pin them down.
    elem = openmc.Perturbations(
        [openmc.LocalPerturbation({71: 92})], n_generation=9).to_xml_element()
    assert elem.tag == 'perturbations'
    assert elem.find('n_generation').text == '9'
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


def test_n_generation_validation():
    # The estimator is a finite difference in depth, so L < 2 is meaningless.
    with pytest.raises(ValueError):
        openmc.Perturbations(n_generation=1)
    with pytest.raises(TypeError):
        openmc.Perturbations(n_generation=10.5)
    assert openmc.Perturbations(n_generation=2).n_generation == 2


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
    ], n_generation=10)
    ps[0].rho, ps[0].std_dev = -40.0, 2.0
    ps[1].rho, ps[1].std_dev = -40.5, 2.0
    ps.covariance = np.array([[4.0, 3.96], [3.96, 4.0]])
    return ps


def test_combine_matches_hand_algebra(results):
    value, std_dev = results.combine({1: 0.25, 2: 0.75})
    w = np.array([0.25, 0.75])
    rho = np.array([-40.0, -40.5])
    assert value == pytest.approx(float(w @ rho))
    assert std_dev == pytest.approx(float(np.sqrt(w @ results.covariance @ w)))


def test_difference_beats_independent_propagation(results):
    """The whole point of running perturbations together.

    Treating two co-located worths as independent gives sqrt(2)*sigma for
    their difference. Carrying the covariance must do far better, or the
    covariance is not being read.
    """
    value, std_dev = results.difference(2, 1)
    assert value == pytest.approx(-0.5)
    independent = np.hypot(2.0, 2.0)
    assert std_dev == pytest.approx(np.sqrt(2 * (4.0 - 3.96)))
    assert std_dev < independent / 5


def test_derivative_scales_by_dz(results):
    value, std_dev = results.difference(2, 1)
    dvalue, dstd = results.derivative(2, 1, 0.5)
    assert dvalue == pytest.approx(value / 0.5)
    assert dstd == pytest.approx(std_dev / 0.5)
    # A negative step must not flip the sign of the uncertainty
    assert results.derivative(2, 1, -0.5)[1] > 0


def test_correlation_matrix(results):
    corr = results.correlation()
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(3.96 / 4.0)


def test_results_methods_raise_without_results():
    ps = openmc.Perturbations([openmc.LocalPerturbation({71: 92},
                                                        perturbation_id=1)])
    with pytest.raises(ValueError):
        ps.combine({1: 1.0})
    with pytest.raises(ValueError):
        ps.correlation()
    with pytest.raises(ValueError):
        ps.linearity(1)


def test_linearity_separates_straight_from_curved():
    ps = openmc.Perturbations([
        openmc.LocalPerturbation({71: 92}, perturbation_id=1),
        openmc.LocalPerturbation({71: 93}, perturbation_id=2),
    ], n_generation=10)
    d = np.arange(11)
    # l(d) is linear in the asymptotic regime; its slope is the worth
    ps.by_id(1).depth_curve = 1e-4 + d * -5e-4
    # a transient that never straightens out
    ps.by_id(2).depth_curve = -5e-4 * d**2
    assert ps.linearity(1) < 1e-6
    assert ps.linearity(2) > ps.linearity(1)


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

    ps = openmc.Perturbations([p], n_generation=12)
    model.perturbations = ps
    assert model.perturbations is ps
    assert model.perturbations.n_generation == 12

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
    ], n_generation=12)

    model.export_to_xml()
    assert Path('perturbations.xml').is_file()

    reloaded = openmc.Model.from_xml()
    assert reloaded.perturbations.n_generation == 12
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
    ], n_generation=8)

    model.export_to_model_xml()
    root = ET.parse('model.xml').getroot()
    assert root.find('perturbations') is not None

    reloaded = openmc.Model.from_model_xml()
    assert reloaded.perturbations.n_generation == 8
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

def _write_statepoint(path, tau, ids, n_generation, trees_per_pert=None):
    """Minimal statepoint carrying only a local_perturbation group.

    ``tau`` is [generation][tree][depth]. Tree 0 is the shared reference; tree
    ``i + 1`` belongs to perturbation ``ids[i]``.
    """
    tau = np.asarray(tau, dtype=float)
    n_rec, n_trees, nd = tau.shape
    assert nd == n_generation + 1
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_('statepoint')
        f.attrs['version'] = [18, 0]
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

        value, std_dev = ps.value(1)
        assert np.isfinite(value) and std_dev > 0.0
        assert abs(value - 1e5 * rho) < 4.0 * std_dev, \
            f'recovered {value:.1f} +/- {std_dev:.1f} pcm, expected ' \
            f'{1e5 * rho:.1f}'

        assert ps.by_id(1).substitutions == {71: 92}
        assert len(ps.by_id(1).depth_curve) == L + 1
        assert np.isfinite(ps.by_id(1).depth_curve).all()


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
        value, std_dev = ps.value(1)
        assert np.isfinite(value), 'extinct generations produced a nan worth'
        assert np.isfinite(std_dev)
        assert np.isfinite(ps.covariance).all()


def test_statepoint_all_extinct_raises(run_in_tmpdir):
    """If no blocking gives usable sums, fail loudly rather than return nan."""
    L, n_gen = 8, 40
    ref = np.ones((n_gen, L + 1))
    pert = np.zeros((n_gen, L + 1))
    pert[:, 0] = 1.0                     # roots survive, nothing else does
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
        assert np.isfinite(ps.difference(2, 1)).all()


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
