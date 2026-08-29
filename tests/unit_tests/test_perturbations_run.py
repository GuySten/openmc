"""Self-asserting integration tests for Branched Exact Perturbation.

These run OpenMC but need no stored reference: every assertion here is an
invariant that must hold whatever the nuclear data or the machine. They are
the tests to run first when something looks wrong.
"""

import lxml.etree as ET
import numpy as np
import pytest

import openmc


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
    sample_surf = openmc.Sphere(x0=4.0, r=1.5)
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


def _materials(model):
    mats = {m.id: m for m in model.materials}
    return mats[2], mats[3]     # water, absorber


def test_null_perturbation_is_exactly_zero(run_in_tmpdir, model):
    """Substituting the material already in place must give exactly zero.

    Common random numbers plus branching at the entry point make the two
    shadow trees the same tree, history for history, so the only thing that
    can separate them is floating-point rounding: score_site() accumulates
    with an atomic add, and the interleaving across threads differs between
    the reference and perturbed slots. The tolerance below is that rounding
    and nothing else -- it is some eight orders of magnitude tighter than any
    real worth, so a genuine break in the branch pairing or the seeding
    cannot hide under it.

    This is the test to run first. Every stream of the shadow root has to be
    seeded from the shared branch id; seeding only STREAM_TRACKING leaves
    STREAM_URR_PTABLE picking up stack garbage, which decorrelates the trees
    wherever a nuclide has unresolved resonances and turns this into noise.
    """
    water, _ = _materials(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): water},
                                 perturbation_id=1, name='null'),
    ], n_generation=6)

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        p = sp.perturbations.by_id(1)
        assert abs(p.rho) < 1.0e-6, \
            f'null perturbation gave {p.rho} pcm; common random numbers are ' \
            'not holding between the reference and perturbed trees'
        assert p.std_dev < 1.0e-6
        assert np.allclose(p.depth_curve, 0.0, atol=1.0e-12)


def test_driver_is_unperturbed(run_in_tmpdir, model):
    """The driver must be an ordinary eigenvalue calculation.

    The invariant that actually matters, and the one that is bit-exact, is the
    FISSION SOURCE: sort_bank() orders the bank by parent and progeny id
    precisely so it does not depend on thread scheduling. If a shadow particle
    reaches the real fission bank, or clobbers a per-source array such as
    progeny_per_particle, the source moves and this catches it.

    k is checked only to rounding. The k tallies are accumulated with
    `#pragma omp atomic` float adds in event_death(), and floating-point
    addition is not associative, so their last bits depend on the order the
    threads finish. BEP shifts that timing simply by doing extra work, which
    means k is NOT bit-reproducible even when the driver is untouched. The
    tolerance below is nine orders of magnitude tighter than the contamination
    it is meant to catch -- a shadow particle contributing to k-eff moves it
    by O(1/N), around 1e-3 here, not 1e-12.
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

    water, absorber = _materials(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
        openmc.LocalPerturbation({_sample_cell(model): water},
                                 perturbation_id=2),
    ], n_generation=6)

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
        'BEP changed k beyond rounding: a shadow particle is reaching k-eff'


def test_absorber_worth_is_negative(run_in_tmpdir, model):
    """Sign convention, and that the estimator produces a real number.

    Replacing water with B10 in a central cavity must be worth less than
    nothing. Loose tolerance: this pins the sign, not the value.
    """
    _, absorber = _materials(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1, name='B10'),
    ], n_generation=8)

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        p = sp.perturbations.by_id(1)
        assert np.isfinite(p.rho), \
            'non-finite worth: a shadow tree went extinct and log1p(-1) ' \
            'leaked into the accumulator'
        assert np.isfinite(p.std_dev)
        assert p.std_dev > 0.0
        assert p.rho < -3.0 * p.std_dev, (
            f'B10 sample worth {p.rho:.0f} +/- {p.std_dev:.0f} pcm is not '
            'resolvably negative. If the sign is right but the error bar is '
            'too large, this is statistics, not correctness: precision scales '
            'as 1/sqrt(branch sites), so raise particles or enlarge the '
            'sample rather than loosening the assertion.')


def test_depth_curve_is_linear_not_flat(run_in_tmpdir, model):
    """The estimator is the slope of l(d), and there is no plateau.

    A curve that comes out flat would mean the perturbation is not reaching
    the shadow trees at all. Check both that it varies and that a straight
    line describes the asymptotic part.
    """
    _, absorber = _materials(model)
    # Shadow trees cost ~k**d histories each, so trade particles for depth
    model.settings.particles = 2000
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ], n_generation=10)

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


def test_covariance_is_symmetric_and_correlated(run_in_tmpdir, model):
    """Co-located perturbations must come out correlated.

    They share branch sites and seeds by construction, so a near-zero
    off-diagonal would mean the shared-seed path is broken and every
    difference would carry a needlessly large error bar.
    """
    water, absorber = _materials(model)
    cell = _sample_cell(model)
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
        openmc.LocalPerturbation({cell: water}, perturbation_id=2),
        openmc.LocalPerturbation({cell: None}, perturbation_id=3),
    ], n_generation=6)

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

        # The absorber and void perturbations act on the same cell through the
        # same branch sites, so they must be positively correlated.
        assert corr[0, 2] > 0.1

        # And the difference must be tighter than independent propagation
        _, sigma = ps.difference(3, 1)
        independent = np.hypot(ps.by_id(1).std_dev, ps.by_id(3).std_dev)
        assert sigma < independent


def test_displacement_matches_difference_of_positions(run_in_tmpdir):
    """Two routes to the same quantity must agree.

    A sliver-only displacement perturbation and the difference of two
    whole-sample perturbations one slice apart estimate the same thing. The
    displacement form should also be the tighter of the two, since it forms
    the difference inside the correlated sample instead of between two larger
    worths.
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

    # Axially sliced channel through a reflected fuel sphere. Reflective for
    # the same reason as the fixture above: a leaky system starves the shadow
    # trees and this test needs two worths precise enough to difference.
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
    # Only the symmetric difference: slice 1 reverts, slice 2 takes the sample
    moved = {ids[1]: water, ids[2]: absorber}

    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation(at_1, perturbation_id=1, name='at slice 1'),
        openmc.LocalPerturbation(at_2, perturbation_id=2, name='at slice 2'),
        openmc.LocalPerturbation(moved, perturbation_id=3, name='moved 1->2'),
    ], n_generation=8)

    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        ps = sp.perturbations
        by_diff, sigma_diff = ps.difference(2, 1)
        by_displacement = ps.by_id(3).rho
        sigma_displacement = ps.by_id(3).std_dev

        combined = np.hypot(sigma_diff, sigma_displacement)
        assert abs(by_displacement - by_diff) < 4.0 * combined, (
            f'displacement {by_displacement:.3f} +/- {sigma_displacement:.3f} '
            f'disagrees with difference {by_diff:.3f} +/- {sigma_diff:.3f}')


def test_rejects_non_material_cell(run_in_tmpdir, model):
    """The swap replaces a material, so a lattice or universe fill must fail.

    Silently doing nothing here would produce a plausible-looking zero worth.
    """
    water, absorber = _materials(model)
    universe = openmc.Universe(cells=[openmc.Cell(
        fill=water, region=-openmc.Sphere(r=0.3))])
    cell = _sample_cell(model)
    cell.fill = universe

    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({cell: absorber}, perturbation_id=1),
    ], n_generation=6)

    with pytest.raises(RuntimeError, match='filled with a material'):
        model.run()


def test_rejects_adjoint_tally_combination(run_in_tmpdir, model):
    """BEP and adjoint tallies both drive the revival loop, incompatibly.

    `adjoint` has no setter on the Python Tally class in this branch, so the
    attribute is written into tallies.xml directly and the executable invoked
    without re-exporting.
    """
    _, absorber = _materials(model)
    tally = openmc.Tally(tally_id=1)
    tally.scores = ['flux']
    model.tallies = openmc.Tallies([tally])
    model.perturbations = openmc.Perturbations([
        openmc.LocalPerturbation({_sample_cell(model): absorber},
                                 perturbation_id=1),
    ], n_generation=6)
    model.export_to_xml()

    tree = ET.parse('tallies.xml')
    tree.getroot().find('tally').set('adjoint', 'true')
    tree.write('tallies.xml')

    with pytest.raises(RuntimeError, match='revival loop'):
        openmc.run()
