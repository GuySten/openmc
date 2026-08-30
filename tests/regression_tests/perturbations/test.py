"""Regression test for Branched Exact Perturbation local perturbation worths.

Covers the three shapes a perturbation can take -- a single-cell material
swap, a swap to void, and a multi-cell substitution representing a rigid
sample displacement -- plus the covariance between them.

The reference values in results_true.dat are nuclear-data dependent, so
regenerate with ``pytest --update`` when the data library changes.
"""

import numpy as np
import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


@pytest.fixture
def model():
    model = openmc.Model()

    fuel = openmc.Material(material_id=1, name='fuel')
    fuel.add_nuclide('U235', 1.0)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.0)

    water = openmc.Material(material_id=2, name='water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 1.0)

    absorber = openmc.Material(material_id=3, name='absorber')
    absorber.add_nuclide('B10', 1.0)
    absorber.set_density('g/cm3', 2.5)

    model.materials = openmc.Materials([fuel, water, absorber])

    # A short axially sliced channel through the middle of a fuel sphere, so
    # that a displacement can be expressed as substitutions on two slices.
    # Reflective: a leaky system makes the shadow trees decay as k**d, so by
    # depth L there is too little descendant weight left for the worths to be
    # worth regressing.
    outer = openmc.Sphere(r=10.0, boundary_type='reflective')
    channel = openmc.ZCylinder(x0=4.0, r=1.2)
    planes = [openmc.ZPlane(z0=z) for z in (-3.0, -1.0, 1.0, 3.0)]

    slices = [
        openmc.Cell(cell_id=100 + i, name=f'slice {i}', fill=water,
                    region=-channel & +planes[i] & -planes[i + 1])
        for i in range(len(planes) - 1)
    ]
    bulk = openmc.Cell(
        cell_id=200, name='bulk', fill=fuel,
        region=-outer & ~openmc.Union([c.region for c in slices]))

    model.geometry = openmc.Geometry(slices + [bulk])

    model.settings.particles = 5000
    model.settings.batches = 60
    model.settings.inactive = 10
    model.settings.seed = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())

    ids = [c.id for c in slices]
    at_0 = dict.fromkeys(ids, water.id) | {ids[0]: absorber.id}
    at_1 = dict.fromkeys(ids, water.id) | {ids[1]: absorber.id}

    model.perturbations = openmc.Perturbations([
        # Single-cell swap
        openmc.LocalPerturbation(
            {ids[1]: absorber}, perturbation_id=1, name='absorber swap'),
        # Swap to void
        openmc.LocalPerturbation(
            {ids[1]: None}, perturbation_id=2, name='void'),
        # Null test: substitutes the material already in place, so this must
        # come out exactly zero regardless of nuclear data
        openmc.LocalPerturbation(
            {ids[1]: water}, perturbation_id=3, name='null'),
        # Whole-sample perturbations at two adjacent positions
        openmc.LocalPerturbation(at_0, perturbation_id=4, name='at slice 0'),
        openmc.LocalPerturbation(at_1, perturbation_id=5, name='at slice 1'),
        # The same move as slivers only: slice 0 reverts, slice 1 takes it
        openmc.LocalPerturbation(
            {ids[0]: water, ids[1]: absorber},
            perturbation_id=6, name='displacement 0->1'),
    ])
    model.settings.perturbation_n_generation = 10

    return model


class PerturbationTestHarness(PyAPITestHarness):
    @staticmethod
    def _check_invariants(perturbations):
        """Assertions that hold whatever the nuclear data says.

        Deliberately checked from _get_results() rather than
        _compare_results(), so that they also fire when someone regenerates
        the reference with --update. A broken null test baked into
        results_true.dat would be far worse than a failing diff.
        """
        # The null perturbation must be exactly zero: common random numbers
        # plus branching at the entry point make its two shadow trees the
        # same tree, history for history.
        null = perturbations.by_id(3)
        assert null.rho.nominal_value == 0.0, \
            f'null perturbation gave {null.rho} pcm, not exactly zero'
        assert null.rho.std_dev == 0.0

        # No sign assertions here, deliberately. This model's worths sit
        # below 1.5 sigma, so any sign check is a coin flip -- and the sign
        # is not obvious anyway: voiding a water cavity in a fast U235
        # system removes moderation and H1 absorption at the same time, and
        # which wins is a question for the data, not for an assertion.
        # test_absorber_worth_is_negative covers the sign on a model built to
        # resolve it.

        # Covariance must be symmetric with a non-negative diagonal, and the
        # correlation matrix well formed for every perturbation that has a
        # variance at all (the null does not).
        cov = perturbations.covariance
        assert np.allclose(cov, cov.T)
        assert (np.diag(cov) >= 0.0).all()

        # A zero-variance perturbation has no correlation with anything, so
        # its row and column are nan by design. Check the rest.
        corr = perturbations.correlation()
        varying = np.diag(cov) > 0.0
        block = corr[np.ix_(varying, varying)]
        assert np.all(np.abs(block) <= 1.0 + 1e-9)
        assert np.allclose(np.diag(block), 1.0)
        assert np.isnan(corr[~varying]).all(), \
            'a zero-variance perturbation should have an undefined, not a ' \
            'zero, correlation'

        # The two routes to the same displacement must agree: differencing
        # two whole-sample worths, and substituting only the slivers.
        by_diff = perturbations.by_id(5).rho - perturbations.by_id(4).rho
        residual = perturbations.by_id(6).rho - by_diff
        assert residual.std_dev > 0.0, \
            'residual has zero uncertainty; the comparison would be vacuous'
        assert abs(residual.nominal_value) < 4.0 * residual.std_dev, (
            f'displacement {perturbations.by_id(6).rho:.4g} disagrees with '
            f'difference {by_diff:.4g} (residual {residual:.4g})')

    def _get_results(self):
        """Digest the perturbation results and return as a string."""
        outstr = super()._get_results()

        with openmc.StatePoint(self._sp_name) as sp:
            perturbations = sp.perturbations
            self._check_invariants(perturbations)

            outstr += f'blocks: {perturbations.n_blocks}\n'
            outstr += 'perturbations:\n'
            for p in perturbations:
                outstr += '{} {} {:12.6E} {:12.6E}\n'.format(
                    p.id, p.name, p.rho.nominal_value, p.rho.std_dev)

            # The covariance is the reason these run together, so regress it
            # rather than only the diagonal.
            outstr += 'covariance:\n'
            for row in perturbations.covariance:
                outstr += ' '.join(f'{v:12.6E}' for v in row) + '\n'

            # The worth is the slope of the depth curve, so a change in its
            # shape matters even when the fitted value lands in the same
            # place.
            outstr += 'depth curves:\n'
            for p in perturbations:
                outstr += '{} {}\n'.format(
                    p.id, ' '.join(f'{v:12.6E}' for v in p.depth_curve))

        return outstr


def test_perturbations(model):
    harness = PerturbationTestHarness('statepoint.60.h5', model)
    harness.main()
