"""Test the IFP beta numerators that score delayed group probabilities in place
of the analog delayed indicator."""

import numpy as np
import pytest
import openmc


@pytest.fixture
def model():
    openmc.reset_auto_ids()
    material = openmc.Material()
    material.add_nuclide("U235", 1.0)
    material.set_density("g/cm3", 16.0)
    sphere = openmc.Sphere(r=10.0, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)

    model = openmc.Model(geometry=openmc.Geometry([cell]))
    model.settings.particles = 2000
    model.settings.batches = 25
    model.settings.inactive = 5
    model.settings.ifp_n_generation = 5
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(*cell.bounding_box),
        constraints={"fissionable": True},
    )

    groups = openmc.DelayedGroupFilter(list(range(1, 7)))
    scores = [
        "ifp-beta-numerator",
        "ifp-beta-numerator-group",
        "ifp-beta-numerator-cond",
    ]
    denominator = openmc.Tally(name="denominator")
    denominator.scores = ["ifp-denominator"]
    model.tallies.append(denominator)
    for score in scores:
        total = openmc.Tally(name=score)
        total.scores = [score]
        groupwise = openmc.Tally(name=f"{score} groupwise")
        groupwise.scores = [score]
        groupwise.filters = [groups]
        model.tallies.extend([total, groupwise])
    return model


def test_conditional_numerators(run_in_tmpdir, model):
    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        def total(name):
            return sp.get_tally(name=name).mean.ravel()[0]

        def groupwise(name):
            return sp.get_tally(name=f"{name} groupwise").mean.ravel()

        analog = total("ifp-beta-numerator")
        group = total("ifp-beta-numerator-group")
        cond = total("ifp-beta-numerator-cond")
        denominator = total("denominator")

        # The group-only estimator keeps the analog delayed indicator, so its
        # total equals the analog total; only the split over groups differs
        assert group == pytest.approx(analog, rel=1e-10)

        # Groupwise tallies add up to the totals
        for name in ["ifp-beta-numerator", "ifp-beta-numerator-group",
                     "ifp-beta-numerator-cond"]:
            assert groupwise(name).sum() == pytest.approx(total(name), rel=1e-10)

        # The conditional estimator scores a probability for every ancestor,
        # so every group receives a score. The analog estimate is too noisy at
        # this run length to compare with, so check a physical range instead.
        assert np.all(groupwise("ifp-beta-numerator-cond") > 0.0)
        assert 0.005 < cond / denominator < 0.008
