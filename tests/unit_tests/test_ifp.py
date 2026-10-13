"""Test the Iterated Fission Probability (IFP) method to compute adjoint-weighted
kinetics parameters using dedicated tallies."""

import numpy as np
import pytest
import openmc


def test_xml_serialization(run_in_tmpdir):
    """Check that a simple use case can be written and read in XML."""
    parameter = 5
    settings = openmc.Settings()
    settings.ifp_n_generation = parameter
    settings.export_to_xml()

    read_settings = openmc.Settings.from_xml()
    assert read_settings.ifp_n_generation == parameter


@pytest.fixture(scope="module")
def geometry():
    openmc.reset_auto_ids()
    material = openmc.Material()
    material.add_nuclide("U235", 1.0)
    sphere = openmc.Sphere(r=1.0, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)
    return openmc.Geometry([cell])


@pytest.mark.parametrize(
    "options, error",
    [
        ({"ifp_n_generation": 0}, ValueError),
        ({"ifp_n_generation": -1}, ValueError),
        ({"run_mode": "fixed source"}, RuntimeError),
        ({"inactive": 5, "ifp_n_generation": 6}, RuntimeError),
        ({"inactive": 9}, RuntimeError)
    ],
)
def test_exceptions(options, error, run_in_tmpdir, geometry):
    """Test settings configuration that should return an error."""
    with pytest.raises(error):
        settings = openmc.Settings(**options)
        settings.particles = 100
        settings.batches = 15
        tally = openmc.Tally(name="ifp-scores")
        tally.scores = ["ifp-time-numerator", "ifp-beta-numerator", "ifp-denominator"]
        tallies = openmc.Tallies([tally])
        model = openmc.Model(geometry=geometry, settings=settings, tallies=tallies)
        model.run()


@pytest.mark.parametrize(
    "num_groups, use_auto_tallies",
    [
        (None, True),
        (None, False),
        (6, True),
        (6, False),
    ],
)
def test_get_kinetics_parameters(run_in_tmpdir, geometry, num_groups, use_auto_tallies):
    # Create basic model
    model = openmc.Model(geometry=geometry)
    model.settings.particles = 1000
    model.settings.batches = 20
    model.settings.inactive = 5
    model.settings.ifp_n_generation = 5

    # Add IFP tallies either via the convenience method or manually
    if use_auto_tallies:
        model.add_kinetics_parameters_tallies(num_groups=num_groups)
    else:
        for score in ["ifp-time-numerator", "ifp-beta-numerator", "ifp-denominator"]:
            tally = openmc.Tally()
            tally.scores = [score]
            if score == "ifp-beta-numerator" and num_groups is not None:
                tally.filters = [openmc.DelayedGroupFilter(list(range(1, num_groups + 1)))]
            model.tallies.append(tally)

    # Run and get kinetics parameters
    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        params = sp.get_kinetics_parameters()
    assert isinstance(params, openmc.KineticsParameters)
    assert params.generation_time is not None
    assert params.beta_effective is not None
    if num_groups is not None:
        assert len(params.beta_effective) == num_groups


@pytest.fixture
def two_nuclide_model():
    """HEU sphere in a natural uranium reflector, so that the ancestors of
    fission neutrons come from fission of two nuclides."""
    openmc.reset_auto_ids()
    core = openmc.Material()
    core.add_nuclide("U235", 0.94)
    core.add_nuclide("U238", 0.06)
    core.set_density("g/cm3", 18.7)
    reflector = openmc.Material()
    reflector.add_nuclide("U238", 0.993)
    reflector.add_nuclide("U235", 0.007)
    reflector.set_density("g/cm3", 19.0)
    inner = openmc.Sphere(r=6.0)
    outer = openmc.Sphere(r=20.0, boundary_type="vacuum")
    cells = [openmc.Cell(region=-inner, fill=core),
             openmc.Cell(region=+inner & -outer, fill=reflector)]

    model = openmc.Model(geometry=openmc.Geometry(cells))
    model.settings.particles = 2000
    model.settings.batches = 25
    model.settings.inactive = 6
    model.settings.ifp_n_generation = 5
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-6, -6, -6], [6, 6, 6]),
        constraints={"fissionable": True},
    )
    return model


def test_beta_numerator_bins(run_in_tmpdir, two_nuclide_model):
    """IFP scores binned by delayed group and by the nuclide whose fission
    emitted the ancestor add up to their totals."""
    model = two_nuclide_model
    scores = ["ifp-time-numerator", "ifp-beta-numerator", "ifp-denominator"]
    groups = openmc.DelayedGroupFilter(list(range(1, 7)))

    total = openmc.Tally(name="total")
    total.scores = scores
    by_nuclide = openmc.Tally(name="nuclide")
    by_nuclide.scores = scores
    by_nuclide.nuclides = ["U235", "U238"]
    by_group = openmc.Tally(name="group")
    by_group.scores = ["ifp-beta-numerator"]
    by_group.filters = [groups]
    by_both = openmc.Tally(name="nuclide and group")
    by_both.scores = ["ifp-beta-numerator"]
    by_both.nuclides = ["U235", "U238"]
    by_both.filters = [groups]
    model.tallies = [total, by_nuclide, by_group, by_both]

    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        def mean(name, score):
            tally = sp.get_tally(name=name)
            return tally.get_values(scores=[score]).squeeze()

        for score in scores:
            nuclide_sum = mean("nuclide", score).sum()
            assert nuclide_sum == pytest.approx(mean("total", score), rel=1e-10)
        beta_total = mean("total", "ifp-beta-numerator")
        assert mean("group", "ifp-beta-numerator").sum() == pytest.approx(
            beta_total, rel=1e-10)
        assert mean("nuclide and group", "ifp-beta-numerator").sum() == \
            pytest.approx(beta_total, rel=1e-10)

        # Every delayed group of both nuclides receives a score, and the
        # effective delayed neutron fraction is close to that of uranium
        assert np.all(mean("nuclide and group", "ifp-beta-numerator") > 0.0)
        beta_eff = beta_total / mean("total", "ifp-denominator")
        assert 0.005 < beta_eff < 0.009


def test_generation_time_only(run_in_tmpdir, geometry):
    """The generation time needs no fission births."""
    model = openmc.Model(geometry=geometry)
    model.settings.particles = 1000
    model.settings.batches = 20
    model.settings.inactive = 5
    model.settings.ifp_n_generation = 5
    tally = openmc.Tally()
    tally.scores = ["ifp-time-numerator", "ifp-denominator"]
    model.tallies = [tally]

    sp_file = model.run()
    with openmc.StatePoint(sp_file) as sp:
        values = sp.tallies[tally.id].mean.ravel()
    assert np.all(values > 0.0)
