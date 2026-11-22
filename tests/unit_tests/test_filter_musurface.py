import math

import pytest

import openmc


def test_musurface(run_in_tmpdir):
    sphere = openmc.Sphere(r=1.0, boundary_type='vacuum')
    cell = openmc.Cell(region=-sphere)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.particles = 1000
    model.settings.batches = 10
    E = 1.0
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.delta_function(E),
    )
    model.settings.run_mode = "fixed source"

    filter1 = openmc.MuSurfaceFilter(200)
    filter2 = openmc.SurfaceFilter(sphere)
    tally = openmc.Tally()
    tally.filters = [filter1, filter2]
    tally.scores = ['current']
    model.tallies = openmc.Tallies([tally])

    # Run OpenMC
    sp_filename = model.run()

    # Get current binned by mu
    with openmc.StatePoint(sp_filename) as sp:
        current_mu = sp.tallies[tally.id].mean.ravel()

    # All contributions should show up in last bin
    assert current_mu[-1] == 1.0
    for element in current_mu[:-1]:
        assert element == 0.0




def _two_cell_model(u):
    """Two void cells split by an x-plane, with a monodirectional source."""
    openmc.reset_auto_ids()

    mid = openmc.XPlane(0.0)
    lo = openmc.XPlane(-10.0, boundary_type='vacuum')
    hi = openmc.XPlane(10.0, boundary_type='vacuum')
    ymin = openmc.YPlane(-5.0, boundary_type='vacuum')
    ymax = openmc.YPlane(5.0, boundary_type='vacuum')
    zmin = openmc.ZPlane(-5.0, boundary_type='vacuum')
    zmax = openmc.ZPlane(5.0, boundary_type='vacuum')
    box = +ymin & -ymax & +zmin & -zmax

    model = openmc.Model()
    model.geometry = openmc.Geometry([
        openmc.Cell(region=+lo & -mid & box),
        openmc.Cell(region=+mid & -hi & box),
    ])

    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((-5.0 * u[0], 0.0, 0.0))
    src.angle = openmc.stats.Monodirectional(u)

    model.settings.run_mode = 'fixed source'
    model.settings.batches = 1
    model.settings.particles = 100
    model.settings.source = src
    return model, mid


def test_musurface_mu_is_signed(run_in_tmpdir):
    """mu carries the sign of the crossing, so both halves of [-1, 1] are used.

    A particle travelling against the surface normal has mu = -1 by the
    definition the bins are documented with. Flipping the normal to face the
    direction of travel would put it at mu = +1 instead, leaving that bin
    disagreeing with the sign of the current scored into it and the whole
    negative half of the range unreachable.
    """
    bins = [-1.0, -0.5, 0.0, 0.5, 1.0]

    model, mid = _two_cell_model((1.0, 0.0, 0.0))
    tally = openmc.Tally()
    tally.filters = [openmc.MuSurfaceFilter(bins), openmc.SurfaceFilter([mid])]
    tally.scores = ['current']
    model.tallies = [tally]
    model.run(apply_tally_results=True)
    # Along the normal: mu = +1, current = +1
    assert tally.mean.ravel() == pytest.approx([0.0, 0.0, 0.0, 1.0])

    model, mid = _two_cell_model((-1.0, 0.0, 0.0))
    tally = openmc.Tally()
    tally.filters = [openmc.MuSurfaceFilter(bins), openmc.SurfaceFilter([mid])]
    tally.scores = ['current']
    model.tallies = [tally]
    model.run(apply_tally_results=True)
    # Against the normal: mu = -1, current = -1
    assert tally.mean.ravel() == pytest.approx([-1.0, 0.0, 0.0, 0.0])


def test_musurface_reflective_surface(run_in_tmpdir):
    """A reflection puts its two halves in mirrored mu bins, not one bin.

    Surface tallies are scored twice at a reflective boundary, once on the way
    in and once on the way back out. With the normal flipped to face the
    direction of travel both scorings landed in the same bin with opposite
    signs and cancelled, reporting no current at any angle.
    """
    openmc.reset_auto_ids()

    refl = openmc.XPlane(-5.0, boundary_type='reflective')
    vac = openmc.XPlane(5.0, boundary_type='vacuum')
    ymin = openmc.YPlane(-5.0, boundary_type='vacuum')
    ymax = openmc.YPlane(5.0, boundary_type='vacuum')
    zmin = openmc.ZPlane(-5.0, boundary_type='reflective')
    zmax = openmc.ZPlane(5.0, boundary_type='reflective')

    model = openmc.Model()
    model.geometry = openmc.Geometry([
        openmc.Cell(region=+refl & -vac & +ymin & -ymax & +zmin & -zmax)])

    # 45 degrees onto the reflective face
    root2 = math.sqrt(0.5)
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((-2.0, 0.0, 0.0))
    src.angle = openmc.stats.Monodirectional((-root2, root2, 0.0))

    model.settings.run_mode = 'fixed source'
    model.settings.batches = 1
    model.settings.particles = 100
    model.settings.source = src

    bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    current = openmc.Tally(name='current')
    current.filters = [openmc.MuSurfaceFilter(bins),
                       openmc.SurfaceFilter([refl])]
    current.scores = ['current']

    flux = openmc.Tally(name='flux')
    flux.filters = [openmc.MuSurfaceFilter(bins), openmc.SurfaceFilter([refl])]
    flux.scores = ['flux']

    model.tallies = [current, flux]
    model.run(apply_tally_results=True)

    # Arriving at mu = +cos(45) and leaving at mu = -cos(45): one crossing in
    # each of the outer bins, with opposite signs, so the net across the
    # reflective surface is zero without the two hiding in a single bin.
    assert current.mean.ravel() == pytest.approx([-1.0, 0.0, 0.0, 1.0])
    assert current.mean.sum() == pytest.approx(0.0)

    # The surface-crossing flux estimator is unsigned, so each half shows up as
    # w/|mu| in its own bin rather than both landing in one
    assert flux.mean.ravel() == pytest.approx(
        [1.0 / root2, 0.0, 0.0, 1.0 / root2])
