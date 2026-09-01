"""Tests for the deterministic octree volume calculation method.

The central invariant is the bracket property:

    lower <= V_true <= lower + slack

reported through Result.volume as (midpoint, half_width). This must hold at
*every* tolerance and depth setting, including deliberately coarse ones. A
failure at max_depth=2 means a surface classifier is unsound, and refinement
will not fix it.
"""

import math
import os

import numpy as np
import pytest

import openmc
import openmc.lib


def brackets(vol_calc_file, domain_ids):
    """Return {id: (lower, upper)} from a volume HDF5 file."""
    vc = openmc.VolumeCalculation.from_hdf5(vol_calc_file)
    out = {}
    for uid in domain_ids:
        mid, half = vc.volumes[uid].n, vc.volumes[uid].s
        out[uid] = (mid - half, mid + half)
    return out


@pytest.fixture
def sphere_model():
    """A sphere of radius 5 inside a 20 cm cube of a second material."""
    inner = openmc.Material(name='inner')
    inner.add_nuclide('U235', 1.0)
    inner.set_density('g/cm3', 10.0)

    outer = openmc.Material(name='outer')
    outer.add_nuclide('H1', 1.0)
    outer.set_density('g/cm3', 1.0)

    sph = openmc.Sphere(r=5.0)
    box = openmc.model.RectangularParallelepiped(
        -10., 10., -10., 10., -10., 10., boundary_type='vacuum')

    c_in = openmc.Cell(cell_id=1, fill=inner, region=-sph)
    c_out = openmc.Cell(cell_id=2, fill=outer, region=+sph & -box)

    model = openmc.Model()
    model.geometry = openmc.Geometry([c_in, c_out])
    model.materials = openmc.Materials([inner, outer])
    model.settings.particles = 100
    model.settings.batches = 10
    model.settings.run_mode = 'volume'
    return model


def test_sphere_bracket(sphere_model, run_in_tmpdir):
    """Analytic volume must fall inside the bracket at every tolerance."""
    exact_in = 4.0 / 3.0 * math.pi * 5.0 ** 3
    exact_out = 20.0 ** 3 - exact_in

    for tol in (1.0, 1.0e-1, 1.0e-2):
        vc = openmc.VolumeCalculation(
            [sphere_model.geometry.get_all_cells()[1],
             sphere_model.geometry.get_all_cells()[2]],
            method='octree', tolerance=tol,
            lower_left=(-10., -10., -10.), upper_right=(10., 10., 10.))
        sphere_model.settings.volume_calculations = [vc]
        sphere_model.export_to_model_xml()
        openmc.run()

        b = brackets('volume_1.h5', [1, 2])
        assert b[1][0] <= exact_in <= b[1][1]
        assert b[2][0] <= exact_out <= b[2][1]
        # the bracket must actually tighten
        assert (b[1][1] - b[1][0]) <= 2 * tol + 1e-12


def test_sphere_unsound_classifier_canary(sphere_model, run_in_tmpdir):
    """The bracket must hold even when refinement is crippled.

    This is the test that catches an unsound surface classifier. With
    max_depth=2 the answer is worthless but it must still be a valid bound.
    """
    exact_in = 4.0 / 3.0 * math.pi * 5.0 ** 3

    vc = openmc.VolumeCalculation(
        [sphere_model.geometry.get_all_cells()[1]],
        method='octree', tolerance=1.0e-6, max_depth=2,
        lower_left=(-10., -10., -10.), upper_right=(10., 10., 10.))
    sphere_model.settings.volume_calculations = [vc]
    sphere_model.export_to_model_xml()
    openmc.run()

    lo, hi = brackets('volume_1.h5', [1])[1]
    assert lo <= exact_in <= hi


@pytest.fixture
def pincell_model():
    """Fuel / gap / clad / moderator, all analytic."""
    mats = []
    for name, nuc in (('fuel', 'U235'), ('gap', 'He4'),
                      ('clad', 'Zr90'), ('water', 'H1')):
        m = openmc.Material(name=name)
        m.add_nuclide(nuc, 1.0)
        m.set_density('g/cm3', 1.0)
        mats.append(m)
    fuel, gap, clad, water = mats

    r_f, r_g, r_c = 0.39218, 0.40005, 0.45720
    pitch, half_z = 1.26, 5.0

    c1 = openmc.ZCylinder(r=r_f)
    c2 = openmc.ZCylinder(r=r_g)
    c3 = openmc.ZCylinder(r=r_c)
    box = openmc.model.RectangularParallelepiped(
        -pitch / 2, pitch / 2, -pitch / 2, pitch / 2, -half_z, half_z,
        boundary_type='reflective')

    cells = [
        openmc.Cell(cell_id=1, fill=fuel, region=-c1 & -box),
        openmc.Cell(cell_id=2, fill=gap, region=+c1 & -c2 & -box),
        openmc.Cell(cell_id=3, fill=clad, region=+c2 & -c3 & -box),
        openmc.Cell(cell_id=4, fill=water, region=+c3 & -box),
    ]

    model = openmc.Model()
    model.geometry = openmc.Geometry(cells)
    model.materials = openmc.Materials(mats)
    model.settings.run_mode = 'volume'
    model._analytic = {
        1: math.pi * r_f ** 2 * 2 * half_z,
        2: math.pi * (r_g ** 2 - r_f ** 2) * 2 * half_z,
        3: math.pi * (r_c ** 2 - r_g ** 2) * 2 * half_z,
        4: (pitch ** 2 - math.pi * r_c ** 2) * 2 * half_z,
    }
    return model


def test_pincell_bracket(pincell_model, run_in_tmpdir):
    cells = pincell_model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[i] for i in (1, 2, 3, 4)],
        method='octree', tolerance=1.0e-3,
        lower_left=(-0.63, -0.63, -5.0), upper_right=(0.63, 0.63, 5.0))
    pincell_model.settings.volume_calculations = [vc]
    pincell_model.export_to_model_xml()
    openmc.run()

    b = brackets('volume_1.h5', [1, 2, 3, 4])
    for uid, exact in pincell_model._analytic.items():
        lo, hi = b[uid]
        assert lo <= exact <= hi, f'cell {uid}: {exact} not in [{lo}, {hi}]'

    # The thin gap is the hard case: it is a shell 0.008 cm thick, so almost
    # every box that touches it straddles. Expect a loose but valid bracket,
    # and expect it to be the loosest of the four.
    widths = {uid: b[uid][1] - b[uid][0] for uid in (1, 2, 3, 4)}
    assert widths[2] == max(widths.values())


def test_agrees_with_stochastic(pincell_model, run_in_tmpdir):
    """Octree bracket and stochastic confidence interval must overlap."""
    cells = pincell_model.geometry.get_all_cells()
    ll, ur = (-0.63, -0.63, -5.0), (0.63, 0.63, 5.0)
    domains = [cells[i] for i in (1, 2, 3, 4)]

    pincell_model.settings.volume_calculations = [
        openmc.VolumeCalculation(domains, method='octree', tolerance=1.0e-3,
                                 lower_left=ll, upper_right=ur),
        openmc.VolumeCalculation(domains, samples=10_000_000,
                                 lower_left=ll, upper_right=ur),
    ]
    pincell_model.export_to_model_xml()
    openmc.run()

    det = brackets('volume_1.h5', [1, 2, 3, 4])
    sto = openmc.VolumeCalculation.from_hdf5('volume_2.h5')
    for uid in (1, 2, 3, 4):
        v = sto.volumes[uid]
        lo_s, hi_s = v.n - 4 * v.s, v.n + 4 * v.s
        lo_d, hi_d = det[uid]
        assert lo_d <= hi_s and lo_s <= hi_d, f'cell {uid} disagrees'


# =============================================================================
# v2: rotated fills
# =============================================================================


def _rotated_model(angles):
    """A sphere inside a universe, that universe placed with a rotation.

    The volume is rotation-invariant, so the answer must not depend on `angles`.
    That invariance is the whole test: it catches a transposed rotation matrix,
    the wrong sign convention on the translation, and any frame handling that
    silently loses or duplicates volume.
    """
    inner = openmc.Material()
    inner.add_nuclide('U235', 1.0)
    inner.set_density('g/cm3', 10.0)
    outer = openmc.Material()
    outer.add_nuclide('H1', 1.0)
    outer.set_density('g/cm3', 1.0)

    sph = openmc.Sphere(r=3.0)
    cyl = openmc.ZCylinder(r=4.0)
    zlo, zhi = openmc.ZPlane(-4.0), openmc.ZPlane(4.0)

    u_in = openmc.Cell(cell_id=1, fill=inner, region=-sph)
    u_out = openmc.Cell(cell_id=2, fill=outer,
                        region=+sph & -cyl & +zlo & -zhi)
    univ = openmc.Universe(universe_id=10, cells=[u_in, u_out])

    box = openmc.model.RectangularParallelepiped(
        -10., 10., -10., 10., -10., 10., boundary_type='vacuum')
    holder = openmc.Cell(cell_id=3, fill=univ, region=-box)
    holder.rotation = angles

    model = openmc.Model()
    model.geometry = openmc.Geometry([holder])
    model.materials = openmc.Materials([inner, outer])
    model.settings.run_mode = 'volume'
    return model


def test_reproducible_across_thread_counts(pincell_model, run_in_tmpdir):
    """The same calculation must agree between 1 and 4 threads.

    The merge sums in nondeterministic order, so results are not bitwise
    identical. Measured, reordering moves a volume by ~1e-13 relative even at
    1e8 contributions, which is invisible at any sensible output precision --
    but that is a measurement, not a guarantee, so assert it here. If a future
    change introduces real nondeterminism (a hash-order dependence, a race)
    this fails loudly instead of showing up as an intermittent regression
    diff.

    1e-9 is deliberately far looser than the measured spread and far tighter
    than anything that could hide a genuine bug.
    """
    cells = pincell_model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[i] for i in (1, 2, 3, 4)],
        method='octree', tolerance=1.0e-4,
        lower_left=(-0.63, -0.63, -5.0), upper_right=(0.63, 0.63, 5.0))
    pincell_model.settings.volume_calculations = [vc]
    pincell_model.export_to_model_xml()

    runs = {}
    saved = os.environ.get('OMP_NUM_THREADS')
    try:
        for n_threads in (1, 4):
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            openmc.run()
            res = openmc.VolumeCalculation.from_hdf5('volume_1.h5')
            runs[n_threads] = {uid: res.volumes[uid].n for uid in (1, 2, 3, 4)}
    finally:
        if saved is None:
            os.environ.pop('OMP_NUM_THREADS', None)
        else:
            os.environ['OMP_NUM_THREADS'] = saved

    for uid in (1, 2, 3, 4):
        a, b = runs[1][uid], runs[4][uid]
        assert abs(a - b) <= 1.0e-9 * max(abs(a), abs(b)), (
            f'cell {uid}: {a} vs {b} across thread counts')


@pytest.mark.parametrize('angles', [
    (0., 0., 0.),
    (0., 0., 30.),
    (15., 40., 75.),
    (90., 90., 90.),
])
def test_rotated_fill_matches_unrotated(angles, run_in_tmpdir):
    """Volume is rotation-invariant; the bracket must contain the truth."""
    exact = 4.0 / 3.0 * math.pi * 3.0 ** 3

    model = _rotated_model(angles)
    cells = model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[1]], method='octree', tolerance=1.0e-2,
        lower_left=(-10., -10., -10.), upper_right=(10., 10., 10.))
    model.settings.volume_calculations = [vc]
    model.export_to_model_xml()
    openmc.run()

    lo, hi = brackets('volume_1.h5', [1])[1]
    assert lo <= exact <= hi, f'rotation {angles}: {exact} not in [{lo}, {hi}]'


def test_rotated_fill_conserves_total(run_in_tmpdir):
    """Inner + outer must sum to the universe volume regardless of rotation.

    A frame bug that enlarges the child box would inflate this sum; one that
    shrinks it would lose volume. Neither shows up in a single-cell check.
    """
    model = _rotated_model((15., 40., 75.))
    cells = model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[1], cells[2]], method='octree', tolerance=1.0e-2,
        lower_left=(-10., -10., -10.), upper_right=(10., 10., 10.))
    model.settings.volume_calculations = [vc]
    model.export_to_model_xml()
    openmc.run()

    b = brackets('volume_1.h5', [1, 2])
    exact_total = math.pi * 4.0 ** 2 * 8.0  # the cylinder holding both
    lo = b[1][0] + b[2][0]
    hi = b[1][1] + b[2][1]
    assert lo <= exact_total <= hi


# =============================================================================
# v2: hexagonal lattices
# =============================================================================


@pytest.fixture
def hex_model():
    """Single-ring hex lattice, 7 elements, each a pin in water."""
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 10.0)
    water = openmc.Material()
    water.add_nuclide('H1', 1.0)
    water.set_density('g/cm3', 1.0)

    r_pin, pitch, half_z = 0.4, 1.3, 5.0

    pin_cyl = openmc.ZCylinder(r=r_pin)
    pin_f = openmc.Cell(cell_id=1, fill=fuel, region=-pin_cyl)
    pin_w = openmc.Cell(cell_id=2, fill=water, region=+pin_cyl)
    pin = openmc.Universe(universe_id=10, cells=[pin_f, pin_w])

    w_cell = openmc.Cell(cell_id=3, fill=water)
    outer = openmc.Universe(universe_id=11, cells=[w_cell])

    lat = openmc.HexLattice(lattice_id=100)
    lat.center = (0., 0.)
    lat.pitch = (pitch,)
    lat.orientation = 'y'
    lat.universes = [[pin] * 6, [pin]]
    lat.outer = outer

    bound = openmc.model.RectangularParallelepiped(
        -4., 4., -4., 4., -half_z, half_z, boundary_type='vacuum')
    root = openmc.Cell(cell_id=4, fill=lat, region=-bound)

    model = openmc.Model()
    model.geometry = openmc.Geometry([root])
    model.materials = openmc.Materials([fuel, water])
    model.settings.run_mode = 'volume'
    model._n_pins = 7
    model._pin_volume = math.pi * r_pin ** 2 * 2 * half_z
    return model


def test_hex_lattice_bracket(hex_model, run_in_tmpdir):
    """Seven pins, all fully inside the bounding box, so the total is exact."""
    exact = hex_model._n_pins * hex_model._pin_volume

    cells = hex_model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[1]], method='octree', tolerance=1.0e-2,
        lower_left=(-4., -4., -5.), upper_right=(4., 4., 5.))
    hex_model.settings.volume_calculations = [vc]
    hex_model.export_to_model_xml()
    openmc.run()

    lo, hi = brackets('volume_1.h5', [1])[1]
    assert lo <= exact <= hi, f'{exact} not in [{lo}, {hi}]'


@pytest.mark.parametrize('orientation', ['x', 'y'])
def test_hex_orientation_independent(orientation, hex_model, run_in_tmpdir):
    """Both hex orientations must work.

    resolve_lattice() makes no orientation assumption -- it only asks whether
    all eight box corners land in the same element -- so this is really a check
    that nothing snuck a convention in.
    """
    hex_model.geometry.get_all_lattices()[100].orientation = orientation
    exact = hex_model._n_pins * hex_model._pin_volume

    cells = hex_model.geometry.get_all_cells()
    vc = openmc.VolumeCalculation(
        [cells[1]], method='octree', tolerance=1.0e-2,
        lower_left=(-4., -4., -5.), upper_right=(4., 4., 5.))
    hex_model.settings.volume_calculations = [vc]
    hex_model.export_to_model_xml()
    openmc.run()

    lo, hi = brackets('volume_1.h5', [1])[1]
    assert lo <= exact <= hi


def test_hex_outer_universe(hex_model, run_in_tmpdir):
    """Water fills the lattice interstitium and everything outside the ring.

    Exercises the branch where all eight corners agree on an index that is
    outside the ring bound, which must route to lat.outer_ rather than being
    treated as an invalid element.
    """
    exact_total = 8.0 * 8.0 * 10.0
    exact_water = exact_total - hex_model._n_pins * hex_model._pin_volume

    mats = hex_model.materials
    vc = openmc.VolumeCalculation(
        [mats[1]], method='octree', tolerance=1.0e-1,
        lower_left=(-4., -4., -5.), upper_right=(4., 4., 5.))
    hex_model.settings.volume_calculations = [vc]
    hex_model.export_to_model_xml()
    openmc.run()

    vcr = openmc.VolumeCalculation.from_hdf5('volume_1.h5')
    v = vcr.volumes[mats[1].id]
    assert v.n - v.s <= exact_water <= v.n + v.s


# =============================================================================
# v2: tori
# =============================================================================


def test_torus_bracket(run_in_tmpdir):
    """Analytic torus volume 2 pi^2 A B C for the elliptical form."""
    m = openmc.Material()
    m.add_nuclide('H1', 1.0)
    m.set_density('g/cm3', 1.0)

    A, B, C = 5.0, 1.0, 1.5
    tor = openmc.ZTorus(x0=0., y0=0., z0=0., a=A, b=B, c=C)
    box = openmc.model.RectangularParallelepiped(
        -8., 8., -8., 8., -4., 4., boundary_type='vacuum')

    c1 = openmc.Cell(cell_id=1, fill=m, region=-tor)
    c2 = openmc.Cell(cell_id=2, region=+tor & -box)

    model = openmc.Model()
    model.geometry = openmc.Geometry([c1, c2])
    model.materials = openmc.Materials([m])
    model.settings.run_mode = 'volume'

    vc = openmc.VolumeCalculation(
        [c1], method='octree', tolerance=1.0e-1,
        lower_left=(-8., -8., -4.), upper_right=(8., 8., 4.))
    model.settings.volume_calculations = [vc]
    model.export_to_model_xml()
    openmc.run()

    exact = 2.0 * math.pi ** 2 * A * B * C
    lo, hi = brackets('volume_1.h5', [1])[1]
    assert lo <= exact <= hi, f'{exact} not in [{lo}, {hi}]'
