"""Test that a surface source file recording which sites came from the same
source history is read back one history per group, not one history per site."""

import shutil

import h5py
import numpy as np
import openmc
import pytest


@pytest.fixture
def scatter_lib(run_in_tmpdir):
    """One-group purely scattering data, so a history crosses the recording
    surface several times before it leaks."""
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata("scatter", groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.0])
    xsdata.set_scatter_matrix(np.array([[[1.0]]]))

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdata(xsdata)
    lib.export_to_hdf5("scatter.h5")
    return "scatter.h5"


@pytest.fixture
def absorb_lib():
    """One-group purely absorbing data, so every emitted site deposits its
    whole weight and the absorption rate counts sites."""
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata("absorb", groups)
    xsdata.order = 0
    xsdata.set_total([100.0])
    xsdata.set_absorption([100.0])
    xsdata.set_scatter_matrix(np.array([[[0.0]]]))

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdata(xsdata)
    lib.export_to_hdf5("absorb.h5")
    return "absorb.h5"


def _writer_model(lib, macroscopic="scatter"):
    """Ball with an interior recording surface."""
    openmc.reset_auto_ids()
    mat = openmc.Material()
    mat.set_density("macro", 1.0)
    mat.add_macroscopic(macroscopic)

    model = openmc.Model()
    model.materials = openmc.Materials([mat])
    model.materials.cross_sections = lib

    inner = openmc.Sphere(r=1.0)
    outer = openmc.Sphere(r=5.0, boundary_type="vacuum")
    c_in = openmc.Cell(fill=mat, region=-inner)
    c_out = openmc.Cell(fill=mat, region=+inner & -outer)
    model.geometry = openmc.Geometry([c_in, c_out])

    model.settings = openmc.Settings()
    model.settings.energy_mode = "multi-group"
    model.settings.run_mode = "fixed source"
    model.settings.particles = 500
    model.settings.batches = 4
    model.settings.seed = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), angle=openmc.stats.Isotropic()
    )
    model.settings.surf_source_write = {
        "surface_ids": [inner.id],
        "max_particles": 100000,
    }
    return model


def _reader_model(lib, path):
    """Strong absorber covering the recording surface, so the absorption rate
    per source particle equals the number of sites emitted per history."""
    openmc.reset_auto_ids()
    mat = openmc.Material()
    mat.set_density("macro", 1.0)
    mat.add_macroscopic("absorb")

    model = openmc.Model()
    model.materials = openmc.Materials([mat])
    model.materials.cross_sections = lib

    outer = openmc.Sphere(r=10.0, boundary_type="vacuum")
    model.geometry = openmc.Geometry([openmc.Cell(fill=mat, region=-outer)])

    model.settings = openmc.Settings()
    model.settings.energy_mode = "multi-group"
    model.settings.run_mode = "fixed source"
    model.settings.particles = 2000
    model.settings.batches = 5
    model.settings.seed = 7
    model.settings.surf_source_read = {"path": path}

    tally = openmc.Tally(name="absorption")
    tally.scores = ["absorption"]
    model.tallies = openmc.Tallies([tally])
    return model


def _absorption_per_source_particle(model):
    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        return sp.get_tally(name="absorption").mean.flatten()[0]


def _strip_groups(src, dst):
    """Copy a surface source file without its history grouping."""
    shutil.copy(src, dst)
    with h5py.File(dst, "a") as fh:
        for key in (
            "group_offsets",
            "batch_offsets",
            "batch_n_particles",
            "batch_complete",
        ):
            if key in fh:
                del fh[key]


def test_grouped_read_emits_whole_history(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """A group is emitted as one history, so a history emits all of its sites."""
    _writer_model(scatter_lib).run()

    particles = openmc.read_source_file("surface_source.h5")
    sizes = [len(g) for b in particles.batches for g in b]
    mean_group_size = np.mean(sizes)

    # The file has to contain multi-site histories for this test to mean
    # anything at all
    assert max(sizes) > 1
    assert mean_group_size > 1.05

    grouped = _absorption_per_source_particle(
        _reader_model(absorb_lib, "surface_source.h5")
    )

    # Every emitted site is absorbed with its full weight, so the absorption
    # rate per source particle counts the sites emitted per history
    assert grouped == pytest.approx(mean_group_size, rel=0.05)


def test_ungrouped_read_is_one_site_per_history(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """Without the grouping the same file is still read one site per history."""
    _writer_model(scatter_lib).run()
    _strip_groups("surface_source.h5", "flat_source.h5")

    assert openmc.read_source_file("flat_source.h5").batches is None

    flat = _absorption_per_source_particle(
        _reader_model(absorb_lib, "flat_source.h5")
    )
    assert flat == pytest.approx(1.0, rel=0.02)


def test_invalid_group_offsets_rejected(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """A grouping that does not partition the sites is refused up front."""
    _writer_model(scatter_lib).run()

    with h5py.File("surface_source.h5", "a") as fh:
        offsets = fh["group_offsets"][...]
        del fh["group_offsets"]
        # Break the trailing total so the groups no longer cover the sites
        offsets[-1] -= 1
        fh.create_dataset("group_offsets", data=offsets)

    with pytest.raises(RuntimeError, match="does not partition"):
        _reader_model(absorb_lib, "surface_source.h5").run()


@pytest.fixture
def leaky_lib():
    """Absorbing data, so many source particles never reach the surface and the
    group count falls below the first-stage source particle count."""
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata("leaky", groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.5])
    xsdata.set_scatter_matrix(np.array([[[0.5]]]))

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdata(xsdata)
    lib.export_to_hdf5("leaky.h5")
    return "leaky.h5"


def test_normalization_to_first_stage(run_in_tmpdir, leaky_lib, absorb_lib):
    """The documented factor converts a per-group tally to a per-first-stage
    -source-particle one."""
    _writer_model(leaky_lib, macroscopic="leaky").run()

    particles = openmc.read_source_file("surface_source.h5")
    n_sites = len(particles)
    n_groups = len(particles.groups)
    n_src = particles.n_source_particles

    # The case worth testing is the one where not every source particle
    # produced a group
    assert n_src is not None
    assert n_groups < n_src

    per_group = _absorption_per_source_particle(
        _reader_model(absorb_lib, "surface_source.h5")
    )

    # Every site is absorbed with its full weight, so the response of the whole
    # file per first-stage source particle is just the site count over it
    truth = n_sites / n_src
    assert per_group * n_groups / n_src == pytest.approx(truth, rel=0.05)


def _tally_with(lib, path, batches, particles, independent):
    model = _reader_model(lib, path)
    model.settings.batches = batches
    model.settings.particles = particles
    ssr = {"path": str(path)}
    if independent:
        ssr["independent_batches"] = True
    model.settings.surf_source_read = ssr
    sp = model.run()
    with openmc.StatePoint(sp) as s:
        t = s.get_tally(name="absorption")
        return float(t.mean.flatten()[0]), float(t.std_dev.flatten()[0])


def test_independent_batches_uncertainty_does_not_collapse(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """The first stage's sampling error cannot be reduced by working harder in
    the second, so a correct uncertainty must stop shrinking. Sharing every
    history between all batches hides that floor; partitioning them exposes it.
    """
    _writer_model(scatter_lib).run()

    lo, hi = 500, 8000
    mean_i_lo, std_i_lo = _tally_with(absorb_lib, "surface_source.h5", 30, lo, True)
    mean_i_hi, std_i_hi = _tally_with(absorb_lib, "surface_source.h5", 30, hi, True)
    mean_p_lo, std_p_lo = _tally_with(absorb_lib, "surface_source.h5", 30, lo, False)
    mean_p_hi, std_p_hi = _tally_with(absorb_lib, "surface_source.h5", 30, hi, False)

    # Sixteen times the second-stage effort. Sharing the whole file between
    # batches lets the reported uncertainty fall away as if the first stage
    # were exact
    assert std_p_hi < 0.5 * std_p_lo

    # Partitioning the file leaves it on the floor set by the first stage
    assert 0.5 < std_i_hi / std_i_lo < 2.0
    assert std_i_hi > 3 * std_p_hi

    # None of this moves the answer: every group still gets an equal chance
    for m in (mean_i_lo, mean_i_hi, mean_p_hi):
        assert m == pytest.approx(mean_p_lo, rel=0.05)


def test_independent_batches_requires_grouping(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """A file whose correlation structure is unknown cannot be partitioned."""
    _writer_model(scatter_lib).run()
    _strip_groups("surface_source.h5", "flat_source.h5")

    with pytest.raises(RuntimeError, match="independent_batches requires"):
        _tally_with(absorb_lib, "flat_source.h5", 30, 500, True)


def test_independent_batches_needs_two_batches(
    run_in_tmpdir, scatter_lib, absorb_lib
):
    """A spread between batches needs more than one batch."""
    _writer_model(scatter_lib).run()

    with pytest.raises(RuntimeError, match="at least two active batches"):
        _tally_with(absorb_lib, "surface_source.h5", 1, 500, True)
