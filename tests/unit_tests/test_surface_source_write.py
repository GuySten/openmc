"""Test the 'surf_source_write' setting used to store particles that cross
surfaces in a file for a given simulation."""

from pathlib import Path
import shutil

import openmc
import openmc.lib
import pytest
import h5py
import numpy as np


@pytest.fixture(scope="module")
def geometry():
    """Simple hydrogen sphere geometry"""
    openmc.reset_auto_ids()
    material = openmc.Material(name="H1")
    material.add_element("H", 1.0)
    sphere = openmc.Sphere(r=1.0, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=material)
    return openmc.Geometry([cell])


@pytest.mark.parametrize(
    "parameter",
    [
        {"max_particles": 200},
        {"max_particles": 200, "cell": 1},
        {"max_particles": 200, "cellto": 1},
        {"max_particles": 200, "cellfrom": 1},
        {"max_particles": 200, "surface_ids": [2]},
        {"max_particles": 200, "surface_ids": [2], "cell": 1},
        {"max_particles": 200, "surface_ids": [2], "cellto": 1},
        {"max_particles": 200, "surface_ids": [2], "cellfrom": 1},
        {"max_particles": 200, "surface_ids": [2], "max_source_files": 1},
    ],
)
def test_xml_serialization(parameter, run_in_tmpdir):
    """Check that the different use cases can be written and read in XML."""
    settings = openmc.Settings()
    settings.surf_source_write = parameter
    settings.export_to_xml()

    read_settings = openmc.Settings.from_xml()
    assert read_settings.surf_source_write == parameter


@pytest.fixture(scope="module")
def model():
    """Simple hydrogen sphere geometry"""
    openmc.reset_auto_ids()
    model = openmc.Model()

    # Material
    h1 = openmc.Material(name="H1")
    h1.add_nuclide("H1", 1.0)
    h1.set_density('g/cm3', 1e-7)

    # Geometry
    radius = 1.0
    sphere = openmc.Sphere(r=radius, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere, fill=h1)
    model.geometry = openmc.Geometry([cell])

    # Settings
    model.settings = openmc.Settings()
    model.settings.run_mode = "fixed source"
    model.settings.particles = 100
    model.settings.batches = 3
    model.settings.seed = 1

    distribution = openmc.stats.Point()
    model.settings.source = openmc.IndependentSource(space=distribution)
    return model


@pytest.mark.parametrize(
    "max_particles, max_source_files",
    [
        (100, 2),
        (100, 3),
        (100, 1),
    ],
)
def test_number_surface_source_file_created(max_particles, max_source_files,
                                            run_in_tmpdir, model):
    """Check the number of surface source files written."""
    model.settings.surf_source_write = {
        "max_particles": max_particles,
        "max_source_files": max_source_files
    }
    model.run()
    should_be_numbered = max_source_files > 1
    for i in range(1, max_source_files + 1):
        if should_be_numbered:
            assert Path(f"surface_source.{i}.h5").exists()
    if not should_be_numbered:
        assert Path("surface_source.h5").exists()

ERROR_MSG_1 = (
    "A maximum number of particles needs to be specified "
    "using the 'max_particles' parameter to store surface "
    "source points."
)
ERROR_MSG_2 = "'cell', 'cellfrom' and 'cellto' cannot be used at the same time."


@pytest.mark.parametrize(
    "parameter, error",
    [
        ({"cell": 1}, ERROR_MSG_1),
        ({"max_particles": 200, "cell": 1, "cellto": 1}, ERROR_MSG_2),
        ({"max_particles": 200, "cell": 1, "cellfrom": 1}, ERROR_MSG_2),
        ({"max_particles": 200, "cellto": 1, "cellfrom": 1}, ERROR_MSG_2),
        ({"max_particles": 200, "cell": 1, "cellto": 1, "cellfrom": 1}, ERROR_MSG_2),
    ],
)
def test_exceptions(parameter, error, run_in_tmpdir, geometry):
    """Test parameters configuration that should return an error."""
    settings = openmc.Settings(run_mode="fixed source", batches=5, particles=100)
    settings.surf_source_write = parameter
    model = openmc.Model(geometry=geometry, settings=settings)
    with pytest.raises(RuntimeError, match=error):
        model.run()


@pytest.fixture(scope="module")
def model():
    """Simple hydrogen sphere divided in two hemispheres
    by a z-plane to form 2 cells."""
    openmc.reset_auto_ids()
    model = openmc.Model()

    # Material
    material = openmc.Material(name="H1")
    material.add_element("H", 1.0)

    # Geometry
    radius = 1.0
    sphere = openmc.Sphere(r=radius, boundary_type="reflective")
    plane = openmc.ZPlane(0.0)
    cell_1 = openmc.Cell(region=-sphere & -plane, fill=material)
    cell_2 = openmc.Cell(region=-sphere & +plane, fill=material)
    root = openmc.Universe(cells=[cell_1, cell_2])
    model.geometry = openmc.Geometry(root)

    # Settings
    model.settings = openmc.Settings()
    model.settings.run_mode = "fixed source"
    model.settings.particles = 100
    model.settings.batches = 3
    model.settings.seed = 1

    bounds = [-radius, -radius, -radius, radius, radius, radius]
    distribution = openmc.stats.Box(bounds[:3], bounds[3:])
    model.settings.source = openmc.IndependentSource(space=distribution)

    return model


@pytest.mark.parametrize(
    "parameter",
    [
        {"max_particles": 200, "cellto": 2, "surface_ids": [2]},
        {"max_particles": 200, "cellfrom": 2, "surface_ids": [2]},
    ],
)
def test_particle_direction(parameter, run_in_tmpdir, model):
    """Test the direction of particles with the 'cellfrom' and 'cellto' parameters
    on a simple model with only one surface of interest.

    Cell 2 is the upper hemisphere and surface 2 is the plane dividing the sphere
    into two hemispheres.

    """
    model.settings.surf_source_write = parameter
    model.run()
    with h5py.File("surface_source.h5", "r") as f:
        source = f["source_bank"]

        assert len(source) == 200

        # We want to verify that the dot product of the surface's normal vector
        # and the direction of the particle is either positive or negative
        # depending on cellfrom or cellto. In this case, it is equivalent
        # to just compare the z component of the direction of the particle.
        for point in source:
            if "cellto" in parameter.keys():
                assert point["u"]["z"] > 0.0
            elif "cellfrom" in parameter.keys():
                assert point["u"]["z"] < 0.0
            else:
                assert False


def test_surface_source_id(run_in_tmpdir):
    """Test handling of surface IDs for source particles."""

    # Surface 20 is the surface matching the source in this model. Surface 11 is
    # absent and surface 10 is reused for an unrelated boundary that does not
    # contain the source position.
    sources = []
    for direction in ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)):
        for source_surface_id in (20, 11, 10):
            filename = f"surface_source_{len(sources)}.h5"
            site = openmc.SourceParticle(
                r=(0.0, 0.0, 0.0), u=direction, surf_id=source_surface_id
            )
            openmc.write_source_file([site], filename)
            sources.append(openmc.FileSource(filename))

    left = openmc.XPlane(-1.0, surface_id=1, boundary_type="vacuum")
    middle = openmc.XPlane(0.0, surface_id=20)
    right = openmc.XPlane(1.0, surface_id=30, boundary_type="vacuum")
    bottom = openmc.YPlane(-1.0, surface_id=40, boundary_type="vacuum")
    top = openmc.YPlane(1.0, surface_id=10, boundary_type="vacuum")
    cells = [
        openmc.Cell(region=+left & -middle & +bottom & -top),
        openmc.Cell(region=+middle & -right & +bottom & -top),
    ]

    model = openmc.Model(geometry=openmc.Geometry(cells))
    model.settings.run_mode = "fixed source"
    model.settings.particles = 1000
    model.settings.batches = 1
    model.settings.source = sources
    model.run()


@pytest.fixture
def model_dagmc(request):
    """Model based on the mesh file 'dagmc.h5m' available from
    tests/regression_tests/dagmc/legacy.

    """
    openmc.reset_auto_ids()
    model = openmc.Model()

    # =============================================================================
    # Materials
    # =============================================================================

    u235 = openmc.Material(name="no-void fuel")
    u235.add_nuclide("U235", 1.0, "ao")
    u235.set_density("g/cc", 11)
    u235.id = 40

    water = openmc.Material(name="water")
    water.add_nuclide("H1", 2.0, "ao")
    water.add_nuclide("O16", 1.0, "ao")
    water.set_density("g/cc", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    water.id = 41

    model.materials = openmc.Materials([u235, water])

    # =============================================================================
    # Geometry
    # =============================================================================
    dagmc_path = Path(request.fspath).parent / "../regression_tests/dagmc/legacy/dagmc.h5m"
    dagmc_univ = openmc.DAGMCUniverse(dagmc_path)
    model.geometry = openmc.Geometry(dagmc_univ)

    # =============================================================================
    # Settings
    # =============================================================================

    model.settings = openmc.Settings()
    model.settings.particles = 300
    model.settings.batches = 5
    model.settings.inactive = 1
    model.settings.seed = 1

    source_box = openmc.stats.Box([-4, -4, -20], [4, 4, 20])
    model.settings.source = openmc.IndependentSource(space=source_box)

    return model


@pytest.mark.skipif(
    not openmc.lib.feature_enabled('dagmc'), reason="DAGMC CAD geometry is not enabled."
)
@pytest.mark.parametrize(
    "parameter",
    [
        {"max_particles": 200, "cellto": 1},
        {"max_particles": 200, "cellfrom": 1},
    ],
)
def test_particle_direction_dagmc(parameter, run_in_tmpdir, model_dagmc):
    """Test the direction of particles with the 'cellfrom' and 'cellto' parameters
    on a DAGMC model."""
    model_dagmc.settings.surf_source_write = parameter
    model_dagmc.run()

    r = 7.0
    h = 20.0

    with h5py.File("surface_source.h5", "r") as f:
        source = f["source_bank"]

        assert len(source) == 200

        for point in source:

            x, y, z = point["r"]
            ux, uy, uz = point["u"]

            # If the point is on the upper or lower circle
            if np.allclose(abs(z), h):
                # If the point is also on the cylindrical surface
                if np.allclose(np.sqrt(x**2 + y**2), r):
                    if "cellfrom" in parameter.keys():
                        assert (uz * z > 0) or (ux * x + uy * y > 0)
                    elif "cellto" in parameter.keys():
                        assert (uz * z < 0) or (ux * x + uy * y < 0)
                    else:
                        assert False
                # If the point is not on the cylindrical surface
                else:
                    if "cellfrom" in parameter.keys():
                        assert uz * z > 0
                    elif "cellto" in parameter.keys():
                        assert uz * z < 0
                    else:
                        assert False
            # If the point is not on the upper or lower circle,
            # meaning it is on the cylindrical surface
            else:
                if "cellfrom" in parameter.keys():
                    assert ux * x + uy * y > 0
                elif "cellto" in parameter.keys():
                    assert ux * x + uy * y < 0
                else:
                    assert False


@pytest.fixture
def model_groups():
    """Leaky hydrogen sphere with an isotropic point source at the center."""
    openmc.reset_auto_ids()
    model = openmc.Model()

    h1 = openmc.Material(name="H1")
    h1.add_nuclide("H1", 1.0)
    h1.set_density("g/cm3", 1e-7)

    sphere = openmc.Sphere(r=1.0, boundary_type="vacuum")
    model.geometry = openmc.Geometry([openmc.Cell(region=-sphere, fill=h1)])

    model.settings = openmc.Settings()
    model.settings.run_mode = "fixed source"
    model.settings.particles = 100
    model.settings.batches = 4
    model.settings.seed = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.delta_function(1.0e6),
    )
    return model


@pytest.fixture
def model_cascade():
    """Photon source in lead, producing several escaping particles per history."""
    openmc.reset_auto_ids()
    model = openmc.Model()

    pb = openmc.Material()
    pb.add_element("Pb", 1.0)
    pb.set_density("g/cm3", 11.35)

    sphere = openmc.Sphere(r=0.5, boundary_type="vacuum")
    model.geometry = openmc.Geometry([openmc.Cell(region=-sphere, fill=pb)])

    model.settings = openmc.Settings()
    model.settings.run_mode = "fixed source"
    model.settings.particles = 2000
    model.settings.batches = 2
    model.settings.seed = 1
    model.settings.photon_transport = True
    model.settings.electron_treatment = "ttb"
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.delta_function(6.0e6),
        particle="photon",
    )
    return model


def test_group_and_batch_metadata(run_in_tmpdir, model_groups):
    """Groups partition the sites and batches partition the groups."""
    model_groups.settings.surf_source_write = {"max_particles": 100000}
    model_groups.run()

    with h5py.File("surface_source.h5", "r") as f:
        n_sites = f["source_bank"].shape[0]
        group_offsets = f["group_offsets"][...]
        batch_offsets = f["batch_offsets"][...]
        n_particles = f["batch_n_particles"][...]
        complete = f["batch_complete"][...]
        n_source_particles = f.attrs["n_source_particles"]

    assert n_sites > 0

    # Groups partition the source bank without gaps or overlap, and no group
    # is empty
    assert group_offsets[0] == 0
    assert group_offsets[-1] == n_sites
    assert np.all(np.diff(group_offsets) > 0)

    # Batches partition the groups. One batch per active batch on a single rank
    n_batches = model_groups.settings.batches
    assert len(batch_offsets) == n_batches + 1
    assert batch_offsets[0] == 0
    assert batch_offsets[-1] == len(group_offsets) - 1
    assert np.all(np.diff(batch_offsets) >= 0)

    assert len(n_particles) == len(complete) == n_batches
    assert np.all(n_particles == model_groups.settings.particles)
    assert np.all(complete == 1)
    assert n_source_particles == n_particles.sum()

    # A neutron in a vacuum-bounded sphere with no secondary production can
    # only leak once, so every group holds exactly one site here
    assert len(group_offsets) - 1 == n_sites


def test_group_sorting_is_deterministic(run_in_tmpdir, model_groups):
    """Sorting each generation by history removes the thread-order dependence."""
    model_groups.settings.surf_source_write = {"max_particles": 100000}

    model_groups.run()
    Path("surface_source.h5").rename("first.h5")
    model_groups.run()

    with h5py.File("first.h5") as f1, h5py.File("surface_source.h5") as f2:
        assert np.array_equal(f1["source_bank"][...], f2["source_bank"][...])
        assert np.array_equal(f1["group_offsets"][...], f2["group_offsets"][...])


def test_multi_site_groups(run_in_tmpdir, model_cascade):
    """A history that leaks several particles produces one group, not several."""
    model_cascade.settings.surf_source_write = {"max_particles": 1000000}
    model_cascade.run()

    particles = openmc.read_source_file("surface_source.h5")
    batches = particles.batches
    assert batches is not None

    groups = [group for batch in batches for group in batch]
    sizes = [len(g) for g in groups]

    # Every site belongs to exactly one group of exactly one batch
    assert sum(sizes) == len(particles)
    # Pair production and bremsstrahlung give histories with several escaping
    # photons, so there are strictly fewer groups than sites
    assert len(groups) < len(particles)
    assert max(sizes) > 1

    # A group is a plain particle list, and its sites are the file's own
    first = batches[0][0]
    assert isinstance(first, openmc.ParticleList)
    assert list(first) == list(particles[:len(first)])


def test_structure_is_not_carried_by_slicing(run_in_tmpdir, model_groups):
    """Offsets index the file, so a selection must not claim to carry them."""
    model_groups.settings.surf_source_write = {"max_particles": 100000}
    model_groups.run()

    particles = openmc.read_source_file("surface_source.h5")
    assert particles.batches is not None
    assert particles[:5].batches is None

    # A file with no structure reports none, rather than one trivial group
    # per site
    openmc.write_source_file(particles[:5], "plain_source.h5")
    assert openmc.read_source_file("plain_source.h5").batches is None


def test_split_source_file(run_in_tmpdir, model_groups):
    """Splitting preserves every site and carries the group structure through."""
    model_groups.settings.surf_source_write = {"max_particles": 100000}
    model_groups.run()

    batches = openmc.read_source_file("surface_source.h5").batches
    assert batches is not None
    complete = [b for b in batches if b.complete]

    paths = openmc.split_source_file(
        "surface_source.h5", len(complete), "split")
    assert len(paths) == len(complete)

    total_sites = 0
    for path, batch in zip(paths, complete):
        piece = openmc.read_source_file(path)
        assert len(piece.batches) == 1
        assert piece.batches[0].n_particles == batch.n_particles
        assert [len(g) for g in piece.batches[0]] == [len(g) for g in batch]
        with h5py.File(path, "r") as f:
            assert f.attrs["n_source_particles"] == batch.n_particles
        total_sites += len(piece)

    assert total_sites == sum(len(b.particles) for b in complete)

    # The pieces are usable as sources in their own right
    model_groups.settings.surf_source_write = None
    model_groups.settings.surf_source_read = {"path": str(paths[0])}
    model_groups.run()


def test_truncated_batch(run_in_tmpdir, model_groups):
    """A bank that fills partway through a batch marks that batch incomplete."""
    model_groups.settings.surf_source_write = {"max_particles": 50}
    model_groups.run()

    with h5py.File("surface_source.h5", "r") as f:
        assert f["source_bank"].shape[0] == 50
        complete = f["batch_complete"][...]
        n_particles = f["batch_n_particles"][...]
        n_source_particles = f.attrs["n_source_particles"]

    # The file is written as soon as the bank fills, so the final batch is the
    # truncated one and every earlier batch is intact
    assert complete[-1] == 0
    assert np.all(complete[:-1] == 1)
    assert n_source_particles == n_particles.sum()


def test_split_source_file_no_groups(run_in_tmpdir):
    """Files without group information cannot be split."""
    particles = [openmc.SourceParticle(E=1.0e6) for _ in range(10)]
    openmc.write_source_file(particles, "plain_source.h5")

    with pytest.raises(ValueError, match="no group information"):
        openmc.split_source_file("plain_source.h5", 2)


def test_pulse_height_rejected(run_in_tmpdir, model_groups):
    """Pulse-height tallies cannot be scored from a surface source file."""
    model_groups.settings.surf_source_write = {"max_particles": 100000}
    model_groups.run()

    model_groups.settings.surf_source_write = None
    model_groups.settings.surf_source_read = {"path": "surface_source.h5"}
    cells = list(model_groups.geometry.get_all_cells().values())
    tally = openmc.Tally()
    tally.filters = [
        openmc.CellFilter([c.id for c in cells]),
        openmc.EnergyFilter(np.linspace(0.0, 2.0e6, 11)),
    ]
    tally.scores = ["pulse-height"]
    model_groups.tallies = openmc.Tallies([tally])

    with pytest.raises(RuntimeError, match="surface source file"):
        model_groups.run()


@pytest.mark.skipif(
    shutil.which("mcpl-config") is None, reason="MCPL is not available."
)
def test_mcpl_group_userflags(run_in_tmpdir, model_cascade):
    """MCPL files carry the group index in the user-flags field."""
    mcpl = pytest.importorskip("mcpl")

    model_cascade.settings.surf_source_write = {
        "max_particles": 1000000,
        "mcpl": True,
    }
    model_cascade.run()

    with mcpl.MCPLFile("surface_source.mcpl") as f:
        assert f.opt_userflags
        flags = np.concatenate([b.userflags for b in f.particle_blocks])
        stat_sums = f.stat_sums

    # Group indices are dense and non-decreasing, and at least one group holds
    # more than one site
    assert flags[0] == 0
    steps = np.diff(flags.astype(np.int64))
    assert np.all((steps == 0) | (steps == 1))
    assert np.any(steps == 0)

    # The same structure comes back through the public reader
    particles = openmc.ParticleList.from_mcpl("surface_source.mcpl")
    batches = particles.batches
    assert batches is not None
    assert sum(len(g) for b in batches for g in b) == len(particles)
    assert stat_sums["openmc_np1"] == batches.n_source_particles

    # And matches what the HDF5 writer produces for the same problem
    model_cascade.settings.surf_source_write = {"max_particles": 1000000}
    model_cascade.run()
    h5_batches = openmc.read_source_file("surface_source.h5").batches
    assert [len(b) for b in h5_batches] == [len(b) for b in batches]
