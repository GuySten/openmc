from pathlib import Path

import h5py
import numpy as np
import openmc
import pytest

from tests.testing_harness import config


@pytest.fixture
def sphere_model():
    openmc.reset_auto_ids()
    mat = openmc.Material()
    mat.add_nuclide('Zr90', 1.0)
    mat.set_density('g/cm3', 1.0)

    model = openmc.Model()
    sph = openmc.Sphere(r=25.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sph)
    model.geometry = openmc.Geometry([cell])

    model.settings.run_mode = 'fixed source'
    model.settings.batches = 2
    model.settings.particles = 50

    return model


def generate_track_file(model, **kwargs):
    # If running in MPI mode, setup proper keyword arguments for run()
    kwargs.setdefault('openmc_exec', config['exe'])
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    model.run(**kwargs)

    if config['mpi'] and int(config['mpi_np']) > 1:
        # With MPI, we need to combine track files
        track_files = Path.cwd().glob('tracks_p*.h5')
        openmc.Tracks.combine(track_files, 'tracks.h5')
    else:
        track_file = Path('tracks.h5')
        assert track_file.is_file()


@pytest.mark.parametrize("particle", ["neutron", "photon"])
def test_tracks(sphere_model, particle, run_in_tmpdir):
    # Set track identifiers
    sphere_model.settings.track = [(1, 1, 1), (1, 1, 10), (2, 1, 15)]

    # Set source particle
    sphere_model.settings.source = openmc.IndependentSource(particle=particle)

    # Run OpenMC to generate tracks.h5 file
    generate_track_file(sphere_model)

    # Open track file and make sure we have correct number of tracks
    tracks = openmc.Tracks('tracks.h5')
    assert len(tracks) == len(sphere_model.settings.track)

    for track, identifier in zip(tracks, sphere_model.settings.track):
        # Check attributes on Track object
        assert isinstance(track, openmc.Track)
        assert track.identifier == identifier
        assert isinstance(track.particle_tracks, list)
        if particle == 'neutron':
            assert len(track.particle_tracks) == 1

        # Check attributes on ParticleTrack object
        particle_track = track.particle_tracks[0]
        assert isinstance(particle_track, openmc.ParticleTrack)

        assert str(particle_track.particle).lower() == particle
        assert isinstance(particle_track.states, np.ndarray)

        # Sanity checks on actual data
        for state in particle_track.states:
            assert np.linalg.norm([*state['r']]) <= 25.0001
            assert np.linalg.norm([*state['u']]) == pytest.approx(1.0)
            assert 0.0 <= state['E'] <= 20.0e6
            assert state['time'] >= 0.0
            assert 0.0 <= state['wgt'] <= 1.0
            assert state['cell_id'] == 1
            assert state['material_id'] == 1

        # Checks on 'sources' property
        sources = track.sources
        assert len(sources) == len(track.particle_tracks)
        x = sources[0]
        state = particle_track.states[0]
        assert x.r == (*state['r'],)
        assert x.u == (*state['u'],)
        assert x.E == state['E']
        assert x.time == state['time']
        assert x.wgt == state['wgt']
        assert x.particle == particle_track.particle


def test_max_tracks(sphere_model, run_in_tmpdir):
    # Set maximum number of tracks per process to write
    sphere_model.settings.max_tracks = expected_num_tracks = 10
    if config['mpi']:
        expected_num_tracks *= int(config['mpi_np'])

    # Run OpenMC to generate tracks.h5 file
    generate_track_file(sphere_model, tracks=True)

    # Open track file and make sure we have correct number of tracks
    tracks = openmc.Tracks('tracks.h5')
    assert len(tracks) == expected_num_tracks


def test_filter(sphere_model, run_in_tmpdir):
    # Set maximum number of tracks per process to write
    sphere_model.settings.max_tracks = 25
    sphere_model.settings.photon_transport = True

    # Run OpenMC to generate tracks.h5 file
    generate_track_file(sphere_model, tracks=True)

    tracks = openmc.Tracks('tracks.h5')
    for track in tracks:
        # Test filtering by particle
        matches = track.filter(particle='photon')
        for x in matches:
            assert x.particle == openmc.ParticleType.PHOTON

        # Test general state filter
        matches = track.filter(state_filter=lambda s: s['cell_id'] == 1)
        assert isinstance(matches, openmc.Track)
        assert matches.particle_tracks == track.particle_tracks
        matches = track.filter(state_filter=lambda s: s['cell_id'] == 2)
        assert matches.particle_tracks == []
        matches = track.filter(state_filter=lambda s: s['E'] < 0.0)
        assert matches.particle_tracks == []

    # Test filter method on Tracks
    matches = tracks.filter(particle='neutron')
    assert isinstance(matches, openmc.Tracks)
    assert matches == tracks
    matches = tracks.filter(state_filter=lambda s: s['E'] > 0.0)
    assert matches == tracks
    matches = tracks.filter(particle='proton')
    assert matches == []
    with pytest.raises(ValueError):
        tracks.filter(particle='bunnytron')


def test_write_to_vtk(sphere_model):
    vtk = pytest.importorskip('vtk')
    # Set maximum number of tracks per process to write
    sphere_model.settings.max_tracks = 25
    sphere_model.settings.photon_transport = True

    # Run OpenMC to generate tracks.h5 file
    generate_track_file(sphere_model, tracks=True)

    tracks = openmc.Tracks('tracks.h5')
    polydata = tracks.write_to_vtk('tracks.vtp')

    assert isinstance(polydata, vtk.vtkPolyData)
    assert Path('tracks.vtp').is_file()


def test_restart_track(run_in_tmpdir, sphere_model):
    # cut the sphere model in half with an improper boundary condition
    plane = openmc.XPlane(x0=-1.0)
    for cell in sphere_model.geometry.get_all_cells().values():
        cell.region &= +plane

    # generate lost particle files
    with pytest.raises(RuntimeError, match='Maximum number of lost particles has been reached.'):
        sphere_model.run(output=False, threads=1)

    lost_particle_files = list(Path.cwd().glob('particle_*.h5'))
    assert len(lost_particle_files) > 0
    particle_file = lost_particle_files[0]
     # restart the lost particle with tracks enabled
    sphere_model.run(tracks=True, restart_file=particle_file)
    tracks_file = Path('tracks.h5')
    assert tracks_file.is_file()

    # check that the last track of the file matches the lost particle file
    tracks = openmc.Tracks(tracks_file)
    initial_state = tracks[0].particle_tracks[0].states[0]
    restart_r = np.array(initial_state['r'])
    restart_u = np.array(initial_state['u'])

    with h5py.File(particle_file, 'r') as lost_particle_file:
        lost_r = np.array(lost_particle_file['xyz'][()])
        lost_u = np.array(lost_particle_file['uvw'][()])

    pytest.approx(restart_r, lost_r)
    pytest.approx(restart_u, lost_u)


def _split_model():
    """Model whose source particles are each split into five by a weight window
    at a surface checkpoint, so that the copies are transported from the shared
    secondary bank"""
    openmc.reset_auto_ids()

    # Void-like one-group material so the only weight window checks happen at
    # surface crossings
    groups = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    xsdata = openmc.XSdata('void', groups)
    xsdata.order = 0
    xsdata.set_total([0.0])
    xsdata.set_absorption([0.0])
    xsdata.set_scatter_matrix([[[0.0]]])
    mg_library = openmc.MGXSLibrary(groups)
    mg_library.add_xsdata(xsdata)
    mg_library.export_to_hdf5('mgxs.h5')

    mat = openmc.Material()
    mat.add_macroscopic('void')
    materials = openmc.Materials([mat])
    materials.cross_sections = 'mgxs.h5'

    x_min = openmc.XPlane(-1.0, boundary_type='vacuum')
    x_mid = openmc.XPlane(0.0)
    x_max = openmc.XPlane(1.0, boundary_type='vacuum')
    yz = openmc.model.RectangularPrism(2.0, 2.0, axis='x',
                                       boundary_type='vacuum')
    left = openmc.Cell(fill=mat, region=+x_min & -x_mid & -yz)
    right = openmc.Cell(fill=mat, region=+x_mid & -x_max & -yz)
    geometry = openmc.Geometry([left, right])

    # The mesh plane lies inside the birth cell, so the surface x = 0 is
    # entirely in the mesh element with the lower window
    mesh = openmc.RectilinearMesh()
    mesh.x_grid = [-1.0, -0.1, 1.0]
    mesh.y_grid = [-1.0, 1.0]
    mesh.z_grid = [-1.0, 1.0]
    ww = openmc.WeightWindows(mesh, lower_ww_bounds=[0.5, 0.1],
                              upper_ww_bounds=[1.5, 0.2])

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.particles = 10
    settings.batches = 1
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((-0.5, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((1.0, 0.0, 0.0)),
    )
    settings.weight_windows = ww
    settings.weight_window_checkpoints = {'collision': False, 'surface': True}
    settings.shared_secondary_bank = True

    return openmc.Model(geometry, materials, settings)


def _check_shared_secondary_tracks(tracks, source_ids):
    """Check that the tracks written are those of the given source particles
    and the four split copies of each"""
    sources = {t.identifier[2]: t for t in tracks if t.root_id is None}
    secondaries = [t for t in tracks if t.root_id is not None]
    assert set(sources) == set(source_ids)
    assert len(secondaries) == 4*len(source_ids)

    for track in tracks:
        # Each dataset holds one particle, with no empty leading track
        assert len(track) == 1
        assert len(track[0].states) > 0

    for track in secondaries:
        # A split copy starts where its source particle crossed x = 0, with
        # the weight it was given by the split
        assert track.root_id in sources
        state = track[0].states[0]
        assert state['r']['x'] == pytest.approx(0.0)
        assert state['wgt'] == pytest.approx(0.2)


@pytest.mark.parametrize("event_based", [False, True])
def test_shared_secondary_max_tracks(run_in_tmpdir, event_based):
    # Only the source particles count toward max_tracks; the copies are written
    # with the source particles whose tracks are written
    model = _split_model()
    model.settings.event_based = event_based
    model.settings.max_tracks = 3
    generate_track_file(model, tracks=True)

    tracks = openmc.Tracks('tracks.h5')
    source_ids = [t.identifier[2] for t in tracks if t.root_id is None]
    if not config['mpi']:
        assert len(source_ids) == 3
    _check_shared_secondary_tracks(tracks, source_ids)


@pytest.mark.parametrize("event_based", [False, True])
def test_shared_secondary_track_identifiers(run_in_tmpdir, event_based):
    # The copies of a source particle requested by its identifier are written
    model = _split_model()
    model.settings.event_based = event_based
    model.settings.track = [(1, 1, 2), (1, 1, 7)]
    generate_track_file(model)

    tracks = openmc.Tracks('tracks.h5')
    _check_shared_secondary_tracks(tracks, [2, 7])
