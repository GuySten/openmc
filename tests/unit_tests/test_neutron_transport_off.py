"""Tests for running without any neutron data.

Photon transport is driven by per-element data, so a calculation in which no
neutrons appear does not need neutron cross sections. These tests run against a
cross section library stripped of everything but its photon data, so a run that
touched neutron data in any way could not succeed.
"""

from pathlib import Path

import h5py
import numpy as np
import openmc
import openmc.data
import pytest


@pytest.fixture
def photon_only_xs(run_in_tmpdir):
    """Configure a library holding only the photon data of the real one."""
    library = openmc.data.DataLibrary.from_xml()

    photon_only = openmc.data.DataLibrary()
    for entry in library:
        if entry['type'] == 'photon':
            photon_only.append(entry)
    if not photon_only:
        pytest.skip('Configured cross section library has no photon data.')

    path = Path('photon_only_cross_sections.xml').resolve()
    photon_only.export_to_xml(path)

    # add_element consults the library as soon as it is called, so the model
    # has to be built with this configuration already in place
    with openmc.config.patch('cross_sections', path):
        yield path


def aluminum_photon_model():
    """A photon source in a sphere of natural aluminum."""
    openmc.reset_auto_ids()

    mat = openmc.Material()
    mat.add_element('Al', 1.0)
    mat.set_density('g/cm3', 2.7)

    sphere = openmc.Sphere(r=5.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sphere)

    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.run_mode = 'fixed source'
    model.settings.photon_transport = True
    model.settings.particles = 50
    model.settings.batches = 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        particle='photon',
    )
    return model, mat


def test_photon_run_without_neutron_data(photon_only_xs):
    """A photon calculation runs against a library with no neutron data."""
    model, mat = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.run()

    # The element was expanded without consulting neutron data
    assert [nuclide.name for nuclide in mat.nuclides] == ['Al27']

    with h5py.File('summary.h5', 'r') as f:
        atom_density = f['materials'][f'material {mat.id}']['atom_density'][()]
        awrs = f['nuclides']['awrs'][()]

    # Densities were normalized with the tabulated atomic mass standing in for
    # the atomic weight ratio that a neutron data file would have supplied
    mass = openmc.data.atomic_mass('Al27')
    assert awrs[0] == pytest.approx(mass / openmc.data.NEUTRON_MASS)
    assert atom_density == pytest.approx(
        1.0e-24 * 2.7 * openmc.data.AVOGADRO / mass)


def test_neutron_data_still_required_by_default(photon_only_xs):
    """Without opting in, the same library is rejected as before.

    This also confirms the library used above really does lack neutron data,
    so that the test above is not passing for the wrong reason.
    """
    model, _ = aluminum_photon_model()
    with pytest.raises(RuntimeError, match='Al27'):
        model.run()


def test_neutron_source_turns_neutron_transport_back_on(photon_only_xs, capsys):
    """A neutron source overrides the setting, as a photon source does."""
    model, _ = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(), particle='neutron')

    # Neutron data is needed again, rather than neutrons being transported
    # without any data to transport them with
    with pytest.raises(RuntimeError, match='Al27'):
        model.run()

    # Overriding what the user asked for is not done silently
    assert 'turned back on' in capsys.readouterr().out


def test_no_source_reports_the_missing_source(photon_only_xs):
    """The default source emits neutrons, so its absence is the real problem.

    Reporting missing neutron data here would point at the wrong thing.
    """
    model, _ = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.settings.source = []

    with pytest.raises(RuntimeError, match='no source was specified'):
        model.run()


def test_thermal_scattering_rejected(photon_only_xs):
    """S(a,b) data only affects neutrons and is not silently ignored."""
    model, mat = aluminum_photon_model()
    model.settings.neutron_transport = False
    mat.add_s_alpha_beta('c_Al27')

    with pytest.raises(RuntimeError, match='Thermal scattering data'):
        model.run()


def test_eigenvalue_rejected(photon_only_xs):
    """An eigenvalue calculation cannot run without neutrons."""
    model, _ = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.settings.run_mode = 'eigenvalue'
    model.settings.inactive = 1

    with pytest.raises(RuntimeError, match='eigenvalue'):
        model.run()

def test_volume_calculation_without_neutron_data(photon_only_xs):
    """A volume calculation reports nuclide inventories, not reaction rates.

    It samples no source and transports nothing, so it has no use for neutron
    data even though it does need to know what the materials are made of.
    """
    model, mat = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.settings.run_mode = 'volume'

    cell = model.geometry.get_all_cells()[mat.id]
    model.settings.volume_calculations = [openmc.VolumeCalculation(
        [cell], 10_000, (-5., -5., -5.), (5., 5., 5.))]

    model.calculate_volumes(apply_volumes=False)

    volumes = openmc.VolumeCalculation.from_hdf5('volume_1.h5')
    result = volumes.atoms[cell.id]
    assert list(result.keys()) == ['Al27']

    # Atom counts follow from the density normalization, which used the
    # tabulated atomic mass in place of the missing atomic weight ratio
    volume = volumes.volumes[cell.id].n
    expected = 1.0e24 * volume * 2.7 * openmc.data.AVOGADRO / (
        1.0e24 * openmc.data.atomic_mass('Al27'))
    assert result['Al27'].n == pytest.approx(expected, rel=1.0e-6)


def test_multigroup_rejected(photon_only_xs):
    """Multi-group data is neutron data, so it cannot be turned off.

    This also covers the random ray solver, which requires multi-group mode.
    """
    model, _ = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.settings.energy_mode = 'multi-group'

    with pytest.raises(RuntimeError, match='multigroup mode'):
        model.run()


def test_neutron_particle_restart_rejected(photon_only_xs):
    """A restarted neutron would otherwise be transported with no data.

    The particle's type is only known once the restart file is read, long after
    the decision of which data to load, so it is checked at that point.
    """
    model, _ = aluminum_photon_model()
    model.settings.neutron_transport = False
    model.export_to_model_xml()

    # Written by hand because OpenMC only produces these files for particles
    # that get lost, which is not something a test can arrange reliably
    with h5py.File('neutron_restart.h5', 'w') as f:
        f.attrs['filetype'] = np.bytes_('particle restart')
        f.attrs['version'] = np.array([2, 1])
        f.create_dataset('current_batch', data=1)
        f.create_dataset('generations_per_batch', data=1)
        f.create_dataset('current_generation', data=1)
        f.create_dataset('n_particles', data=1)
        f.create_dataset('run_mode', data=np.bytes_('fixed source'))
        f.create_dataset('id', data=1)
        f.create_dataset('type', data=2112)  # PDG code for a neutron
        f.create_dataset('weight', data=1.0)
        f.create_dataset('energy', data=1.0e6)
        f.create_dataset('xyz', data=np.zeros(3))
        f.create_dataset('uvw', data=np.array([0., 0., 1.]))
        f.create_dataset('time', data=0.0)

    with pytest.raises(RuntimeError, match='restarted is a neutron'):
        openmc.run(restart_file='neutron_restart.h5')
