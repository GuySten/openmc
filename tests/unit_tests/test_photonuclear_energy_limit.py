"""Source particles above the maximum photon energy lowered for photonuclear
physics must stop the run rather than be silently rejected and resampled."""

import os

import h5py
import numpy as np
import pytest
import openmc

# Heavy water whose photonuclear data extend well above the photon energy at
# which a photoneutron can exceed 20 MeV neutron data (30 to 45 MeV for 2H and
# 16O, depending on the library). The neutron data range of the problem is set
# by the nuclide whose data stop lowest, here a trace of Zr90 (20 MeV in most
# libraries). Source energies well on either side of the resulting limit.
E_BELOW = 10.0e6
E_ABOVE = 60.0e6


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    """Cross section library with photonuclear data for H2, or skip."""
    xs_path = openmc.config.get('cross_sections')
    if xs_path is None:
        pytest.skip('No cross section library configured')
    lib = openmc.data.DataLibrary.from_xml(xs_path)
    if lib.get_by_material('H2', data_type='photonuclear') is None:
        path = os.path.join(os.path.dirname(xs_path), 'photonuclear', 'H2.h5')
        if not os.path.isfile(path):
            pytest.skip('No photonuclear data for H2')
        lib.register_file(path)
    pn = lib.get_by_material('H2', data_type='photonuclear')
    with h5py.File(pn['path']) as f:
        pn_max = f['H2/energy'][-1]
    n_max = np.inf
    for name in ('H2', 'O16', 'Zr90'):
        neutron = lib.get_by_material(name, data_type='neutron')
        if neutron is None:
            pytest.skip(f'No neutron data for {name}')
        with h5py.File(neutron['path']) as f:
            group = f[f'{name}/energy']
            n_max = min(n_max, max(group[t][-1] for t in group))
    if pn_max <= E_ABOVE or n_max > 25.0e6:
        pytest.skip('H2 data do not lower the maximum photon energy '
                    'between the test source energies')
    out = tmp_path_factory.mktemp('library') / 'cross_sections.xml'
    lib.export_to_xml(out)
    return str(out)


def d2_model(library, source):
    """Thin heavy-water slab in vacuum with photonuclear physics enabled."""
    model = openmc.Model()
    d2o = openmc.Material()
    d2o.add_nuclide('H2', 2.0)
    d2o.add_nuclide('O16', 1.0)
    d2o.add_nuclide('Zr90', 1.0e-6)
    d2o.set_density('g/cm3', 1.1)
    model.materials = openmc.Materials([d2o])
    model.materials.cross_sections = library

    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([openmc.Cell(fill=d2o, region=-sph)])

    model.settings.run_mode = 'fixed source'
    model.settings.particles = 100
    model.settings.batches = 2
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.source = source
    return model


def photon_source(energy):
    return openmc.IndependentSource(particle='photon', energy=energy)


def test_monoenergetic_below_limit(run_in_tmpdir, library):
    model = d2_model(library, photon_source(openmc.stats.delta_function(E_BELOW)))
    model.run()


def test_monoenergetic_above_limit(run_in_tmpdir, library):
    model = d2_model(library, photon_source(openmc.stats.delta_function(E_ABOVE)))
    with pytest.raises(RuntimeError, match='Source photon of'):
        model.run()


def test_continuous_above_limit(run_in_tmpdir, library):
    # Previously rejected and resampled, silently truncating the spectrum
    energy = openmc.stats.Uniform(E_BELOW, E_ABOVE)
    model = d2_model(library, photon_source(energy))
    with pytest.raises(RuntimeError, match='Source photon of'):
        model.run()


def test_file_source_above_limit(run_in_tmpdir, library):
    # Previously not checked at all
    particles = [openmc.SourceParticle(E=E, particle=openmc.ParticleType.PHOTON)
                 for E in (E_BELOW, E_ABOVE)]
    openmc.write_source_file(particles, 'source.h5')
    model = d2_model(library, openmc.FileSource('source.h5'))
    with pytest.raises(RuntimeError, match='Source photon of'):
        model.run()


def test_file_source_below_limit(run_in_tmpdir, library):
    particles = [openmc.SourceParticle(E=E_BELOW, particle=openmc.ParticleType.PHOTON)]
    openmc.write_source_file(particles, 'source.h5')
    model = d2_model(library, openmc.FileSource('source.h5'))
    model.run()


def test_energy_max_removes_limit(run_in_tmpdir, library):
    # A maximum neutron energy within the neutron data removes the limit, and
    # photoneutrons above it are killed instead
    energy = openmc.stats.Uniform(E_BELOW, E_ABOVE)
    model = d2_model(library, photon_source(energy))
    model.settings.energy_max = {'neutron': 20.0e6}
    model.run()
