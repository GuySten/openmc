from collections.abc import Callable
import os

import h5py
import numpy as np
import pytest
import openmc.data
import openmc

from . import needs_njoy


@pytest.fixture(scope='module')
def pu239():
    """Pu239 HDF5 data."""
    directory = os.path.dirname(openmc.config.get('cross_sections'))
    filename = os.path.join(directory, 'photonuclear', 'Pu239.h5')
    return openmc.data.IncidentPhotonuclear.from_hdf5(filename)


@pytest.fixture(scope='module')
def u235(endf_data):
    endf_file = os.path.join(endf_data, 'gammas', 'g-092_U_235.endf')
    return openmc.data.IncidentPhotonuclear.from_njoy(endf_file)


@pytest.fixture(scope='module')
def be9(endf_data):
    """Be9 ENDF data (contains laboratory angle-energy distribution)."""
    filename = os.path.join(endf_data, 'gammas', 'g-004_Be_009.endf')
    return openmc.data.IncidentPhotonuclear.from_endf(filename)


@pytest.fixture(scope='module')
def h2(endf_data):
    endf_file = os.path.join(endf_data, 'gammas', 'g-001_H_002.endf')
    return openmc.data.IncidentPhotonuclear.from_njoy(endf_file)


def test_attributes(pu239):
    assert pu239.name == 'Pu239'
    assert pu239.mass_number == 239
    assert pu239.metastable == 0
    assert pu239.atomic_symbol == 'Pu'
    assert pu239.atomic_weight_ratio == pytest.approx(236.9986)


@needs_njoy        
def test_fission_energy(u235):
    fer = u235.fission_energy
    assert isinstance(fer, openmc.data.FissionEnergyRelease)
    components = ['betas', 'delayed_neutrons', 'delayed_photons', 'fragments',
                  'neutrinos', 'prompt_neutrons', 'prompt_photons', 'recoverable',
                  'total', 'q_prompt', 'q_recoverable', 'q_total']
    for c in components:
        assert isinstance(getattr(fer, c), Callable)         


def test_energy_grid(pu239):
    grid = pu239.energy
    assert np.all(np.diff(grid) >= 0.0)


def test_reactions(pu239):
    assert 18 in pu239.reactions
    assert isinstance(pu239.reactions[18], openmc.data.PhotonuclearReaction)
    with pytest.raises(KeyError):
        pu239.reactions[2]


def test_fission(pu239):
    fission = pu239.reactions[18]
    assert not fission.center_of_mass
    assert fission.q_value == pytest.approx(197380000.0)
    assert fission.mt == 18
    assert len(fission.products) == 1
    prompt = fission.products[0]
    assert prompt.particle == 'neutron'
    assert prompt.yield_(1.0e-5) == pytest.approx(1.74559)


@needs_njoy
def test_kerma(run_in_tmpdir, h2):
    assert 301 in h2
    h2.export_to_hdf5("H2.h5")
    read_in = openmc.data.IncidentPhotonuclear.from_hdf5("H2.h5")
    assert 301 in read_in
    assert np.all(read_in[301].xs.y == h2[301].xs.y)


@needs_njoy
def test_get_reaction_components(h2):
    assert h2.get_reaction_components(1) == [50]
    assert h2.get_reaction_components(51) == []


def test_export_to_hdf5(tmpdir, pu239, be9):
    filename = str(tmpdir.join('pu239.h5'))
    pu239.export_to_hdf5(filename)
    assert os.path.exists(filename)
    with pytest.raises(NotImplementedError):
        be9.export_to_hdf5('be9.h5')


@needs_njoy
def test_ace_convert(endf_data, run_in_tmpdir):
    filename = os.path.join(endf_data, 'gammas', 'g-001_H_002.endf')
    ace_ascii = 'ace_ascii'
    ace_binary = 'ace_binary'
    openmc.data.njoy.make_ace_photonuclear(filename, acer=ace_ascii)

    # Convert to binary
    openmc.data.ace.ascii_to_binary(ace_ascii, ace_binary)

    # Make sure conversion worked
    lib_ascii = openmc.data.ace.Library(ace_ascii)
    lib_binary = openmc.data.ace.Library(ace_binary)
    for tab_a, tab_b in zip(lib_ascii.tables, lib_binary.tables):
        assert tab_a.name == tab_b.name
        assert tab_a.atomic_weight_ratio == pytest.approx(tab_b.atomic_weight_ratio)
        assert tab_a.temperature == pytest.approx(tab_b.temperature)
        assert np.all(tab_a.nxs == tab_b.nxs)
        assert np.all(tab_a.jxs == tab_b.jxs)
        assert tab_a.zaid == tab_b.zaid
        assert tab_a.data_type == tab_b.data_type


def test_ace_table_types():
    TT = openmc.data.ace.TableType
    assert TT.from_suffix('u') == TT.PHOTONUCLEAR
    assert TT.from_suffix('80u') == TT.PHOTONUCLEAR




def test_nbody_incident_particle(run_in_tmpdir):
    """The incident particle of an N-body distribution survives an HDF5 round
    trip, and data written without it are read as neutron data."""
    photon = openmc.data.NBodyPhaseSpace(2.9877, 3, 2.9896, -6.2575e6, 'photon')
    neutron = openmc.data.NBodyPhaseSpace(2.9877, 3, 2.9896, -6.2575e6)
    assert neutron.particle == 'neutron'
    with pytest.raises(ValueError):
        openmc.data.NBodyPhaseSpace(2.9877, 3, 2.9896, -6.2575e6, 'proton')

    with h5py.File('nbody.h5', 'w') as f:
        photon.to_hdf5(f.create_group('photon'))
        legacy = f.create_group('legacy')
        neutron.to_hdf5(legacy)
        del legacy.attrs['particle']

    with h5py.File('nbody.h5', 'r') as f:
        assert openmc.data.NBodyPhaseSpace.from_hdf5(f['photon']).particle == 'photon'
        assert openmc.data.NBodyPhaseSpace.from_hdf5(f['legacy']).particle == 'neutron'


def test_redundant_reaction():
    """A redundant reaction that is not stored is built from its components."""
    nuc = openmc.data.IncidentPhotonuclear('H2', 1, 2, 0, 1.9968)
    nuc.energy = np.array([2.3e6, 5.0e6, 1.0e7])
    rx = openmc.data.PhotonuclearReaction(50)
    rx.xs = openmc.data.Tabulated1D(nuc.energy, [0.0, 2.0e-3, 1.0e-3])
    nuc.reactions[50] = rx

    total = nuc[1]
    assert total.redundant
    assert total.xs(5.0e6) == pytest.approx(2.0e-3)


def test_level_inelastic_from_endf(endf_data, run_in_tmpdir):
    """A level reaction given only an angular distribution (MF=4) gets a
    photon-incident LevelInelastic energy distribution."""
    # No ENDF/B photonuclear evaluation has such a reaction, so make one from
    # H2 by replacing the MF=6 distribution of MT=50 with an isotropic MF=4 one
    filename = os.path.join(endf_data, 'gammas', 'g-001_H_002.endf')
    with open(filename) as f:
        lines = f.read().splitlines()
    i_mf6 = next(i for i, line in enumerate(lines) if line[70:75] == ' 6 50')
    head = lines[i_mf6]
    mat = head[66:70]
    za, awr = head[:11], head[11:22]
    zero = ' 0.000000+0'
    mf4 = [
        f'{za}{awr}{0:11d}{0:11d}{0:11d}{0:11d}{mat} 4 50',
        f'{zero}{awr}{1:11d}{1:11d}{0:11d}{0:11d}{mat} 4 50',
        f'{zero}{zero}{0:11d}{0:11d}{0:11d}{0:11d}{mat} 4  0',
    ]
    lines = [line for line in lines if line[70:75] != ' 6 50']
    lines[i_mf6:i_mf6] = mf4
    with open('h2_mf4.endf', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    h2 = openmc.data.IncidentPhotonuclear.from_endf('h2_mf4.endf')
    rx = h2[50]
    assert len(rx.products) == 1
    product = rx.products[0]
    assert product.particle == 'neutron'
    dist = product.distribution[0].energy
    assert isinstance(dist, openmc.data.LevelInelastic)
    assert dist.particle == 'photon'
    assert dist.q_value == rx.q_value
    assert dist.mass == pytest.approx(h2.atomic_weight_ratio)
    # The photon threshold lies just above |Q| because of the recoil
    assert -rx.q_value < dist.threshold < -rx.q_value * 1.001
