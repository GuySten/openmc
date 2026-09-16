from collections.abc import Callable
import os

import h5py
import numpy as np
import pytest
import openmc.data
import openmc

from . import needs_njoy


def _gamma_file(endf_data, name):
    """Path to an ENDF gamma sublibrary evaluation, or skip.

    The ENDF tarball CI downloads carries neutrons, photoat and atomic_relax
    but no gammas, so these have to skip rather than fail.
    """
    path = os.path.join(endf_data, 'gammas', name)
    if not os.path.exists(path):
        pytest.skip(f'no ENDF gamma sublibrary evaluation at {path}')
    return path


@pytest.fixture(scope='module')
def pu239():
    """Pu239 HDF5 data."""
    # Photonuclear data is not part of the standard library, so a missing file
    # is a skip. Reading openmc.config directly would raise TypeError when no
    # cross sections are configured at all.
    cross_sections = openmc.config.get('cross_sections')
    if cross_sections is None:
        pytest.skip('no cross section configuration')
    filename = os.path.join(os.path.dirname(cross_sections),
                            'photonuclear', 'Pu239.h5')
    if not os.path.exists(filename):
        pytest.skip('no photonuclear data for Pu239')
    return openmc.data.IncidentPhotonuclear.from_hdf5(filename)


@pytest.fixture(scope='module')
def u235(endf_data):
    return openmc.data.IncidentPhotonuclear.from_njoy(
        _gamma_file(endf_data, 'g-092_U_235.endf'))


@pytest.fixture(scope='module')
def be9(endf_data):
    """Be9 ENDF data (contains laboratory angle-energy distribution)."""
    return openmc.data.IncidentPhotonuclear.from_endf(
        _gamma_file(endf_data, 'g-004_Be_009.endf'))


@pytest.fixture(scope='module')
def h2(endf_data):
    return openmc.data.IncidentPhotonuclear.from_njoy(
        _gamma_file(endf_data, 'g-001_H_002.endf'))


@pytest.fixture
def synthetic():
    """A small but structurally complete IncidentPhotonuclear.

    Built in memory so that the writer/reader contract, the sum rules and the
    redundant-reaction bookkeeping are all covered without NJOY and without a
    photonuclear library.
    """
    d = openmc.data.IncidentPhotonuclear('Pb208', 82, 208, 0, 206.19)
    energy = np.array([1.0e6, 7.5e6, 1.0e7, 2.0e7])
    d.energy = energy
    for mt, x, y in ((51, energy[1:], [0.0, 1.0, 2.0]),
                     (52, energy[2:], [0.0, 3.0]),
                     (301, energy, [0.0, 1.0e6, 2.0e6, 3.0e6])):
        rx = openmc.data.PhotonuclearReaction(mt)
        rx.xs = openmc.data.Tabulated1D(x, np.array(y, dtype=float))
        d.reactions[mt] = rx
    d.reactions[301].redundant = True

    product = openmc.data.Product('neutron')
    product.yield_ = openmc.data.Tabulated1D([7.5e6, 2.0e7], [1.0, 1.0])
    d.reactions[51].products.append(product)
    return d


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


def test_export_to_hdf5(tmpdir, pu239):
    filename = str(tmpdir.join('pu239.h5'))
    pu239.export_to_hdf5(filename)
    assert os.path.exists(filename)


def test_endf_export_refused_before_opening(tmpdir, be9):
    """ENDF-derived data cannot be written, and the refusal must not leave a
    stub file behind."""
    filename = str(tmpdir.join('be9.h5'))
    with pytest.raises(NotImplementedError):
        be9.export_to_hdf5(filename)
    assert not os.path.exists(filename)


def test_sum_rules(synthetic):
    """A redundant MT is assembled from its components on demand."""
    assert synthetic.get_reaction_components(4) == [51, 52]
    rx4 = synthetic[4]
    assert rx4.redundant
    energy = np.array([8.0e6, 1.5e7])
    np.testing.assert_allclose(
        rx4.xs(energy), synthetic[51].xs(energy) + synthetic[52].xs(energy))


def test_hdf5_layout(tmpdir, synthetic):
    """Every field the C++ reader depends on, checked against the writer."""
    filename = str(tmpdir.join('pb208.h5'))
    synthetic.export_to_hdf5(filename, 'w')
    with h5py.File(filename, 'r') as f:
        assert f.attrs['filetype'] == b'data_photonuclear'
        assert tuple(f.attrs['version']) == openmc.data.HDF5_VERSION
        g = f['Pb208']
        assert g.attrs['Z'] == 82
        assert g.attrs['A'] == 208
        grid = g['energy'][()]
        assert np.all(np.diff(grid) > 0.0)
        for rx in g['reactions'].values():
            threshold_idx = rx['xs'].attrs['threshold_idx']
            assert 0 <= threshold_idx < grid.size
            # PhotonuclearReaction::xs() indexes the cross section relative to
            # threshold_idx and never bounds-checks the top
            assert rx['xs'].shape[0] == grid.size - threshold_idx
            assert rx.attrs['center_of_mass'] in (0, 1)
            assert rx.attrs['redundant'] in (0, 1)
            # np.bytes_ of an int silently yields b''
            assert rx.attrs['label'] != b''


def test_hdf5_roundtrip(tmpdir, synthetic):
    filename = str(tmpdir.join('pb208.h5'))
    synthetic.export_to_hdf5(filename, 'w')
    back = openmc.data.IncidentPhotonuclear.from_hdf5(filename)

    assert back.name == 'Pb208'
    assert back.atomic_number == 82
    assert back.mass_number == 208
    energy = np.array([2.0e6, 8.0e6, 1.9e7])
    for mt in (51, 52, 301):
        np.testing.assert_allclose(back[mt].xs(energy),
                                   synthetic[mt].xs(energy))
        assert back[mt].redundant == synthetic[mt].redundant


def test_export_does_not_mutate(synthetic, tmpdir):
    """Writing a file must not rewrite the object being written."""
    before = [(rx.mt, rx.xs.x.copy(), rx.xs.y.copy()) for rx in synthetic]
    synthetic.export_to_hdf5(str(tmpdir.join('a.h5')), 'w')
    after = {rx.mt: rx for rx in synthetic}
    for mt, x, y in before:
        np.testing.assert_array_equal(after[mt].xs.x, x)
        np.testing.assert_array_equal(after[mt].xs.y, y)


@pytest.mark.parametrize('particle', ['neutron', 'photon'])
def test_level_inelastic_hdf5(tmpdir, particle):
    """The level distribution round trips, and a neutron one still carries the
    pre-3.1 attributes an older OpenMC needs."""
    dist = openmc.data.LevelInelastic(-7.368e6, 206.19, particle)
    filename = str(tmpdir.join('level.h5'))
    with h5py.File(filename, 'w') as f:
        dist.to_hdf5(f.create_group('d'))
    with h5py.File(filename, 'r') as f:
        attrs = f['d'].attrs
        assert attrs['particle'].decode() == particle
        if particle == 'neutron':
            A = dist.mass
            assert attrs['threshold'] == pytest.approx(dist.threshold)
            assert attrs['mass_ratio'] == pytest.approx((A/(A + 1.0))**2)
        else:
            assert 'threshold' not in attrs
        back = openmc.data.LevelInelastic.from_hdf5(f['d'])

    assert back.q_value == pytest.approx(dist.q_value)
    assert back.mass == pytest.approx(dist.mass)
    assert back.particle == particle
    assert back.threshold == pytest.approx(dist.threshold)


def test_level_inelastic_legacy_attrs(tmpdir):
    """A pre-3.1 file, which carries only the neutron form, still reads."""
    A = 206.19
    filename = str(tmpdir.join('old.h5'))
    with h5py.File(filename, 'w') as f:
        g = f.create_group('d')
        g.attrs['type'] = np.bytes_('level')
        g.attrs['threshold'] = (A + 1.0)/A*7.368e6
        g.attrs['mass_ratio'] = (A/(A + 1.0))**2
    with h5py.File(filename, 'r') as f:
        back = openmc.data.LevelInelastic.from_hdf5(f['d'])

    assert back.mass == pytest.approx(A)
    assert back.q_value == pytest.approx(-7.368e6)
    assert back.particle == 'neutron'


def test_kalbach_slope_rejects_other_projectiles():
    with pytest.raises(NotImplementedError):
        openmc.data.kalbach_slope(15.0e6, 3.0e6, 1000, 1, 82208)
    with pytest.raises(NotImplementedError):
        openmc.data.kalbach_slope(15.0e6, 3.0e6, 1003, 1, 82208)


@pytest.mark.parametrize('e_gamma,e_b_cm', [
    (15.0e6, 3.0e6), (40.0e6, 10.0e6), (100.0e6, 90.0e6)])
def test_kalbach_slope_photon(e_gamma, e_b_cm):
    """Eq. 6.5 of the ENDF-6 Formats Manual (BNL-224854-2023, section 6.2):

        a_gamma = a_n(E_gamma, E_b_cm) * sqrt(E_gamma/(2 m_n))
                  * min(4, max(1, 9.3/sqrt(E_b_cm)))

    with E_b_cm and m_n in MeV, and a_n evaluated by plugging E_gamma into
    the incident-neutron slot. Both of those are easy to get subtly wrong --
    the channel energy epsilon_b and the true photon compound system are the
    tempting substitutions, and both are incorrect here -- so the formula is
    pinned against an independent evaluation.
    """
    from openmc.data.data import NEUTRON_MASS_EV, EV_PER_MEV

    slope_n = openmc.data.kalbach_slope(e_gamma, e_b_cm, 1, 1, 82208)
    expected = (slope_n*np.sqrt(e_gamma/(2.0*NEUTRON_MASS_EV))
                * min(4.0, max(1.0, 9.3/np.sqrt(e_b_cm/EV_PER_MEV))))

    got = openmc.data.kalbach_slope(e_gamma, e_b_cm, 0, 1, 82208)
    assert got == pytest.approx(expected, rel=1e-12)
    # A photon carries less momentum than a nucleon of the same energy
    assert 0.0 < got < slope_n


def test_kalbach_slope_photon_clip_saturates():
    """The clipping factor saturates at 4 below 5.41 MeV and at 1 above
    86.5 MeV, so the ratio to the unclipped scaling is flat outside that
    window."""
    def ratio(e_b_cm):
        photon = openmc.data.kalbach_slope(20.0e6, e_b_cm, 0, 1, 82208)
        neutron = openmc.data.kalbach_slope(20.0e6, e_b_cm, 1, 1, 82208)
        return photon/neutron

    assert ratio(1.0e6) == pytest.approx(ratio(3.0e6), rel=1e-12)   # both at 4
    assert ratio(9.0e7) == pytest.approx(ratio(1.0e8), rel=1e-12)   # both at 1


def test_kalbach_slope_photon_zero_outgoing_energy():
    """A zero outgoing energy is a normal first grid point of an ENDF
    LAW=1/LANG=2 table, and must not raise a divide-by-zero warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert openmc.data.kalbach_slope(15.0e6, 0.0, 0, 1, 82208) >= 0.0


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


