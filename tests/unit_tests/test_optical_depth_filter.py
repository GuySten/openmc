"""Tests of the optical depth filter.

These exercise the Python API only and need no nuclear data.
"""

import numpy as np
import pytest

import openmc


def test_defaults_to_the_transport_total():
    """The attenuation a photon actually experiences is the default.

    Excluding coherent scattering is a convention that buildup-factor
    compilations use, not the physics of a beam, so it has to be asked for.
    """
    assert openmc.OpticalDepthFilter([0.0, 1.0, 2.0]).attenuation == 'total'


@pytest.mark.parametrize('attenuation', ['total', 'no-coherent'])
def test_xml_round_trip(attenuation):
    original = openmc.OpticalDepthFilter(np.linspace(0.0, 40.0, 81),
                                         attenuation=attenuation)
    restored = openmc.Filter.from_xml_element(original.to_xml_element())

    assert isinstance(restored, openmc.OpticalDepthFilter)
    assert restored.attenuation == attenuation
    assert np.array_equal(restored.values, original.values)
    assert restored.num_bins == original.num_bins


def test_hdf5_round_trip(run_in_tmpdir):
    """The statepoint representation, built the way to_statepoint() writes it."""
    h5py = pytest.importorskip('h5py')

    edges = np.array([0.0, 2.5, 7.5, 20.0])
    with h5py.File('filter.h5', 'w') as f:
        group = f.create_group('filter 1')
        group.create_dataset('type', data=np.bytes_('opticaldepth'))
        group.create_dataset('n_bins', data=len(edges) - 1)
        group.create_dataset('bins', data=edges)
        group.create_dataset('attenuation', data=np.bytes_('no-coherent'))
    with h5py.File('filter.h5', 'r') as f:
        restored = openmc.OpticalDepthFilter.from_hdf5(f['filter 1'])

    assert restored.attenuation == 'no-coherent'
    assert np.array_equal(restored.values, edges)


def test_the_basis_is_part_of_the_filter_identity():
    """Two depth grids that agree but are measured differently are not the same.

    A tally folded against a buildup factor has to know which convention its
    depths are on, so the two must not compare or hash alike.
    """
    edges = np.linspace(0.0, 40.0, 81)
    total = openmc.OpticalDepthFilter(edges)
    no_coh = openmc.OpticalDepthFilter(edges, attenuation='no-coherent')

    assert total != no_coh
    assert hash(total) != hash(no_coh)


@pytest.mark.parametrize('bad', [
    dict(values=[-1.0, 1.0]),                          # negative optical depth
    dict(values=[0.0, 1.0], attenuation='nonsense'),   # unknown convention
])
def test_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        openmc.OpticalDepthFilter(**bad)


def test_registered_in_lib():
    import openmc.lib
    assert (openmc.lib.filter._FILTER_TYPE_MAP['opticaldepth']
            is openmc.lib.OpticalDepthFilter)
