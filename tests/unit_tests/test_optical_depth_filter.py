"""Tests of the optical depth filter.

Python API only; these need no nuclear data.
"""

import numpy as np
import pytest

import openmc


def test_has_exactly_one_bin():
    """It weights rather than selects, so there is nothing to bin."""
    assert openmc.OpticalDepthFilter().num_bins == 1


def test_defaults_to_the_transport_total():
    """The attenuation a photon actually experiences is the default.

    Excluding coherent scattering is a convention some buildup-factor
    compilations use, not the physics of a beam, so it has to be asked for.
    """
    assert openmc.OpticalDepthFilter().attenuation == 'total'


@pytest.mark.parametrize('attenuation', ['total', 'no-coherent'])
def test_xml_round_trip(attenuation):
    original = openmc.OpticalDepthFilter(attenuation=attenuation)
    restored = openmc.Filter.from_xml_element(original.to_xml_element())

    assert isinstance(restored, openmc.OpticalDepthFilter)
    assert restored.attenuation == attenuation
    assert restored.num_bins == 1


def test_hdf5_round_trip(run_in_tmpdir):
    h5py = pytest.importorskip('h5py')

    with h5py.File('filter.h5', 'w') as f:
        group = f.create_group('filter 1')
        group.create_dataset('type', data=np.bytes_('opticaldepth'))
        group.create_dataset('n_bins', data=1)
        group.create_dataset('attenuation', data=np.bytes_('no-coherent'))
    with h5py.File('filter.h5', 'r') as f:
        restored = openmc.OpticalDepthFilter.from_hdf5(f['filter 1'])

    assert restored.attenuation == 'no-coherent'


def test_the_convention_is_part_of_the_identity():
    """A depth means nothing without saying what it was measured with.

    A tally folded against a buildup factor has to know which convention its
    depth is on, so two filters that differ only in that must not compare or
    hash alike.
    """
    total = openmc.OpticalDepthFilter()
    no_coh = openmc.OpticalDepthFilter(attenuation='no-coherent')

    assert total != no_coh
    assert hash(total) != hash(no_coh)


def test_rejects_an_unknown_convention():
    with pytest.raises(ValueError):
        openmc.OpticalDepthFilter(attenuation='nonsense')


def test_registered_in_lib():
    import openmc.lib
    assert (openmc.lib.filter._FILTER_TYPE_MAP['opticaldepth']
            is openmc.lib.OpticalDepthFilter)
