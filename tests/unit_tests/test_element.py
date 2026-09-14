from pathlib import Path

import lxml.etree as ET
import openmc
from pytest import approx, raises, warns

from openmc.data import NATURAL_ABUNDANCE, atomic_mass, isotopes


def test_expand_no_enrichment():
    """ Expand Li in natural compositions"""
    lithium = openmc.Element('Li')

    # Verify the expansion into ATOMIC fraction against natural composition
    for isotope in lithium.expand(100.0, 'ao'):
        assert isotope[1] == approx(NATURAL_ABUNDANCE[isotope[0]] * 100.0)

    # Verify the expansion into WEIGHT fraction against natural composition
    natural = {'Li6': NATURAL_ABUNDANCE['Li6'] * atomic_mass('Li6'),
               'Li7': NATURAL_ABUNDANCE['Li7'] * atomic_mass('Li7')}
    li_am = sum(natural.values())
    for key in natural:
        natural[key] /= li_am

    for isotope in lithium.expand(100.0, 'wo'):
        assert isotope[1] == approx(natural[isotope[0]] * 100.0)


def test_expand_enrichment():
    """ Expand and verify enrichment of Li """
    lithium = openmc.Element('Li')

    # Verify the enrichment by atoms
    ref = {'Li6': 75.0, 'Li7': 25.0}
    for isotope in lithium.expand(100.0, 'ao', 25.0, 'Li7', 'ao'):
        assert isotope[1] == approx(ref[isotope[0]])

    # Verify the enrichment by weight
    for isotope in lithium.expand(100.0, 'wo', 25.0, 'Li7', 'wo'):
        assert isotope[1] == approx(ref[isotope[0]])


def test_expand_no_isotopes():
    """Test that correct warning is raised for elements with no isotopes"""
    with warns(UserWarning, match='No naturally occurring'):
        element = openmc.Element('Tc')
        element.expand(100.0, 'ao')


def test_expand_ta():
    ref = {'Ta181': 100.0}
    element = openmc.Element('Ta')
    for isotope in element.expand(100.0, 'ao'):
        assert isotope[1] == approx(ref[isotope[0]])


def test_expand_exceptions():
    """ Test that correct exceptions are raised for invalid input """

    # 1 Isotope Element
    with raises(ValueError):
        element = openmc.Element('Be')
        element.expand(70.0, 'ao', 4.0, 'Be9')

    # 3 Isotope Element
    with raises(ValueError):
        element = openmc.Element('Cr')
        element.expand(70.0, 'ao', 4.0, 'Cr52')

    # Non-present Enrichment Target
    with raises(ValueError):
        element = openmc.Element('H')
        element.expand(70.0, 'ao', 4.0, 'H4')

    # Enrichment Procedure for Uranium if not Uranium
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0)

    # Missing Enrichment Target
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0, enrichment_type='ao')

    # Invalid Enrichment Type Entry
    with raises(ValueError):
        element = openmc.Element('Li')
        element.expand(70.0, 'ao', 4.0, 'Li7', 'Grand Moff Tarkin')

    # Trying to enrich Uranium
    with raises(ValueError):
        element = openmc.Element('U')
        element.expand(70.0, 'ao', 4.0, 'U235', 'wo')

    # Trying to enrich Uranium with wrong enrichment_target
    with raises(ValueError):
        element = openmc.Element('U')
        element.expand(70.0, 'ao', 4.0, enrichment_type='ao')


def _write_cross_sections(path, libraries):
    """Write a minimal cross_sections.xml listing the given (type, materials)."""
    root = ET.Element('cross_sections')
    for data_type, materials in libraries:
        lib = ET.SubElement(root, 'library')
        lib.set('materials', materials)
        lib.set('path', f'{materials}.h5')
        lib.set('type', data_type)
    ET.ElementTree(root).write(str(path))


def test_expand_photon_only_library(run_in_tmpdir):
    """Expand an element against a library that has no neutron data.

    Photon data is tabulated per element rather than per nuclide, so such a
    library says nothing about which isotopes are available and the element
    should be expanded by natural abundance instead of raising.
    """
    xs_path = Path('cross_sections.xml')
    _write_cross_sections(xs_path, [('photon', 'Fe')])

    iron = openmc.Element('Fe')
    expanded = dict(
        (name, percent)
        for name, percent, _ in iron.expand(100.0, 'ao', cross_sections=xs_path)
    )

    natural = {name: abundance for name, abundance in isotopes('Fe')}
    assert expanded.keys() == natural.keys()
    for name, abundance in natural.items():
        assert expanded[name] == approx(abundance * 100.0)


def test_expand_ignores_non_neutron_libraries(run_in_tmpdir):
    """Only neutron entries determine which isotopes a library provides."""
    xs_path = Path('cross_sections.xml')
    _write_cross_sections(
        xs_path, [('neutron', 'Li6'), ('photon', 'Li'), ('wmp', 'Li7')]
    )

    # Li7 is present only as multipole data, which cannot be used on its own,
    # so the expansion has no way to partition its abundance
    with raises(ValueError, match='Unsure how to partition'):
        openmc.Element('Li').expand(100.0, 'ao', cross_sections=xs_path)
