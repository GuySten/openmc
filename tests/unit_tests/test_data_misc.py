#!/usr/bin/env python

from math import log
import os
from pathlib import Path
import shutil
import warnings

import h5py
import numpy as np
import pytest
import openmc.data
from openmc.deplete import Chain, Nuclide


def test_data_library(tmpdir):
    lib = openmc.data.DataLibrary.from_xml()
    for f in lib:
        assert sorted(f.keys()) == ['materials', 'path', 'type']

    f = lib.get_by_material('U235')
    assert f['type'] == 'neutron'
    assert 'U235' in f['materials']

    f = lib.get_by_material('c_H_in_H2O', data_type='thermal')
    assert f['type'] == 'thermal'
    assert 'c_H_in_H2O' in f['materials']

    lib.remove_by_material('Pu239')
    assert lib.get_by_material('Pu239') is None

    filename = str(tmpdir.join('test.xml'))
    lib.export_to_xml(filename)
    assert os.path.exists(filename)

    new_lib = openmc.data.DataLibrary()
    directory = os.path.dirname(openmc.config.get('cross_sections'))
    new_lib.register_file(os.path.join(directory, 'H1.h5'))
    assert new_lib[-1]['type'] == 'neutron'
    new_lib.register_file(os.path.join(directory, 'c_Zr_in_ZrH.h5'))
    assert new_lib[-1]['type'] == 'thermal'


def test_depletion_chain_data_library(run_in_tmpdir):
    dep_lib = openmc.data.DataLibrary.from_xml()
    prev_len = len(dep_lib.libraries)
    chain_path = Path(__file__).parents[1] / "chain_simple.xml"
    dep_lib.register_file(chain_path)
    assert len(dep_lib.libraries) == prev_len + 1
    # Inspect
    dep_dict = dep_lib.libraries[-1]
    assert dep_dict['materials'] == []
    assert dep_dict['type'] == 'depletion_chain'
    assert dep_dict['path'] == str(chain_path)

    out_path = "cross_section_chain.xml"
    dep_lib.export_to_xml(out_path)

    dep_import = openmc.data.DataLibrary.from_xml(out_path)
    for lib in reversed(dep_import.libraries):
        if lib['type'] == 'depletion_chain':
            break
    else:
        raise IndexError("depletion_chain not found in exported DataLibrary")
    assert os.path.exists(lib['path'])


def _fake_data_file(path, materials, data_type='neutron'):
    """Write an HDF5 file that register_file reads like a data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.attrs['filetype'] = np.bytes_(f'data_{data_type}')
        for name in materials:
            f.create_group(name)
    return path


def _paths(lib):
    return [(Path(e['path']).name, e['materials']) for e in lib]


def test_register_file_duplicate_warns(tmp_path):
    a = _fake_data_file(tmp_path / 'a' / 'U235.h5', ['U235'])
    b = _fake_data_file(tmp_path / 'b' / 'U235.h5', ['U235'])
    lib = openmc.data.DataLibrary()
    lib.register_file(a)
    with pytest.warns(UserWarning, match='takes precedence'):
        lib.register_file(b)
    # The first entry is the one used
    assert lib.get_by_material('U235')['path'] == str(a)
    assert list(lib.duplicates()) == [('neutron', 'U235')]


def test_register_file_replace(tmp_path):
    u = _fake_data_file(tmp_path / 'U.h5', ['U235', 'U238'])
    h = _fake_data_file(tmp_path / 'H1.h5', ['H1'])
    new = _fake_data_file(tmp_path / 'new' / 'U235.h5', ['U235'])
    lib = openmc.data.DataLibrary()
    lib.register_file(u)
    lib.register_file(h)
    lib.register_file(new, replace=True)
    # The new file takes the place of the old entry, which keeps U238 only
    assert _paths(lib) == [('U235.h5', ['U235']), ('U.h5', ['U238']),
                           ('H1.h5', ['H1'])]
    assert lib.get_by_material('U235')['path'] == str(new)
    assert not lib.duplicates()


def test_update(tmp_path):
    base = openmc.data.DataLibrary()
    base.register_file(_fake_data_file(tmp_path / 'base' / 'H1.h5', ['H1']))
    base.register_file(_fake_data_file(tmp_path / 'base' / 'U235.h5', ['U235']))
    other = openmc.data.DataLibrary()
    other.register_file(_fake_data_file(tmp_path / 'other' / 'U235.h5', ['U235']))
    other.register_file(_fake_data_file(tmp_path / 'other' / 'O16.h5', ['O16']))
    other.export_to_xml(tmp_path / 'other' / 'cross_sections.xml')

    # Default: other takes precedence
    lib = openmc.data.DataLibrary(base)
    lib.update(tmp_path / 'other' / 'cross_sections.xml')
    assert [Path(e['path']).parent.name for e in lib] == ['base', 'other', 'other']
    assert 'other' in lib.get_by_material('U235')['path']
    assert not lib.duplicates()

    # replace=False only fills in what is missing, without warnings
    lib = openmc.data.DataLibrary(base)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lib.update(other, replace=False)
    assert [Path(e['path']).parent.name for e in lib] == ['base', 'base', 'other']
    assert 'base' in lib.get_by_material('U235')['path']
    assert lib.get_by_material('O16') is not None


def test_remove_by_material(tmp_path):
    lib = openmc.data.DataLibrary()
    lib.register_file(_fake_data_file(tmp_path / 'U.h5', ['U235', 'U238']))
    lib.register_file(_fake_data_file(tmp_path / 'Pu.h5', ['Pu239']))
    lib.register_file(_fake_data_file(tmp_path / 'U_photon.h5', ['U'], 'photon'))
    with pytest.warns(UserWarning):
        lib.register_file(_fake_data_file(tmp_path / 'U235.h5', ['U235']))

    # Several names, every matching entry, only the given materials
    lib.remove_by_material(['U235', 'Pu239'])
    assert _paths(lib) == [('U.h5', ['U238']), ('U_photon.h5', ['U'])]

    # Other data types are kept unless data_type=None
    lib.remove_by_material('U', data_type='neutron')
    assert lib.get_by_material('U', data_type='photon') is not None
    lib.remove_by_material('U', data_type=None)
    assert _paths(lib) == [('U.h5', ['U238'])]


def test_get_by_material_exact_type(tmp_path):
    lib = openmc.data.DataLibrary()
    lib.register_file(_fake_data_file(tmp_path / 'U.h5', ['U'], 'photonuclear'))
    assert lib.get_by_material('U', data_type='photon') is None
    assert lib.get_by_material('U', data_type='photonuclear') is not None


def test_export_relocatable(tmp_path):
    data = tmp_path / 'lib' / 'data'
    lib = openmc.data.DataLibrary()
    lib.register_file(_fake_data_file(data / 'neutron' / 'U235.h5', ['U235']))
    lib.register_file(_fake_data_file(data / 'thermal' / 'c_H.h5', ['c_H'],
                                      'thermal'))
    chain = Path(__file__).parents[1] / 'chain_simple.xml'
    lib.register_file(chain)
    lib.export_to_xml(tmp_path / 'lib' / 'cross_sections.xml')

    # Files under the XML's directory get relative paths, others absolute
    # ones, and there is no <directory> element
    text = (tmp_path / 'lib' / 'cross_sections.xml').read_text()
    assert '<directory>' not in text
    assert 'path="data/neutron/U235.h5"' in text
    assert f'path="{os.path.abspath(chain)}"' in text

    # The directory can be moved as a whole
    shutil.move(tmp_path / 'lib', tmp_path / 'moved')
    moved = openmc.data.DataLibrary.from_xml(tmp_path / 'moved' / 'cross_sections.xml')
    assert len(moved) == 3
    for entry in moved:
        assert Path(entry['path']).exists()

    # absolute=True writes absolute paths
    moved.export_to_xml(tmp_path / 'abs.xml', absolute=True)
    for entry in openmc.data.DataLibrary.from_xml(tmp_path / 'abs.xml'):
        assert Path(entry['path']).is_absolute()


def test_from_xml_duplicate_warns(tmp_path):
    xml = tmp_path / 'cross_sections.xml'
    xml.write_text(
        '<?xml version="1.0"?>\n<cross_sections>\n'
        '  <library materials="U235" path="a.h5" type="neutron"/>\n'
        '  <library materials="U235 U238" path="b.h5" type="neutron"/>\n'
        '  <library materials="U235" path="c.h5" type="photon"/>\n'
        '</cross_sections>\n')
    with pytest.warns(UserWarning, match=r'U235 \(neutron\)'):
        lib = openmc.data.DataLibrary.from_xml(xml)
    assert lib.get_by_material('U235')['path'].endswith('a.h5')
    assert lib.duplicates() == {
        ('neutron', 'U235'): [str(tmp_path / 'a.h5'), str(tmp_path / 'b.h5')]}


def test_run_with_duplicate_and_unknown_type(run_in_tmpdir, capsys):
    # OpenMC uses the first entry of a duplicated material, warns about it,
    # and skips library types it does not know instead of aborting
    u235 = openmc.data.DataLibrary.from_xml().get_by_material('U235')
    lib = openmc.data.DataLibrary([
        {'path': u235['path'], 'type': 'neutron', 'materials': ['U235']},
        {'path': 'missing.h5', 'type': 'neutron', 'materials': ['U235']},
        {'path': 'future.h5', 'type': 'future_type', 'materials': ['U235']},
    ])
    lib.export_to_xml('custom_cross_sections.xml')

    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 1.0)
    sphere = openmc.Sphere(r=1.0, boundary_type='vacuum')
    model = openmc.Model()
    model.geometry = openmc.Geometry([openmc.Cell(region=-sphere, fill=fuel)])
    model.materials = openmc.Materials([fuel])
    model.materials.cross_sections = Path('custom_cross_sections.xml').resolve()
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 100
    model.settings.batches = 1
    model.settings.source = openmc.IndependentSource()
    model.run()

    output = capsys.readouterr().out
    assert "unrecognized type 'future_type'" in output
    assert 'more than once' in output


def test_linearize():
    """Test linearization of a continuous function."""
    x, y = openmc.data.linearize([-1., 1.], lambda x: 1 - x*x)
    f = openmc.data.Tabulated1D(x, y)
    assert f(-0.5) == pytest.approx(1 - 0.5*0.5, 0.001)
    assert f(0.32) == pytest.approx(1 - 0.32*0.32, 0.001)


def test_thin():
    """Test thinning of a tabulated function."""
    x = np.linspace(0., 2*np.pi, 1000)
    y = np.sin(x)
    x_thin, y_thin = openmc.data.thin(x, y)
    f = openmc.data.Tabulated1D(x_thin, y_thin)
    assert f(1.0) == pytest.approx(np.sin(1.0), 0.001)


def test_atomic_mass():
    assert openmc.data.atomic_mass('H1') == 1.007825031898
    assert openmc.data.atomic_mass('U235') == 235.043928117
    assert openmc.data.atomic_mass('Li6') == 6.01512288742
    assert openmc.data.atomic_mass('Pb220') == 220.025905
    with pytest.raises(KeyError):
        openmc.data.atomic_mass('U100')


def test_atomic_weight():
    assert openmc.data.atomic_weight('C') == 12.011115164865895
    assert openmc.data.atomic_weight('carbon') == 12.011115164865895
    with pytest.raises(ValueError):
        openmc.data.atomic_weight('Qt')


def test_water_density():
    dens = openmc.data.water_density
    # These test values are from IAPWS R7-97(2012).  They are actually specific
    # volumes so they need to be inverted.  They also need to be divided by 1000
    # to convert from [kg / m^3] to [g / cm^3].
    assert dens(300.0, 3.0) == pytest.approx(1e-3/0.100215168e-2, 1e-6)
    assert dens(300.0, 80.0) == pytest.approx(1e-3/0.971180894e-3, 1e-6)
    assert dens(500.0, 3.0) == pytest.approx(1e-3/0.120241800e-2, 1e-6)


def test_gnds_name():
    assert openmc.data.gnds_name(1, 1) == 'H1'
    assert openmc.data.gnds_name(40, 90) == ('Zr90')
    assert openmc.data.gnds_name(95, 242, 0) == ('Am242')
    assert openmc.data.gnds_name(95, 242, 1) == ('Am242_m1')
    assert openmc.data.gnds_name(95, 242, 10) == ('Am242_m10')


def test_isotopes():
    hydrogen_isotopes = [('H1', 0.99984426), ('H2', 0.00015574)]
    assert openmc.data.isotopes('H') == hydrogen_isotopes
    assert openmc.data.isotopes('hydrogen') == hydrogen_isotopes
    assert openmc.data.isotopes('Al') == [('Al27', 1.0)]
    assert openmc.data.isotopes('Aluminum') == [('Al27', 1.0)]
    assert openmc.data.isotopes('aluminium') == [('Al27', 1.0)]
    with pytest.raises(ValueError):
        openmc.data.isotopes('Чорнобиль')


def test_zam():
    assert openmc.data.zam('H1') == (1, 1, 0)
    assert openmc.data.zam('Zr90') == (40, 90, 0)
    assert openmc.data.zam('Am242') == (95, 242, 0)
    assert openmc.data.zam('Am242_m1') == (95, 242, 1)
    assert openmc.data.zam('Am242_m10') == (95, 242, 10)
    with pytest.raises(ValueError):
        openmc.data.zam('garbage')
    with pytest.raises(ValueError):
        openmc.data.zam('Am242-m1')


def test_half_life(tmp_path):
    assert openmc.data.half_life('H2') is None
    assert openmc.data.half_life('U235') == pytest.approx(2.22102e16)
    assert openmc.data.half_life('Am242') == pytest.approx(57672.0)
    assert openmc.data.half_life('Am242_m1') == pytest.approx(4449622000.0)
    assert openmc.data.decay_constant('H2') == 0.0
    assert openmc.data.decay_constant('U235') == pytest.approx(log(2.0)/2.22102e16)
    assert openmc.data.decay_constant('Am242') == pytest.approx(log(2.0)/57672.0)
    assert openmc.data.decay_constant('Am242_m1') == pytest.approx(log(2.0)/4449622000.0)

    # Create minimal chain with H3 and Am242 to test half-life and decay
    # constant retrieval from chain file
    chain = Chain()
    h3 = Nuclide("H3")
    h3.half_life = 1.0
    chain.add_nuclide(h3)
    am242 = Nuclide("Am242")
    chain.add_nuclide(am242)

    assert openmc.data.half_life('H3', chain_file=chain) == 1.0
    assert openmc.data.decay_constant('H3', chain_file=chain) == pytest.approx(log(2.0))

    # Nuclides that are present but stable in the chain should not fall back to
    # ENDF/B-VIII.0 data.
    assert openmc.data.half_life('Am242', chain_file=chain) is None
    assert openmc.data.decay_constant('Am242', chain_file=chain) == 0.0

    # Nuclides missing from the chain fall back to ENDF/B-VIII.0 data.
    assert openmc.data.half_life('U235', chain_file=chain) == pytest.approx(2.22102e16)

    chain_path = tmp_path / "chain.xml"
    chain.export_to_xml(chain_path)
    assert openmc.data.half_life('H3', chain_file=chain_path) == 1.0

    endf_h3 = openmc.data.half_life('H3')
    with openmc.config.patch('chain_file', chain_path):
        assert openmc.data.half_life('H3', chain_file=None) == 1.0
        assert openmc.data.half_life('H3', chain_file=False) == endf_h3
