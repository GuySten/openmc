"""FissionGenerationFilter: the number of fissions in a particle's ancestry."""

import numpy as np
import pytest
import openmc
import openmc.lib


def _model(photons=False):
    """A subcritical bare sphere of U-235 and water with a 2 MeV neutron
    point source at its centre."""
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.add_nuclide('H1', 20.0)
    fuel.add_nuclide('O16', 10.0)
    fuel.set_density('g/cm3', 1.5)
    sphere = openmc.Sphere(r=8.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=fuel, region=-sphere)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    s = model.settings
    s.run_mode = 'fixed source'
    s.particles = 1000
    s.batches = 5
    s.source = openmc.IndependentSource(
        energy=openmc.stats.delta_function(2.0e6))
    s.photon_transport = photons
    return model, cell


def _flux_tallies(model, particle, generations):
    total = openmc.Tally(name='total')
    total.filters = [openmc.ParticleFilter([particle])]
    total.scores = ['flux']
    by_gen = openmc.Tally(name='by generation')
    by_gen.filters = [openmc.ParticleFilter([particle]),
                      openmc.FissionGenerationFilter(generations)]
    by_gen.scores = ['flux']
    model.tallies = [total, by_gen]


def _values(sp):
    total = sp.get_tally(name='total').mean.ravel()[0]
    by_gen = sp.get_tally(name='by generation').mean.ravel()
    return total, by_gen


def test_filter_xml_roundtrip():
    f = openmc.FissionGenerationFilter([0, 1, 2], filter_id=7)
    elem = f.to_xml_element()
    assert elem.get('type') == 'fissiongeneration'
    g = openmc.Filter.from_xml_element(elem)
    assert isinstance(g, openmc.FissionGenerationFilter)
    assert np.array_equal(g.bins, [0, 1, 2])
    with pytest.raises(ValueError):
        openmc.FissionGenerationFilter([-1])


def test_generations_partition_the_flux(run_in_tmpdir):
    """Every particle is in exactly one generation: the bins sum to the
    total, the source's own flux is in bin 0 and the chains below it."""
    model, _ = _model()
    _flux_tallies(model, 'neutron', list(range(100)))
    with openmc.StatePoint(model.run()) as sp:
        total, by_gen = _values(sp)
        f = sp.get_tally(name='by generation').find_filter(
            openmc.FissionGenerationFilter)
        assert isinstance(f, openmc.FissionGenerationFilter)
    assert by_gen.sum() == pytest.approx(total, rel=1e-12)
    assert by_gen[0] > 0.0 and by_gen[1] > 0.0 and by_gen[2] > 0.0


def test_no_fission_neutrons_only_generation_zero(run_in_tmpdir):
    """Without fission neutrons every neutron is the source's own."""
    model, _ = _model()
    model.settings.create_fission_neutrons = False
    _flux_tallies(model, 'neutron', [0, 1])
    with openmc.StatePoint(model.run()) as sp:
        total, by_gen = _values(sp)
    assert by_gen[0] == pytest.approx(total, rel=1e-12)
    assert by_gen[1] == 0.0


def test_fission_photons_start_at_generation_one(run_in_tmpdir):
    """A photon born in a fission is one generation below the neutron that
    fissioned, so with fission photons only none is in generation 0."""
    model, _ = _model(photons=True)
    model.settings.fission_photons_only = True
    model.settings.create_fission_neutrons = False
    _flux_tallies(model, 'photon', [0, 1, 2])
    with openmc.StatePoint(model.run()) as sp:
        total, by_gen = _values(sp)
    assert total > 0.0
    assert by_gen[0] == 0.0
    assert by_gen[1] == pytest.approx(total, rel=1e-12)


def test_split_copies_keep_their_generation(run_in_tmpdir):
    """Weight-window copies continue their particle's history: the bins
    still partition the total."""
    model, _ = _model()
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-8.0, -8.0, -8.0)
    mesh.upper_right = (8.0, 8.0, 8.0)
    mesh.dimension = (2, 2, 2)
    model.settings.weight_windows = openmc.WeightWindows(
        mesh, np.full(8, 0.05), upper_bound_ratio=5.0,
        energy_bounds=[0.0, 20.0e6])
    model.settings.weight_windows_on = True
    model.settings.survival_biasing = True
    _flux_tallies(model, 'neutron', list(range(100)))
    with openmc.StatePoint(model.run()) as sp:
        total, by_gen = _values(sp)
    assert by_gen.sum() == pytest.approx(total, rel=1e-12)


def test_lib_filter_type(run_in_tmpdir):
    model, _ = _model()
    _flux_tallies(model, 'neutron', [0, 1])
    model.export_to_model_xml()
    openmc.lib.init(output=False)
    try:
        types = {type(f).__name__ for f in openmc.lib.filters.values()}
        assert 'FissionGenerationFilter' in types
    finally:
        openmc.lib.finalize()
