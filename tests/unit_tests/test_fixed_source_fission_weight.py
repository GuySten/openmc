"""Fission sites in fixed-source mode carry the weight of the neutron that
made them (in eigenvalue mode they carry unit weight)."""

import numpy as np
import openmc
import pytest


@pytest.fixture
def model():
    # Bare HEU sphere, well subcritical, so that every chain ends
    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 18.7)
    sphere = openmc.Sphere(r=4.0, boundary_type='vacuum')
    model = openmc.Model()
    model.geometry = openmc.Geometry([openmc.Cell(region=-sphere, fill=fuel)])
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 2000
    model.settings.batches = 10
    # Two sources sampled equally: weights 1.98 and 0.02
    energy = openmc.stats.delta_function(1.0e6)
    model.settings.source = [
        openmc.IndependentSource(energy=energy, strength=99.0),
        openmc.IndependentSource(energy=energy, strength=1.0)]
    model.settings.uniform_source_sampling = True
    model.settings.surf_source_write = {'max_particles': 100000}

    tallies = []
    for estimator in ('tracklength', 'analog'):
        t = openmc.Tally(name=estimator)
        t.scores = ['nu-fission']
        t.estimator = estimator
        tallies.append(t)
    model.tallies = tallies
    return model


def test_sites_keep_parent_weight(run_in_tmpdir, model):
    # Without survival biasing only fission could change a weight: every
    # escaping neutron, fission progeny included, has a source weight
    sp_file = model.run()
    weights = {round(p.wgt, 12) for p in
               openmc.ParticleList.from_hdf5('surface_source.h5')}
    assert weights == {1.98, 0.02}

    # The analog estimator scores the banked weight: it must agree with the
    # track-length estimator
    with openmc.StatePoint(sp_file) as sp:
        tl = sp.get_tally(name='tracklength')
        an = sp.get_tally(name='analog')
        diff = abs(tl.mean.sum() - an.mean.sum())
        sd = np.hypot(tl.std_dev.sum(), an.std_dev.sum())
    assert diff < 4*sd


def test_same_mean_as_analog(run_in_tmpdir, model):
    # Unit source weights (no uniform source sampling) give the same
    # nu-fission rate per source neutron as the weighted run
    results = []
    for uniform in (False, True):
        model.settings.uniform_source_sampling = uniform
        sp_file = model.run()
        with openmc.StatePoint(sp_file) as sp:
            t = sp.get_tally(name='tracklength')
            results.append((t.mean.sum(), t.std_dev.sum()))
    (m0, s0), (m1, s1) = results
    assert abs(m0 - m1) < 4*np.hypot(s0, s1)
