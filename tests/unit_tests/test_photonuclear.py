"""Transport-level tests for photonuclear physics.

Everything here runs the actual transport path, which is otherwise untested:
`src/photonuclear.cpp`, the `photonuclear_collision()` branch of
`src/physics.cpp`, and the photon ceiling that `set_photonuclear_bounds()`
derives. They skip when no photonuclear library is configured, since that data
is not yet part of the standard OpenMC library.
"""

import os

import numpy as np
import pytest
import openmc


def _photonuclear_available(nuclide):
    """Path check rather than a run, so a missing library is a skip."""
    cross_sections = openmc.config.get('cross_sections')
    if cross_sections is None:
        return False
    path = os.path.join(os.path.dirname(cross_sections), 'photonuclear',
                        f'{nuclide}.h5')
    return os.path.exists(path)


needs_photonuclear = pytest.mark.skipif(
    not _photonuclear_available('W186'),
    reason='no photonuclear data for W186')


def _sphere_model(nuclide, density, energy, particles=20000, batches=5,
                  biasing=False, run_mode='fixed source'):
    """A bare sphere driven by a monoenergetic photon point source."""
    openmc.reset_auto_ids()
    mat = openmc.Material()
    mat.set_density('g/cm3', density)
    mat.add_nuclide(nuclide, 1.0)

    sphere = openmc.Sphere(r=3.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sphere)

    settings = openmc.Settings()
    settings.run_mode = run_mode
    settings.particles = particles
    settings.batches = batches
    settings.photon_transport = True
    settings.photonuclear_physics = True
    settings.photoneutron_biasing = biasing
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
        energy=openmc.stats.delta_function(energy),
        particle='photon')
    if run_mode == 'eigenvalue':
        settings.inactive = 1

    neutrons = openmc.Tally(name='neutron flux')
    neutrons.filters = [openmc.ParticleFilter(['neutron'])]
    neutrons.scores = ['flux']

    heating = openmc.Tally(name='heating')
    heating.filters = [openmc.CellFilter(cell)]
    heating.scores = ['heating']

    return openmc.Model(geometry=openmc.Geometry([cell]),
                        materials=openmc.Materials([mat]),
                        settings=settings,
                        tallies=openmc.Tallies([neutrons, heating]))


def _run(model, tmpdir, name):
    statepoint = model.run(cwd=str(tmpdir.join(name)), output=False)
    results = {}
    with openmc.StatePoint(statepoint) as sp:
        for tally_name in ('neutron flux', 'heating'):
            tally = sp.get_tally(name=tally_name)
            results[tally_name] = (float(tally.mean.ravel()[0]),
                                   float(tally.std_dev.ravel()[0]))
        results['photonuclear_physics'] = sp.photonuclear_physics
        results['photoneutron_biasing'] = sp.photoneutron_biasing
    return results


@needs_photonuclear
def test_photoneutrons_are_produced(tmpdir):
    """An 18 MeV photon in tungsten is well above the giant dipole resonance,
    so photoneutrons must appear and the statepoint must record the mode."""
    results = _run(_sphere_model('W186', 19.3, 18.0e6), tmpdir, 'analog')

    mean, std_dev = results['neutron flux']
    assert mean > 0.0
    assert std_dev > 0.0
    assert bool(results['photonuclear_physics'])
    assert not results['photoneutron_biasing']


@needs_photonuclear
def test_no_photoneutrons_below_threshold(tmpdir):
    """Below the photonuclear threshold the channel is closed entirely, so the
    energy-range guard in Material::calculate_photon_xs must hold."""
    results = _run(_sphere_model('W186', 19.3, 2.0e6), tmpdir, 'sub')
    assert results['neutron flux'][0] == 0.0


@needs_photonuclear
def test_biased_and_analog_agree(tmpdir):
    """Photoneutron biasing emits one weighted neutron per photon collision
    instead of an integer number at the rare absorptions. The two must give the
    same expected neutron production and the same energy deposition.

    This is the invariant the long comment in
    PhotonuclearInteraction::create_derived() exists to protect: the yield is
    deliberately left out of XS_NEUTRON_PROD because
    sample_photoneutron_product() enumerates products the same way, and
    emit_forced_photoneutron() weights by neutron_prod/total. Change one side
    without the other and these two numbers separate.
    """
    analog = _run(_sphere_model('W186', 19.3, 18.0e6, particles=40000),
                  tmpdir, 'analog')
    biased = _run(_sphere_model('W186', 19.3, 18.0e6, particles=40000,
                                biasing=True), tmpdir, 'biased')

    assert bool(biased['photoneutron_biasing'])

    for quantity in ('neutron flux', 'heating'):
        a_mean, a_std = analog[quantity]
        b_mean, b_std = biased[quantity]
        sigma = np.hypot(a_std, b_std)
        assert sigma > 0.0
        assert abs(a_mean - b_mean) < 4.0*sigma, (
            f'{quantity}: analog {a_mean:.6e} +- {a_std:.2e} vs biased '
            f'{b_mean:.6e} +- {b_std:.2e} differ by '
            f'{abs(a_mean - b_mean)/sigma:.1f} sigma')

    # Biasing exists to cut the variance of photoneutron production, so if it
    # is not doing that something is wrong with the weighting
    assert biased['neutron flux'][1] < analog['neutron flux'][1]


@needs_photonuclear
def test_cell_density_override_scales_photonuclear(tmpdir):
    """A per-cell density override must scale the photonuclear cross section
    the same way it scales the photo-atomic one.

    Material::calculate_photon_xs() applies Cell::density_mult() to the
    photo-atomic loop; if the photonuclear loop does not, macro_xs().total
    mixes a scaled part with an unscaled one, the photonuclear reaction rate is
    wrong by roughly 1/f, and once the photonuclear term exceeds the scaled
    photo-atomic one the cutoff in sample_photon_element() goes negative.
    """
    factor = 0.5

    # Halve the density on the material itself
    halved = _sphere_model('W186', 19.3*factor, 18.0e6, particles=40000)
    reference = _run(halved, tmpdir, 'half_material')

    # Halve it through the cell override instead -- same physics
    overridden = _sphere_model('W186', 19.3, 18.0e6, particles=40000)
    cell = next(iter(overridden.geometry.get_all_cells().values()))
    cell.density = 19.3*factor
    override = _run(overridden, tmpdir, 'half_cell')

    for quantity in ('neutron flux', 'heating'):
        a_mean, a_std = reference[quantity]
        b_mean, b_std = override[quantity]
        sigma = np.hypot(a_std, b_std)
        assert sigma > 0.0
        assert abs(a_mean - b_mean) < 4.0*sigma, (
            f'{quantity}: material density {a_mean:.6e} +- {a_std:.2e} vs '
            f'cell override {b_mean:.6e} +- {b_std:.2e} differ by '
            f'{abs(a_mean - b_mean)/sigma:.1f} sigma')


@needs_photonuclear
def test_photon_ceiling_rejects_high_energy_source(tmpdir):
    """A photonuclear library reaching well past the neutron data would let a
    photon make a neutron nobody can transport. set_photonuclear_bounds()
    lowers the photon ceiling so that photon cannot exist, which turns a
    mid-transport abort into a source-sampling error up front."""
    if not _photonuclear_available('Ag107'):
        pytest.skip('no photonuclear data for Ag107')

    # Ag107 photonuclear data reaches 140 MeV while its neutron data stops at
    # 20 MeV, so the ceiling lands well below the source energy here
    model = _sphere_model('Ag107', 10.5, 1.0e8, particles=200, batches=2)
    with pytest.raises(RuntimeError, match='Source energy above range'):
        model.run(cwd=str(tmpdir.join('ceiling')), output=False)


@needs_photonuclear
def test_photofission_refused_in_eigenvalue_mode(tmpdir):
    """Photofission neutrons contribute to no k-eigenvalue estimator, so the
    combination is refused rather than returning a subtly wrong keff."""
    if not _photonuclear_available('U238'):
        pytest.skip('no photonuclear data for U238')

    model = _sphere_model('U238', 19.0, 1.0e7, particles=100, batches=2,
                          run_mode='eigenvalue')
    with pytest.raises(RuntimeError, match='not supported in k-eigenvalue'):
        model.run(cwd=str(tmpdir.join('eigen')), output=False)
