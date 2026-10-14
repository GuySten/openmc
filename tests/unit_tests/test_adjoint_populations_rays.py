"""Run the ray probes of the adjoint side populations in transport."""

import numpy as np
import openmc

from tests.regression_tests import config


def test_ray_and_probe_fission_weights_agree(run_in_tmpdir):
    """The rays and the probes are emitted from the same recorded fission
    events, so their fission weights, sums of w/k sigma_f/sigma_t over the
    events, must agree in every batch. Under MPI both must be summed over the
    ranks, with the rays' scores normalized by them."""
    model = openmc.Model()

    fuel = openmc.Material()
    fuel.add_nuclide('U235', 1.0)
    fuel.set_density('g/cm3', 18.0)
    heavy_water = openmc.Material()
    heavy_water.add_nuclide('H2', 2.0)
    heavy_water.add_nuclide('O16', 1.0)
    heavy_water.set_density('g/cm3', 1.1)
    model.materials = openmc.Materials([fuel, heavy_water])

    core = openmc.Sphere(r=6.0)
    outer = openmc.Sphere(r=20.0, boundary_type='vacuum')
    model.geometry = openmc.Geometry([
        openmc.Cell(fill=fuel, region=-core),
        openmc.Cell(fill=heavy_water, region=+core & -outer),
    ])

    model.settings.particles = 400
    model.settings.batches = 5
    model.settings.inactive = 2
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point())
    model.settings.photon_transport = True
    model.settings.photonuclear_physics = True
    model.settings.adjoint_populations = {
        'n_generation': 1,
        'photoneutrons': True,
        'photoneutron_probe_energies': [2.4e6, 2.8e6, 3.2e6],
        'photoneutron_ray_comb_points': 3,
    }

    kwargs = {}
    if config['mpi']:
        kwargs['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    sp_path = model.run(**kwargs)

    with openmc.StatePoint(sp_path) as sp:
        pops = sp.adjoint_populations
        probe_f = pops.probe_fission_weight
        ray_f = pops.ray_fission_weight
        ray_w = pops.ray_weight

    assert probe_f.shape == ray_f.shape == (3, 1)
    assert np.all(probe_f > 0.0)
    np.testing.assert_allclose(ray_f, probe_f, rtol=1e-12)
    assert np.all(ray_w.sum(axis=(1, 2, 3)) > 0.0)
