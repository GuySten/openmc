"""Random ray in fixed-source mode with create_fission_neutrons = False: fission
is absorption, exactly as if the multigroup data had no fission neutrons."""

import numpy as np
import openmc
from openmc.examples import random_ray_lattice


def run_lattice(nu_fission_zero, create_fission_neutrons):
    openmc.reset_auto_ids()
    model = random_ray_lattice()
    if nu_fission_zero:
        lib = openmc.MGXSLibrary.from_hdf5('mgxs.h5')
        uo2 = lib.get_by_name('UO2')
        for i in range(len(uo2.temperatures)):
            uo2._nu_fission[i] = np.zeros_like(uo2._nu_fission[i])
        lib.export_to_hdf5('mgxs.h5')
    fuel = next(m for m in model.materials if m.name == 'UO2 fuel')
    s = model.settings
    s.run_mode = 'fixed source'
    s.create_fission_neutrons = create_fission_neutrons
    s.batches = 20
    s.inactive = 10
    s.particles = 50
    s.source = [openmc.IndependentSource(
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [fuel]})]
    sp_path = model.run()
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(name='Mesh tally')
        return tally.get_values(scores=['flux']).ravel().copy()


def test_create_fission_neutrons_is_absorption(run_in_tmpdir):
    # Fission kept in the data but not producing neutrons ...
    flag = run_lattice(nu_fission_zero=False, create_fission_neutrons=False)
    # ... is the same problem as data without fission neutrons
    zeroed = run_lattice(nu_fission_zero=True, create_fission_neutrons=True)
    # and differs from the multiplying problem
    multiplying = run_lattice(nu_fission_zero=False, create_fission_neutrons=True)
    assert np.allclose(flag, zeroed, rtol=1.0e-10, atol=0.0)
    assert not np.allclose(flag, multiplying, rtol=1.0e-3)
    assert multiplying.sum() > flag.sum()
