import os
import numpy as np

import openmc
import openmc.stats

from tests.testing_harness import PyAPITestHarness

# Path to charged particle data in the test data directory
CHARGED_PARTICLE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'tests', 'cpp_unit_tests', 'data'
)


class ChargedParticleTestHarness(PyAPITestHarness):
    def _get_results(self):
        """Digest info in the statepoint and return as a string."""
        outstr = ''
        with openmc.StatePoint(self._sp_name) as sp:
            # Write out tally data
            for i, tally_ind in enumerate(sp.tallies):
                tally = sp.tallies[tally_ind]
                results = np.zeros((tally.sum.size * 2,))
                results[0::2] = tally.sum.ravel()
                results[1::2] = tally.sum_sq.ravel()
                results = ['{0:12.6E}'.format(x) for x in results]

                outstr += 'tally ' + str(i + 1) + ':\n'
                outstr += '\n'.join(results) + '\n'

        return outstr


def test_charged_particle():
    """Test D-T fusion: 100 keV deuterons on tritium target."""

    # Tritium target material at 0.1 g/cc
    mat = openmc.Material()
    mat.add_nuclide('H3', 1.0)
    mat.set_density('g/cc', 0.1)
    # Enable constant stopping power model (low enough to allow collisions)
    mat.stopping_power_model = 'constant'
    mat.stopping_power_constant = 1.0e4  # 10 keV/cm

    # Large sphere to approximate infinite medium
    surf = openmc.Sphere(r=1.0e90, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-surf)

    model = openmc.model.Model()
    model.geometry.root_universe = openmc.Universe(cells=[cell])
    model.materials.append(mat)

    # Fixed source settings with deuteron source at 100 keV
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 1
    model.settings.particles = 10
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([100.0e3], [1.0]),  # 100 keV
        particle='deuteron'
    )

    # Enable deuteron transport with energy cutoff and data path
    model.settings.deuteron_transport = True
    model.settings.cutoff = {'energy_deuteron': 20.0}  # 20 eV cutoff
    model.settings.charged_particle_path = CHARGED_PARTICLE_DATA_PATH

    # Tally neutron spectrum from D-T fusion
    # Expected: ~14.1 MeV neutrons
    particle_filter = openmc.ParticleFilter(['neutron'])
    energy_filter = openmc.EnergyFilter([0.0, 1.0e6, 5.0e6, 10.0e6, 15.0e6, 20.0e6])

    tally = openmc.Tally()
    tally.filters = [particle_filter, energy_filter]
    tally.scores = ['flux']
    model.tallies.append(tally)

    harness = ChargedParticleTestHarness('statepoint.1.h5', model)
    harness.main()
