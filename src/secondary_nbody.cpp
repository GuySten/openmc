#include "openmc/secondary_nbody.h"

#include <cmath>  // for log
#include <string> // for string

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/particle_type.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"

namespace openmc {

//==============================================================================
// NBodyPhaseSpace implementation
//==============================================================================

NBodyPhaseSpace::NBodyPhaseSpace(hid_t group)
{
  read_attribute(group, "n_particles", n_bodies_);
  read_attribute(group, "total_mass", mass_ratio_);
  read_attribute(group, "atomic_weight_ratio", A_);
  read_attribute(group, "q_value", Q_);

  // Files written before the incident particle was recorded are neutron data
  if (attribute_exists(group, "particle")) {
    std::string temp;
    read_attribute(group, "particle", temp);
    auto type = ParticleType(temp);
    if (type.is_photon()) {
      photon_ = true;
    } else if (!type.is_neutron()) {
      fatal_error("Unrecognized particle: " + type.str());
    }
  }
}

double NBodyPhaseSpace::available_energy(double E_in) const
{
  // Kinetic energy in the center-of-mass system, non-relativistic. A massless
  // photon carries momentum E/c, so the center of mass moves with kinetic
  // energy E^2/(2 M c^2) where M is the target mass.
  if (photon_) {
    return E_in - E_in * E_in / (2.0 * MASS_NEUTRON_EV * A_) + Q_;
  } else {
    return A_ / (A_ + 1.0) * E_in + Q_;
  }
}

double NBodyPhaseSpace::sample_energy(double E_in, uint64_t* seed) const
{
  // Determine E_max parameter
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * available_energy(E_in);

  // x is essentially a Maxwellian distribution
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 3:
    y = maxwell_spectrum(1.0, seed);
    break;
  case 4:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    y = -std::log(r1 * r2 * r3);
    break;
  case 5:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    r4 = prn(seed);
    r5 = prn(seed);
    r6 = prn(seed);
    y = -std::log(r1 * r2 * r3 * r4) -
        std::log(r5) * std::pow(std::cos(PI / 2.0 * r6), 2);
    break;
  default:
    fatal_error("N-body phase space with >5 bodies.");
  }

  // Now determine v and E_out
  double v = x / (x + y);
  return E_max * v;
}

void NBodyPhaseSpace::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
  mu = uniform_distribution(-1., 1., seed);

  E_out = sample_energy(E_in, seed);
}

double NBodyPhaseSpace::sample_energy_and_pdf(
  double E_in, double mu, double& E_out, uint64_t* seed) const
{
  E_out = sample_energy(E_in, seed);
  return 0.5;
}

double NBodyPhaseSpace::max_energy(double E_in) const
{
  // sample_energy() returns E_max * v with v = x/(x+y) in [0,1)
  double Ap = mass_ratio_;
  return (Ap - 1.0) / Ap * available_energy(E_in);
}

} // namespace openmc
