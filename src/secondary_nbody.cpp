#include "openmc/secondary_nbody.h"

#include <cmath> // for log, sqrt, pow

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"

namespace openmc {

//==============================================================================
// Helper function to sample from Gamma(0.5, 1) distribution
//==============================================================================

// Gamma(0.5, 1) = (Z^2)/2 where Z ~ N(0,1)
// We use Box-Muller to generate a standard normal
double sample_gamma_half(uint64_t* seed)
{
  double r1 = prn(seed);
  double r2 = prn(seed);
  double z = std::sqrt(-2.0 * std::log(r1)) * std::cos(2.0 * PI * r2);
  return 0.5 * z * z;
}

//==============================================================================
// NBodyPhaseSpace implementation
//==============================================================================

NBodyPhaseSpace::NBodyPhaseSpace(hid_t group)
{
  read_attribute(group, "n_particles", n_bodies_);
  read_attribute(group, "total_mass", mass_ratio_);
  read_attribute(group, "atomic_weight_ratio", A_);
  read_attribute(group, "q_value", Q_);
}

void NBodyPhaseSpace::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
  mu = uniform_distribution(-1., 1., seed);

  // Determine E_max parameter (Eq. 6.27 in ENDF-6 manual)
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * (A_ / (A_ + 1.0) * E_in + Q_);

  // The energy distribution follows Eq. 6.21 in ENDF-6 manual:
  // P(E') ~ sqrt(E') * (E_max - E')^((3n/2) - 4)
  //
  // This is sampled using v = x/(x+y) where:
  // - x ~ Gamma(3/2, 1) = Maxwell spectrum
  // - y ~ Gamma((3n-6)/2, 1) = Gamma(3(n-2)/2, 1)

  // x ~ Gamma(3/2, 1) = Maxwell spectrum
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 2:
    // For n=2, shape = 0, so y -> 0 and v -> 1
    // Use uniform distribution as a reasonable approximation
    E_out = E_max * prn(seed);
    return;
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
    // General case for n > 5 (or any n not handled above)
    // y ~ Gamma(shape, 1) where shape = 3(n-2)/2
    // Sample as: integer part via sum of exponentials, plus Gamma(0.5) if needed
    {
      int n_integer = (3 * (n_bodies_ - 2)) / 2;
      bool has_half = ((3 * (n_bodies_ - 2)) % 2) == 1;

      if (n_integer > 0) {
        double product = 1.0;
        for (int i = 0; i < n_integer; ++i) {
          product *= prn(seed);
        }
        y = -std::log(product);
      } else {
        y = 0.0;
      }

      if (has_half) {
        y += sample_gamma_half(seed);
      }
    }
    break;
  }

  // Now determine v and E_out
  double v = x / (x + y);
  E_out = E_max * v;
}

} // namespace openmc
