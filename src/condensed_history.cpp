#include "openmc/condensed_history.h"

#include <algorithm> // for max, min
#include <cmath>     // for abs, exp, sqrt

#include "openmc/random_lcg.h"

namespace openmc {

SoftScattering soft_scattering(double mu1, double mu2)
{
  SoftScattering s;

  // Nothing is left to interpolate between when the step is so long that the
  // direction has forgotten where it came from
  if (!(std::abs(mu1) > 0.0)) {
    s.mu_0 = 0.0;
    s.a = 0.5;
    return s;
  }

  double mu_0 = (3.0 * mu2 - 1.0) / (2.0 * mu1);

  // The solution is confined to the range over which the piece below mu_0
  // carries a probability between zero and one, and the exact answer sits on
  // one end of it whenever sigma_2 reaches an end of its own physical range:
  // mu_0 = 1 when sigma_2 = sigma_1, and a = 0 when sigma_2 = 3 sigma_1. So
  // rounding puts it a couple of ulps outside often enough to matter, and the
  // range is clamped to rather than tested against. Clamping costs the second
  // moment a little where it is not reachable at all and costs the first
  // nothing anywhere, which is the right way round: the first moment is what
  // sets the transport mean free path.
  double lower = std::max(-1.0, 2.0 * mu1 - 1.0);
  s.mu_0 = std::max(lower, std::min(1.0, mu_0));
  s.a = 0.5 * (1.0 + s.mu_0 - 2.0 * mu1);
  s.a = std::max(0.0, std::min(1.0, s.a));
  return s;
}

double sample_soft_deflection(
  double s_lambda1, double s_lambda2, uint64_t* seed)
{
  // Moments of the accumulated deflection after the step
  double mu1 = std::exp(-std::max(0.0, s_lambda1));
  double p2 = std::exp(-std::max(0.0, s_lambda2));
  double mu2 = (2.0 * p2 + 1.0) / 3.0;

  SoftScattering d = soft_scattering(mu1, mu2);

  // Below mu_0 the piece runs down to -1, above it up to 1
  double xi = prn(seed);
  double mu;
  if (xi < d.a) {
    double u = (d.a > 0.0) ? xi / d.a : 0.0;
    mu = -1.0 + u * (d.mu_0 + 1.0);
  } else {
    double u = (d.a < 1.0) ? (xi - d.a) / (1.0 - d.a) : 0.0;
    mu = d.mu_0 + u * (1.0 - d.mu_0);
  }
  return std::max(-1.0, std::min(1.0, mu));
}

double sample_soft_energy_loss(double mean, double variance, uint64_t* seed)
{
  if (!(mean > 0.0))
    return 0.0;
  if (!(variance > 0.0))
    return mean;

  // A uniform distribution centred on the mean reaches a variance of
  // mean^2/3 as its lower end arrives at zero. While it fits, use it.
  if (3.0 * variance <= mean * mean) {
    double half_width = std::sqrt(3.0 * variance);
    return std::max(0.0, mean - half_width + 2.0 * half_width * prn(seed));
  }

  // Past that the loss is broad enough that most steps must lose nothing at
  // all: an atom at zero of probability a, and a uniform piece up to w.
  //   (1-a) w / 2 = mean,   (1-a) w^2 / 3 = mean^2 + variance
  double w = 1.5 * (mean * mean + variance) / mean;
  double one_minus_a = 2.0 * mean / w;
  double xi = prn(seed);
  if (xi >= one_minus_a)
    return 0.0;
  return w * (xi / one_minus_a);
}

} // namespace openmc
