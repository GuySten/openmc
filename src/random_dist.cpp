#include "openmc/random_dist.h"

#include <cmath>

#include "openmc/constants.h"
#include "openmc/random_lcg.h"

namespace openmc {

double uniform_distribution(double a, double b, uint64_t* seed)
{
  return a + (b - a) * prn(seed);
}

int64_t uniform_int_distribution(int64_t a, int64_t b, uint64_t* seed)
{
  return a + static_cast<int64_t>(prn(seed) * (b - a + 1));
}

double maxwell_spectrum(double T, uint64_t* seed)
{
  // Set the random numbers
  double r1 = prn(seed);
  double r2 = prn(seed);
  double r3 = prn(seed);

  // determine cosine of pi/2*r
  double c = std::cos(PI / 2. * r3);

  // Determine outgoing energy
  return -T * (std::log(r1) + std::log(r2) * c * c);
}

double watt_spectrum(double a, double b, uint64_t* seed)
{
  double w = maxwell_spectrum(a, seed);
  return w + 0.25 * a * a * b +
         uniform_distribution(-1., 1., seed) * std::sqrt(a * a * b * w);
}

double normal_variate(double mean, double standard_deviation, uint64_t* seed)
{
  // Sample a normal variate using Marsaglia's polar method
  double x, y, r2;
  do {
    x = uniform_distribution(-1., 1., seed);
    y = uniform_distribution(-1., 1., seed);
    r2 = x * x + y * y;
  } while (r2 > 1 || r2 == 0);
  double z = std::sqrt(-2.0 * std::log(r2) / r2);
  return mean + standard_deviation * z * x;
}

double beta_variate(int a, int b, uint64_t* seed)
{
  // Shapes of one have a closed-form inverse CDF
  if (a == 1 && b == 1) {
    return prn(seed);
  } else if (b == 1) {
    return std::pow(prn(seed), 1.0 / a);
  } else if (a == 1) {
    return 1.0 - std::pow(prn(seed), 1.0 / b);
  }

  // A gamma variate with integer shape n is minus the log of a product of n
  // uniform variates. The product is folded into the sum before it underflows.
  auto gamma_variate = [seed](int n) {
    double x = 0.0;
    double product = 1.0;
    for (int i = 0; i < n; ++i) {
      product *= prn(seed);
      if (product < 1.0e-200) {
        x -= std::log(product);
        product = 1.0;
      }
    }
    return x - std::log(product);
  };
  double x = gamma_variate(a);
  double y = gamma_variate(b);
  return x / (x + y);
}

} // namespace openmc
