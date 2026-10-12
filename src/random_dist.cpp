#include "openmc/random_dist.h"

#include <algorithm>
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
  // Beta(a, b) is the distribution of the a-th smallest of n = a + b - 1
  // uniform variates. The m-th largest of n uniform variates is distributed as
  // U_1^(1/n) U_2^(1/(n-1)) ... U_m^(1/(n-m+1)), so the variate needs only
  // min(a, b) uniform variates: it is the b-th largest, or one minus the a-th
  // largest by symmetry.
  int n = a + b - 1;
  int m = std::min(a, b);
  double log_x = 0.0;
  for (int i = 0; i < m; ++i) {
    log_x += std::log(prn(seed)) / (n - i);
  }
  double x = std::exp(log_x);
  return (a <= b) ? 1.0 - x : x;
}

} // namespace openmc
