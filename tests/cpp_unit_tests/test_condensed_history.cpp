#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>

#include "openmc/condensed_history.h"
#include "openmc/math_functions.h"
#include "openmc/position.h"
#include "openmc/random_lcg.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// The moments the artificial distribution actually carries, integrated rather
// than sampled: two uniform pieces, so both are elementary.
void moments_of(const openmc::SoftScattering& d, double& mu1, double& mu2)
{
  double lo_1 = 0.5 * (d.mu_0 - 1.0);
  double hi_1 = 0.5 * (d.mu_0 + 1.0);
  double lo_2 = (1.0 - d.mu_0 + d.mu_0 * d.mu_0) / 3.0;
  double hi_2 = (1.0 + d.mu_0 + d.mu_0 * d.mu_0) / 3.0;
  mu1 = d.a * lo_1 + (1.0 - d.a) * hi_1;
  mu2 = d.a * lo_2 + (1.0 - d.a) * hi_2;
}

} // namespace

// A class II step replaces the grouped elastic collisions by one artificial
// deflection. All that is asked of it is the first two Legendre moments, so
// the thing to check is that it carries them -- not what shape it has.
TEST_CASE("the soft deflection carries the moments it was built from")
{
  // Optical depths spanning a step far shorter than a transport mean free
  // path up to one far longer, with sigma_2 between sigma_1 and 3 sigma_1,
  // which is the whole range a real angular distribution can produce
  for (double s1 : {1.0e-6, 1.0e-3, 0.03, 0.3, 3.0, 30.0}) {
    for (double ratio : {1.0, 1.5, 2.0, 2.9, 3.0}) {
      double mu1 = std::exp(-s1);
      double mu2 = (2.0 * std::exp(-ratio * s1) + 1.0) / 3.0;

      auto d = openmc::soft_scattering(mu1, mu2);
      CHECK(d.mu_0 >= -1.0);
      CHECK(d.mu_0 <= 1.0);
      CHECK(d.a >= 0.0);
      CHECK(d.a <= 1.0);

      // Both moments are bounded by one, so an absolute tolerance is what
      // means anything: at thirty transport mean free paths <mu> is 1e-13 and
      // <P_2(mu)> has underflowed into 1/3, which is isotropy, and asking for
      // relative agreement there would be asking about rounding rather than
      // about the distribution.
      double got1, got2;
      moments_of(d, got1, got2);
      CHECK_THAT(got1, WithinAbs(mu1, 1.0e-10));
      CHECK_THAT(got2, WithinAbs(mu2, 1.0e-10));
    }
  }
}

TEST_CASE("a vanishing step leaves the direction alone")
{
  // The forward limit has to be exact rather than nearly so: a mixed scheme
  // is checked against the single-event transport it reduces to, and a step
  // that deflects a particle by anything at all as its length goes to zero
  // would never reduce to it.
  uint64_t seed = 1;
  for (int i = 0; i < 1000; ++i) {
    CHECK(openmc::sample_soft_deflection(0.0, 0.0, &seed) == 1.0);
  }
}

TEST_CASE("a long step relaxes to isotropy")
{
  // Once the step is many transport mean free paths the direction has to have
  // forgotten where it came from
  auto d = openmc::soft_scattering(
    std::exp(-40.0), (2.0 * std::exp(-120.0) + 1.0) / 3.0);
  double mu1, mu2;
  moments_of(d, mu1, mu2);
  CHECK_THAT(mu1, WithinAbs(0.0, 1.0e-12));
  CHECK_THAT(mu2, WithinRel(1.0 / 3.0, 1.0e-10));
}

TEST_CASE("the sampled deflection reproduces its own moments")
{
  // The integrals above say the distribution carries the moments; this says
  // the sampling of it does too, which is a different claim and the one the
  // transport depends on.
  uint64_t seed = 12345;
  for (double s1 : {1.0e-3, 0.1, 1.0, 10.0}) {
    double s2 = 2.5 * s1;
    int n = 400000;
    double sum = 0.0;
    double sum_sq = 0.0;
    double lowest = 1.0;
    double highest = -1.0;
    for (int i = 0; i < n; ++i) {
      double mu = openmc::sample_soft_deflection(s1, s2, &seed);
      lowest = std::min(lowest, mu);
      highest = std::max(highest, mu);
      sum += mu;
      sum_sq += mu * mu;
    }
    CHECK(lowest >= -1.0);
    CHECK(highest <= 1.0);
    double mu1 = std::exp(-s1);
    double mu2 = (2.0 * std::exp(-s2) + 1.0) / 3.0;
    // Three standard errors of a quantity bounded by one, over n samples
    double tol = 3.0 / std::sqrt(static_cast<double>(n));
    CHECK_THAT(sum / n, WithinAbs(mu1, tol));
    CHECK_THAT(sum_sq / n, WithinAbs(mu2, tol));
  }
}

// The soft energy loss is the other half of the step: a restricted stopping
// power says what it loses on average, and the second moment says how much
// that varies, which is the straggling grouping would otherwise throw away.
TEST_CASE("the soft energy loss carries its mean and variance")
{
  uint64_t seed = 54321;
  double mean = 1000.0;
  // Both branches: narrow enough for a uniform distribution to fit inside
  // the positive axis, and broad enough that it cannot
  for (double variance : {1.0e2, 1.0e4, mean * mean / 3.0, 1.0e6, 1.0e8}) {
    int n = 400000;
    double sum = 0.0;
    double sum_sq = 0.0;
    double lowest = mean;
    for (int i = 0; i < n; ++i) {
      double w = openmc::sample_soft_energy_loss(mean, variance, &seed);
      lowest = std::min(lowest, w);
      sum += w;
      sum_sq += w * w;
    }
    CHECK(lowest >= 0.0);
    double got_mean = sum / n;
    double got_var = sum_sq / n - got_mean * got_mean;
    double stderr_mean = std::sqrt(variance / n);
    CHECK_THAT(got_mean, WithinAbs(mean, 4.0 * stderr_mean));
    CHECK_THAT(got_var, WithinRel(variance, 0.05));
  }
}

TEST_CASE("a soft loss with no spread is its own mean")
{
  // Atomic excitation on its own is a single loss per collision, so a step
  // dominated by it has almost no spread and must not acquire one here
  uint64_t seed = 7;
  CHECK(openmc::sample_soft_energy_loss(500.0, 0.0, &seed) == 500.0);
  CHECK(openmc::sample_soft_energy_loss(0.0, 1.0, &seed) == 0.0);
  CHECK(openmc::sample_soft_energy_loss(-1.0, 1.0, &seed) == 0.0);
}

// The property the whole scheme rests on: a step is allowed to stand in for
// the collisions inside it only if splitting it in two changes nothing. The
// moments decay exponentially in path length, so two steps of s must compose
// to one of 2s -- and because the artificial distribution is nothing like the
// true one, that is a real constraint on it rather than a restatement.
TEST_CASE("short steps compose into a long one")
{
  uint64_t seed = 2024;
  int n = 2000000;

  for (double s_total : {0.05, 0.5, 2.0}) {
    double ratio = 2.5;
    for (int pieces : {1, 2, 8}) {
      double s1 = s_total / pieces;
      double sum = 0.0;
      double sum_p2 = 0.0;
      for (int i = 0; i < n; ++i) {
        openmc::Direction u {0.0, 0.0, 1.0};
        for (int k = 0; k < pieces; ++k) {
          double mu = openmc::sample_soft_deflection(s1, ratio * s1, &seed);
          u = openmc::rotate_angle(u, mu, nullptr, &seed);
        }
        sum += u.z;
        sum_p2 += 0.5 * (3.0 * u.z * u.z - 1.0);
      }
      // Whatever the step is cut into, the accumulated deflection has to be
      // the one the transport cross sections prescribe for the whole path
      double tol = 4.0 / std::sqrt(static_cast<double>(n));
      CHECK_THAT(sum / n, WithinAbs(std::exp(-s_total), tol));
      CHECK_THAT(sum_p2 / n, WithinAbs(std::exp(-ratio * s_total), tol));
    }
  }
}
