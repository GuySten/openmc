#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>

#include "openmc/condensed_history.h"
#include "openmc/constants.h"
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

// What bounds a step, and the decision -- taken step by step, from nothing but
// the step itself -- of whether to group at all.
TEST_CASE("the step is bounded by collisions, by energy and by geometry")
{
  uint64_t seed = 99;
  double xs_hard = 1.0;   // one hard collision per cm
  double xs_soft = 1.0e5; // plenty to group
  double E = 1.0e7;
  double c2 = 0.05;
  double max_loss = c2 * E;

  SECTION("a hard collision ends the ordinary step")
  {
    // With no stopping power and no boundary in the way, the length is
    // exponentially distributed about the hard mean free path
    double sum = 0.0;
    int n = 200000;
    bool hinge_inside = true;
    bool always_grouped = true;
    for (int i = 0; i < n; ++i) {
      auto s = openmc::sample_mixed_step(
        xs_hard, xs_soft, 0.0, 0.0, max_loss, 0.0, openmc::INFTY, &seed);
      always_grouped = always_grouped && s.grouped;
      hinge_inside = hinge_inside && s.hinge >= 0.0 && s.hinge <= s.length;
      sum += s.length;
    }
    CHECK(hinge_inside);
    // Whether to group is decided before the length is drawn, so a run of
    // short steps cannot turn some of them into single-event flights -- which
    // is what would bias the mean below
    CHECK(always_grouped);
    CHECK_THAT(sum / n, WithinRel(1.0 / xs_hard, 0.02));
  }

  SECTION("the energy ceiling shortens it")
  {
    // A stopping power steep enough that c2 of the energy is gone in a
    // hundredth of the hard mean free path
    double stopping_power = c2 * E / 0.01;
    for (int i = 0; i < 1000; ++i) {
      auto s = openmc::sample_mixed_step(xs_hard, xs_soft, 0.0, stopping_power,
        max_loss, 0.0, openmc::INFTY, &seed);
      CHECK(s.length <= 0.01 * (1.0 + 1.0e-12));
    }
  }

  SECTION("geometry has the last word")
  {
    for (int i = 0; i < 1000; ++i) {
      auto s = openmc::sample_mixed_step(
        xs_hard, xs_soft, 0.0, 0.0, max_loss, 0.0, 0.001, &seed);
      CHECK(s.length <= 0.001);
      CHECK(!s.ends_in_collision);
    }
  }
}

TEST_CASE("a step with too little in it is not grouped")
{
  uint64_t seed = 4242;
  double E = 1.0e7;

  // Nothing soft to group, which is what C1 = 0 and no cutoffs leave behind:
  // the scheme must hand every step back to the single-event transport
  for (int i = 0; i < 1000; ++i) {
    auto s = openmc::sample_mixed_step(
      1.0, 0.0, 0.0, 0.0, 0.05 * E, 0.0, openmc::INFTY, &seed);
    CHECK(!s.grouped);
  }

  // A foil thin enough that the step holds fewer than the minimum number of
  // collisions is transported one collision at a time, however the run was
  // configured. This is the whole of the switch, and it is made from the step
  // rather than from the geometry.
  double xs_soft = 1.0e4;
  double thin = 0.5 * openmc::MIN_GROUPED_COLLISIONS / xs_soft;
  double thick = 2.0 * openmc::MIN_GROUPED_COLLISIONS / xs_soft;
  for (int i = 0; i < 1000; ++i) {
    CHECK(!openmc::sample_mixed_step(
      1.0, xs_soft, 0.0, 0.0, 0.05 * E, 0.0, thin, &seed)
             .grouped);
    CHECK(openmc::sample_mixed_step(
      1.0, xs_soft, 0.0, 0.0, 0.05 * E, 0.0, thick, &seed)
            .grouped);
  }
}

TEST_CASE("the grouping decision does not bias the step length")
{
  // Sitting just above the threshold is where deciding from a sampled length
  // would do its damage: the short steps would be handed to the single-event
  // path and the ones left behind would be drawn from an exponential with its
  // lower end removed, which is a longer mean free path than the cross
  // section says. The mean has to come back as the mean free path itself.
  uint64_t seed = 31337;
  double xs_hard = 1.0;
  double xs_soft = 1.1 * openmc::MIN_GROUPED_COLLISIONS * xs_hard;
  int n = 400000;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    auto s = openmc::sample_mixed_step(
      xs_hard, xs_soft, 0.0, 0.0, 0.05 * 1.0e7, 0.0, openmc::INFTY, &seed);
    REQUIRE(s.grouped);
    sum += s.length;
  }
  CHECK_THAT(sum / n, WithinRel(1.0 / xs_hard, 0.01));
}

TEST_CASE("a step cannot carry the projectile past its own cutoff")
{
  // The energy ceiling is not only C2. A step that took a charged particle
  // below its transport cutoff would carry it past the point where it should
  // have stopped where it was -- and a positron past the energy at which it
  // should have annihilated, which is the whole reason a photoneutron run sets
  // that cutoff below the others. So the loss is capped by the headroom above
  // the cutoff too, and near the cutoff that is the smaller of the two.
  uint64_t seed = 8080;
  double stopping_power = 1.0e6; // eV/cm
  double E = 1.05e7;
  double cutoff = 1.0e7;
  double headroom = E - cutoff;
  double max_loss = std::min(0.05 * E, headroom);
  CHECK(
    max_loss == headroom); // C2 alone would allow 5.25e5 eV, past the cutoff

  for (int i = 0; i < 1000; ++i) {
    auto s = openmc::sample_mixed_step(
      1.0e-6, 1.0e6, 0.0, stopping_power, max_loss, 0.0, openmc::INFTY, &seed);
    REQUIRE(s.grouped);
    CHECK(s.length * stopping_power <= headroom * (1.0 + 1.0e-12));
  }
}

TEST_CASE("the step bounds have a coarsest setting")
{
  // Both parameters are quality knobs and both stop meaning anything at the
  // same place, which is where PENELOPE caps them and PenRed enforces it.
  // Nothing here reads settings, so what is checked is that the limit exists,
  // is the same for both, and leaves the default well inside it.
  CHECK(openmc::MAX_STEP_COARSENESS == 0.2);
  CHECK(openmc::MAX_STEP_COARSENESS > 0.0);

  // A step at the limit turns the particle through about 37 degrees
  double theta = std::acos(1.0 - openmc::MAX_STEP_COARSENESS) * 180.0 / M_PI;
  CHECK(theta > 36.0);
  CHECK(theta < 38.0);

  // and the artificial distribution still has to carry its moments there
  double mu1 = std::exp(-openmc::MAX_STEP_COARSENESS);
  double mu2 = (2.0 * std::exp(-3.0 * openmc::MAX_STEP_COARSENESS) + 1.0) / 3.0;
  auto d = openmc::soft_scattering(mu1, mu2);
  CHECK(d.mu_0 >= -1.0);
  CHECK(d.mu_0 <= 1.0);
  CHECK(d.a >= 0.0);
  CHECK(d.a <= 1.0);
}

TEST_CASE("Step majorant bounds the cross section it is drawn from")
{
  // The property the whole delta-interaction scheme rests on: over every
  // energy a step beginning at a grid point can reach, the bound is not below
  // what the transport would read there. Reading is an interpolation between
  // tabulated points, so the check is against the interpolated value and not
  // only against the points -- which is exactly the distinction a scan that
  // stops at the first point above `lowest` gets wrong.
  //
  // A hard cross section rises as the energy falls: the lower the energy, the
  // less of the collision spectrum is soft enough to group. That is the shape
  // the table is given here, since it is the shape that makes an off-by-one
  // bite, and a scan that stopped one point short would return a bound below
  // the interpolated value over the last interval every time.
  int n = 40;
  openmc::vector<double> energy(n), hard(n), lowest(n);
  for (int j = 0; j < n; ++j) {
    energy[j] = 1.0e4 * std::pow(10.0, 3.0 * j / (n - 1.0)); // 10 keV .. 10 MeV
    hard[j] = 1.0e6 / energy[j];
    lowest[j] = energy[j] * 0.8; // a step may lose a fifth of the energy
  }

  auto majorant = openmc::step_majorant(energy, hard, lowest, 1.0);

  auto interpolate = [&](double E) {
    if (E <= energy[0])
      return hard[0];
    int k = 0;
    while (k + 1 < n && energy[k + 1] < E)
      ++k;
    double f = (E - energy[k]) / (energy[k + 1] - energy[k]);
    return hard[k] + f * (hard[k + 1] - hard[k]);
  };

  for (int j = 0; j < n; ++j) {
    // Walk the reachable range finely rather than checking the grid points,
    // since the grid points are what a too-short scan already agrees with
    for (int i = 0; i <= 50; ++i) {
      double E = lowest[j] + (energy[j] - lowest[j]) * i / 50.0;
      CHECK(majorant[j] >= interpolate(E));
    }
  }

  // and it is a bound, not just any large number. The scan deliberately runs
  // one grid point past `lowest`, since that is the point the interpolation
  // over the last interval needs, so the tightest it can be is the cross
  // section there -- and it should be exactly that, never more.
  for (int j = 1; j < n; ++j) {
    int k = j;
    while (k > 0 && energy[k] > lowest[j])
      --k;
    CHECK(majorant[j] == hard[k]);
  }
}

TEST_CASE("Step majorant covers a step that overshoots its energy budget")
{
  // The step is aimed at losing no more than its budget on average, but the
  // sampled loss overshoots the mean, so the scan has to reach past the budget
  // by MAX_SOFT_LOSS_OVERSHOOT. The two branches of sample_soft_energy_loss
  // are what that number is derived from, so check the derivation still holds
  // at the variance the soft cutoff allows: var <= MAX_SOFT_LOSS_SHARE mean^2.
  double mean = 1.0;
  double variance = openmc::MAX_SOFT_LOSS_SHARE * mean * mean;

  // Uniform branch, whose largest loss is mean + sqrt(3 var)
  REQUIRE(3.0 * variance <= mean * mean);
  double widest = mean + std::sqrt(3.0 * variance);
  CHECK(widest <= openmc::MAX_SOFT_LOSS_OVERSHOOT * mean);

  // The other branch takes over past that, and reaches further
  double var_atom = mean * mean / 2.0;
  REQUIRE(3.0 * var_atom > mean * mean);
  double w = 1.5 * (mean * mean + var_atom) / mean;
  CHECK(w > widest);

  // The sampler never returns more than the branch it is in allows
  uint64_t seed = 7;
  for (int i = 0; i < 20000; ++i) {
    double loss = openmc::sample_soft_energy_loss(mean, variance, &seed);
    CHECK(loss >= 0.0);
    CHECK(loss <= openmc::MAX_SOFT_LOSS_OVERSHOOT * mean);
  }
}
