#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// The kinematics an inelastic collision is built on. Both polar angles follow
// from the recoil energy Q alone; these are the identities that makes that
// true, written here as the transport writes them so that a change to either
// form has to be deliberate.
namespace {

constexpr double TWO_M = 2.0 * openmc::MASS_ELECTRON_EV;

double pc_of(double E)
{
  return std::sqrt(E * (E + TWO_M));
}

//! Smallest momentum transfer, in the cancellation-free form the transport
//! uses: (pc)^2 - (pc')^2 = W(2E - W + 2mc^2) exactly
double cq_min_of(double E, double W)
{
  return W * (2.0 * E - W + TWO_M) / (pc_of(E) + pc_of(E - W));
}

//! Smallest recoil energy, likewise without subtracting near-equal terms
double q_min_of(double E, double W)
{
  double cq = cq_min_of(E, W);
  return cq * cq /
         (std::sqrt(
            openmc::MASS_ELECTRON_EV * openmc::MASS_ELECTRON_EV + cq * cq) +
           openmc::MASS_ELECTRON_EV);
}

//! Deflection of the projectile, as Element::emit_knock_on computes it
double one_minus_mu(double E, double W, double Q)
{
  double pc = pc_of(E);
  double pc_out = pc_of(E - W);
  double cq_min = cq_min_of(E, W);
  return (Q * (Q + TWO_M) - cq_min * cq_min) / (2.0 * pc * pc_out);
}

//! Deflection of the knock-on, which leaves along the momentum transfer
double one_minus_mu_knock(double E, double W, double Q)
{
  double pc = pc_of(E);
  double pc_out = pc_of(E - W);
  double cq = std::sqrt(Q * (Q + TWO_M));
  double cq_min = cq_min_of(E, W);
  return (cq - cq_min) * (pc + pc_out - cq) / (2.0 * pc * cq);
}

//! The free binary collision the general form must reduce to
double moller_mu(double E, double E_prime)
{
  return std::sqrt((1.0 + TWO_M / E) / (1.0 + TWO_M / E_prime));
}

} // namespace

TEST_CASE("inelastic kinematics reduce to the free collision at Q = W")
{
  // A free electron takes up the whole transfer, and then both angles must be
  // the familiar Moller pair. This is what makes the general form safe to use
  // for close collisions.
  for (double E : {1.0e4, 1.0e6, 4.27e6, 2.2e7}) {
    for (double frac : {1.0e-5, 1.0e-3, 0.1, 0.49}) {
      double W = frac * E;
      double mu = 1.0 - one_minus_mu(E, W, W);
      double mu_k = 1.0 - one_minus_mu_knock(E, W, W);
      CHECK_THAT(mu, WithinRel(moller_mu(E, E - W), 1.0e-12));
      CHECK_THAT(mu_k, WithinRel(moller_mu(E, W), 1.0e-10));
    }
  }
}

TEST_CASE("the smallest recoil leaves both particles undeflected")
{
  // Q_min is the recoil left when the projectile is not deflected at all, so
  // both angles must vanish there identically -- not merely to rounding. The
  // distant transverse channel returns exactly this value.
  for (double E : {1.0e4, 1.0e6, 4.27e6, 2.2e7}) {
    for (double W : {1.0, 38.0, 812.0}) {
      if (W >= E)
        continue;
      double q_min = q_min_of(E, W);
      CHECK_THAT(one_minus_mu(E, W, q_min), WithinAbs(0.0, 1.0e-12));
      CHECK_THAT(one_minus_mu_knock(E, W, q_min), WithinAbs(0.0, 1.0e-10));
    }
  }
}

TEST_CASE("a distant collision deflects less than a close one")
{
  // The whole point of the split: below the oscillator resonance the atom is
  // excited as a whole and the recoil is far smaller than the binary value.
  double E = 4.27e6;
  double W = 38.0;
  double w_r = 25.8; // carbon L3 resonance in graphite
  CHECK(one_minus_mu(E, W, w_r) < one_minus_mu(E, W, W));
}

TEST_CASE("the longitudinal recoil inversion stays inside its bounds")
{
  // Q is drawn from 1/(Q(Q+2mc^2)) between the smallest recoil and the
  // resonance, by inverting the logarithm in closed form. The endpoints must
  // come back exactly, or a sampled recoil can leave the physical range.
  double E = 4.27e6;
  double W = 20.0;
  double w_r = 25.8;
  double q_min = q_min_of(E, W);

  double c_lon = std::log(w_r * (q_min + TWO_M) / (q_min * (w_r + TWO_M)));
  auto invert = [&](double xi) {
    double a = std::exp(xi * c_lon);
    return TWO_M * a * q_min / (q_min + TWO_M - a * q_min);
  };

  CHECK_THAT(invert(0.0), WithinRel(q_min, 1.0e-12));
  CHECK_THAT(invert(1.0), WithinRel(w_r, 1.0e-10));
  for (double xi : {0.1, 0.25, 0.5, 0.75, 0.9}) {
    double q = invert(xi);
    CHECK(q > q_min);
    CHECK(q < w_r);
  }
}

TEST_CASE("the Salvat positron bremsstrahlung factor is a ratio below one")
{
  // A positron radiates less than an electron of the same energy, and the two
  // must agree in the ultrarelativistic limit.
  for (int z : {6, 13, 74, 82}) {
    double z_sq = static_cast<double>(z) * z;
    double previous = 0.0;
    for (double E : {1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8}) {
      double f = openmc::salvat_factor(z_sq, E);
      CHECK(f > 0.0);
      CHECK(f <= 1.0);
      CHECK(f >= previous); // rises monotonically toward one
      previous = f;
    }
    // The fit approaches one from below, more slowly for heavy elements
    CHECK(openmc::salvat_factor(z_sq, 1.0e9) > 0.99);
  }
}
