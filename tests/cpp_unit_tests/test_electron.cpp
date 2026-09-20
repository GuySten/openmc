#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "openmc/bremsstrahlung.h"
#include "openmc/condensed_history.h"
#include "openmc/constants.h"
#include "openmc/distribution_angle.h"
#include "openmc/gos.h"
#include "openmc/material.h"
#include "openmc/particle_data.h"
#include "openmc/photon.h"
#include "openmc/settings.h"

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

// The transport moments an eventual condensed-history step length is built
// from. They are integrals of the tabulated angular distribution, so what
// matters is that the quadrature is exact for the interpolation law -- these
// distributions are forward-peaked enough that a trapezoidal error in
// <1-mu> would show up directly in the step size.
TEST_CASE("elastic transport moments are exact for a linear density")
{
  // p(mu) = (1 + mu)/2 on [-1, 1], normalised, with
  //   <1 - mu>          = 1 - 1/3 = 2/3
  //   <(3/2)(1 - mu^2)> = (3/2)(1 - 1/3) = 1
  // Both are polynomial in mu, so an exact per-segment integration returns
  // them for ANY partition of the interval; a trapezoidal one does not.
  for (int n : {3, 5, 17}) {
    std::vector<double> x(n), p(n);
    for (int i = 0; i < n; ++i) {
      x[i] = -1.0 + 2.0 * i / (n - 1);
      p[i] = 0.5 * (1.0 + x[i]);
    }

    double mu1, mu2;
    openmc::angular_moments(x, p, false, mu1, mu2);
    CHECK_THAT(mu1, WithinRel(2.0 / 3.0, 1.0e-12));
    CHECK_THAT(mu2, WithinRel(1.0, 1.0e-12));
  }
}

TEST_CASE("an isotropic distribution has the moments of isotropy")
{
  // p(mu) = 1/2 gives <1-mu> = 1 and <(3/2)(1-mu^2)> = 1. A scheme that
  // grouped soft collisions would relax to isotropy over one transport mean
  // free path, so these are the values the step length is measured against.
  int n = 9;
  std::vector<double> x(n), p(n, 0.5);
  for (int i = 0; i < n; ++i)
    x[i] = -1.0 + 2.0 * i / (n - 1);

  double mu1, mu2;
  openmc::angular_moments(x, p, true, mu1, mu2);
  CHECK_THAT(mu1, WithinRel(1.0, 1.0e-12));
  CHECK_THAT(mu2, WithinRel(1.0, 1.0e-12));
}

// The soft/hard split a mixed condensed-history step is built on. The cutoff
// is defined implicitly, by C1 = sigma_1_soft / sigma_hard, so what has to be
// checked is that the solver lands on the cosine that equation names.
TEST_CASE("the soft/hard split solves for the cutoff C1 asks for")
{
  // p(mu) = (1 + mu)/2 again. Putting the cutoff at mu = 0 leaves
  //   P_hard   = (1 + mu_c)^2/4                 = 1/4
  //   <1-mu>_s = (1/2)(2/3 - mu_c + mu_c^3/3)   = 1/3
  //   <(3/2)(1-mu^2)>_s                         = 11/16
  // so C1 = (1/3)/(1/4) = 4/3 must put it exactly there. n = 6 is included
  // because it has no node at mu = 0: the root then falls strictly inside a
  // segment and the bisection, not the partition, has to find it.
  for (int n : {3, 5, 6, 17}) {
    std::vector<double> x(n), p(n);
    for (int i = 0; i < n; ++i) {
      x[i] = -1.0 + 2.0 * i / (n - 1);
      p[i] = 0.5 * (1.0 + x[i]);
    }

    double mu_cut, p_hard, mu1_soft, mu2_soft;
    openmc::restricted_angular_moments(
      x, p, false, 4.0 / 3.0, mu_cut, p_hard, mu1_soft, mu2_soft);
    CHECK_THAT(mu_cut, WithinAbs(0.0, 1.0e-12));
    CHECK_THAT(p_hard, WithinRel(0.25, 1.0e-12));
    CHECK_THAT(mu1_soft, WithinRel(1.0 / 3.0, 1.0e-12));
    CHECK_THAT(mu2_soft, WithinRel(11.0 / 16.0, 1.0e-12));
  }

  // The same for a histogram density: p = 1/2 cut at mu = 0 leaves P_hard =
  // 1/2 and <1-mu>_soft = 1/4, so C1 = 1/2 names that cutoff.
  int n = 9;
  std::vector<double> x(n), p(n, 0.5);
  for (int i = 0; i < n; ++i)
    x[i] = -1.0 + 2.0 * i / (n - 1);

  double mu_cut, p_hard, mu1_soft, mu2_soft;
  openmc::restricted_angular_moments(
    x, p, true, 0.5, mu_cut, p_hard, mu1_soft, mu2_soft);
  CHECK_THAT(mu_cut, WithinAbs(0.0, 1.0e-12));
  CHECK_THAT(p_hard, WithinRel(0.5, 1.0e-12));
  CHECK_THAT(mu1_soft, WithinRel(0.25, 1.0e-12));
  CHECK_THAT(mu2_soft, WithinRel(0.5, 1.0e-12));
}

TEST_CASE("C1 = 0 is exactly single-event transport")
{
  // Nothing may be grouped: the cutoff sits at the top of the range, the whole
  // cross section is hard and both soft moments vanish. A mixed scheme that
  // did not reduce to this could not be checked against the transport it
  // replaces.
  int n = 11;
  std::vector<double> x(n), p(n);
  for (int i = 0; i < n; ++i) {
    x[i] = -1.0 + 2.0 * i / (n - 1);
    p[i] = 0.5 * (1.0 + x[i]);
  }

  double mu_cut, p_hard, mu1_soft, mu2_soft;
  openmc::restricted_angular_moments(
    x, p, false, 0.0, mu_cut, p_hard, mu1_soft, mu2_soft);
  CHECK(mu_cut == 1.0);
  CHECK(p_hard == 1.0);
  CHECK(mu1_soft == 0.0);
  CHECK(mu2_soft == 0.0);
}

TEST_CASE("the split moves monotonically with C1 and conserves the total")
{
  // A forward-peaked density, closer to what the elastic data look like than
  // anything polynomial: p(mu) ~ 1/(1 + a - mu)^2 with a small, the Wentzel
  // form. Raising C1 must move the cutoff away from forward, take cross
  // section out of the hard part and put first moment into the soft one --
  // and the soft moment can never exceed the total.
  int n = 4001;
  double a = 1.0e-3;
  std::vector<double> x(n), p(n);
  for (int i = 0; i < n; ++i) {
    x[i] = -1.0 + 2.0 * i / (n - 1);
    p[i] = 1.0 / ((1.0 + a - x[i]) * (1.0 + a - x[i]));
  }

  double mu1_total, mu2_total;
  openmc::angular_moments(x, p, false, mu1_total, mu2_total);

  double last_cut = 1.0, last_hard = 1.0, last_m1 = 0.0;
  for (double c1 : {0.01, 0.02, 0.05, 0.1, 0.2}) {
    double mu_cut, p_hard, mu1_soft, mu2_soft;
    openmc::restricted_angular_moments(
      x, p, false, c1, mu_cut, p_hard, mu1_soft, mu2_soft);

    // The defining equation, which is the whole point of the solver
    CHECK_THAT(mu1_soft, WithinRel(c1 * p_hard, 1.0e-10));

    CHECK(mu_cut < last_cut);
    CHECK(p_hard < last_hard);
    CHECK(mu1_soft > last_m1);
    CHECK(mu1_soft < mu1_total);
    CHECK(mu2_soft < mu2_total);
    CHECK(p_hard > 0.0);
    last_cut = mu_cut;
    last_hard = p_hard;
    last_m1 = mu1_soft;
  }
}

// The free Bhabha moments, which carry a positron's transfers above the Moller
// limit into the stopping power and the straggling. The shape itself is
// PENELOPE's and is not restated here; what is checked are the properties any
// correct set of moments has, which a wrong power or a dropped term breaks.
TEST_CASE("the Bhabha moments are additive and bracket their own range")
{
  double E = 1.0e7;
  for (double W_lo : {6.0e6, 8.0e6}) {
    double W_mid = 0.5 * (W_lo + E);
    for (int order : {0, 1, 2}) {
      double whole = openmc::detail::bhabha_moment(E, W_lo, E, order);
      double lower = openmc::detail::bhabha_moment(E, W_lo, W_mid, order);
      double upper = openmc::detail::bhabha_moment(E, W_mid, E, order);
      CHECK_THAT(lower + upper, WithinRel(whole, 1.0e-12));
      CHECK(whole > 0.0);
    }

    // Each moment is the cross section times a mean of W^order over the
    // range, so it lies between the endpoints raised to that power
    double m0 = openmc::detail::bhabha_moment(E, W_lo, E, 0);
    double m1 = openmc::detail::bhabha_moment(E, W_lo, E, 1);
    double m2 = openmc::detail::bhabha_moment(E, W_lo, E, 2);
    CHECK(m1 > W_lo * m0);
    CHECK(m1 < E * m0);
    CHECK(m2 > W_lo * W_lo * m0);
    CHECK(m2 < E * E * m0);

    // Cauchy-Schwarz on the same measure
    CHECK(m1 * m1 <= m0 * m2);

    // An empty range carries nothing
    CHECK(openmc::detail::bhabha_moment(E, E, E, 1) == 0.0);
    CHECK(openmc::detail::bhabha_moment(E, E, W_lo, 1) == 0.0);
  }
}

// Where the soft/hard split of the inelastic channels falls is not a free
// parameter: it follows from the cutoffs. A collision may be grouped when
// nothing it emits would have been transported AND the projectile survives it,
// and those two conditions bring in all three cutoffs.
TEST_CASE("the inelastic thresholds follow the transport cutoffs")
{
  int photon = openmc::ParticleType::photon().transport_index();
  int electron = openmc::ParticleType::electron().transport_index();
  int positron = openmc::ParticleType::positron().transport_index();
  auto saved = openmc::settings::energy_cutoff;
  double saved_loss = openmc::settings::energy_loss_cutoff;
  openmc::settings::energy_loss_cutoff = 0.05;
  double E = 2.2e7;

  // Three things bound what a collision may transfer and still be grouped,
  // and each is checked where it is the one that binds.

  // The step's own energy budget, which is what binds in a photoneutron run:
  // the step may lose 0.05 of 22 MeV and one collision may carry a tenth of
  // that. Left out, the 8 MeV cutoffs below would let a single grouped
  // collision carry seven times the energy the step was allowed to lose.
  openmc::settings::energy_cutoff[photon] = 8.0e6;
  openmc::settings::energy_cutoff[electron] = 8.0e6;
  openmc::settings::energy_cutoff[positron] = 7.24e6;
  CHECK_THAT(openmc::soft_collision_cutoff(openmc::ParticleType::electron(), E),
    WithinRel(0.1 * 0.05 * E, 1.0e-12));
  CHECK_THAT(openmc::soft_radiative_cutoff(openmc::ParticleType::electron(), E),
    WithinRel(0.1 * 0.05 * E, 1.0e-12));

  // What the collision emits, which binds once the cutoffs are low. An
  // ionization collision emits a knock-on and, through the vacancy it leaves,
  // fluorescence and Auger products, every one below the transfer itself, so
  // the lower of the electron and photon cutoffs bounds it. Bremsstrahlung
  // emits only a photon, so the electron cutoff does not bound it and the two
  // thresholds part company.
  openmc::settings::energy_cutoff[photon] = 1.0e5;
  openmc::settings::energy_cutoff[electron] = 1.0e4;
  CHECK(openmc::soft_collision_cutoff(openmc::ParticleType::electron(), E) ==
        1.0e4);
  CHECK(openmc::soft_radiative_cutoff(openmc::ParticleType::electron(), E) ==
        1.0e5);

  // The projectile itself, which binds at the cutoff. There the two charges
  // part company: a positron may still be grouped a little where an electron
  // may not, because its own cutoff is the lower one -- that being the whole
  // reason a photoneutron run sets it lower.
  openmc::settings::energy_cutoff[photon] = 8.0e6;
  openmc::settings::energy_cutoff[electron] = 8.0e6;
  CHECK(openmc::soft_projectile_headroom(
          openmc::ParticleType::electron(), 8.0e6) == 0.0);
  CHECK(openmc::soft_projectile_headroom(
          openmc::ParticleType::positron(), 8.0e6) == 8.0e6 - 7.24e6);
  CHECK(openmc::soft_collision_cutoff(
          openmc::ParticleType::electron(), 8.0e6) == 0.0);
  CHECK(openmc::soft_collision_cutoff(openmc::ParticleType::positron(), 8.0e6) >
        0.0);
  CHECK(openmc::soft_radiative_cutoff(
          openmc::ParticleType::electron(), 8.0e6) == 0.0);

  // OpenMC's default transports every electron to rest, so every knock-on is
  // followed and no collision may be grouped. Radiative losses under the
  // photon cutoff still may, up to the step's share of them.
  openmc::settings::energy_cutoff[photon] = 1000.0;
  openmc::settings::energy_cutoff[electron] = 0.0;
  openmc::settings::energy_cutoff[positron] = 0.0;
  CHECK(
    openmc::soft_projectile_headroom(openmc::ParticleType::electron(), E) == E);
  CHECK(
    openmc::soft_collision_cutoff(openmc::ParticleType::electron(), E) == 0.0);
  CHECK(openmc::soft_radiative_cutoff(openmc::ParticleType::electron(), E) ==
        1000.0);

  openmc::settings::energy_cutoff = saved;
  openmc::settings::energy_loss_cutoff = saved_loss;
}

// The identity the grouped collision channel is built on
// ------------------------------------------------------
// A mixed scheme needs the collision loss split in two at a cutoff: the part
// below it, carried continuously, and the part above it, sampled one
// collision at a time. Berger and Seltzer give the first in closed form and
// the free binary cross section gives the second, and the two are not
// independent -- the transfers the restricted stopping power leaves out are
// exactly the ones the binary cross section describes. So
//
//     F(tau, Delta_max) - F(tau, Delta) = integral of eps dsigma/deps
//
// over [Delta, Delta_max]. If it holds, a scheme built from those two pieces
// reproduces the ICRU 37 total exactly at every energy and every cutoff, with
// nothing left to calibrate. If it ever stops holding, the two halves no
// longer add up and the grouped channel is silently wrong by the difference.
TEST_CASE("restricted stopping power and the free cross section are "
          "complementary")
{
  constexpr double mc2 = openmc::MASS_ELECTRON_EV;

  for (double tau : {0.1, 1.0, 10.0, 100.0, 200.0}) {
    double E = tau * mc2;
    for (double frac : {1.0e-5, 1.0e-3, 1.0e-2, 0.1}) {
      double d = frac * tau;

      // Electron: the free cross section is Moller's and stops at E/2
      double lost = openmc::berger_seltzer_spin_term(tau, 0.5 * tau, false) -
                    openmc::berger_seltzer_spin_term(tau, d, false);
      double hard = openmc::detail::moller_moment(E, frac * E, 0.5 * E, 1);
      CHECK_THAT(hard, WithinRel(lost, 1.0e-12));

      // Positron: Bhabha's, and it runs to the whole kinetic energy
      lost = openmc::berger_seltzer_spin_term(tau, tau, true) -
             openmc::berger_seltzer_spin_term(tau, d, true);
      hard = openmc::detail::bhabha_moment(E, frac * E, E, 1);
      CHECK_THAT(hard, WithinRel(lost, 1.0e-12));
    }
  }
}

// The restricted form at its kinematic limit must be the unrestricted one,
// which the material stopping power writes in the more familiar closed form.
// Two derivations of the same quantity, which is what makes the check worth
// making.
//
// Compared absolutely rather than relatively. The two forms are algebraically
// identical and differ only by rounding, but the restricted one reaches the
// answer by subtracting ln(tau^2/4) from terms of its own size, and at
// tau = 1000 that leaves about eleven digits. What the term feeds is a
// bracket of order twenty-five, so an absolute agreement of 1e-9 is five
// orders tighter than anything the stopping power can notice, where a
// relative one would be testing floating-point arithmetic.
TEST_CASE("the unrestricted limit recovers the ICRU 37 spin term")
{
  for (double tau : {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}) {
    double gamma = tau + 1.0;
    double beta_sq = tau * (tau + 2.0) / (gamma * gamma);

    double closed_form = (1.0 - beta_sq) * (1.0 + tau * tau / 8.0 -
                                             (2.0 * tau + 1.0) * std::log(2.0));
    CHECK_THAT(openmc::berger_seltzer_spin_term(tau, 0.5 * tau, false),
      WithinAbs(closed_form, 1.0e-9));

    double t = tau + 2.0;
    closed_form =
      std::log(4.0) -
      (beta_sq / 12.0) * (23.0 + 14.0 / t + 10.0 / (t * t) + 4.0 / (t * t * t));
    CHECK_THAT(openmc::berger_seltzer_spin_term(tau, tau, true),
      WithinAbs(closed_form, 1.0e-9));
  }

  // Asking for more than the kinematics allow is the unrestricted case
  CHECK(openmc::berger_seltzer_spin_term(5.0, 100.0, false) ==
        openmc::berger_seltzer_spin_term(5.0, 2.5, false));
  CHECK(openmc::berger_seltzer_spin_term(5.0, 100.0, true) ==
        openmc::berger_seltzer_spin_term(5.0, 5.0, true));
}

// The sum rule the oscillator model stands on
// -------------------------------------------
// The Sternheimer-Liljequist GOS is not fitted to stopping-power data. Its
// oscillator strengths sum to Z and its resonance energies satisfy
// sum_k f_k ln W_k = ln I, and those are the same two constraints that
// determine the Bethe stopping power -- so the model has to reproduce ICRU 37
// rather than merely happening to.
//
// The purest statement of that needs no data at all. One oscillator carrying
// every electron, placed at W = I, must give back the Bethe formula for a
// medium with that mean excitation energy. If it ever stops doing so, the
// model has come loose from the thing that makes it defensible.
TEST_CASE("one oscillator at the mean excitation energy gives Bethe")
{
  for (double I : {78.0, 322.0, 823.0}) {
    for (double E : {1.0e5, 1.0e6, 1.0e7, 1.0e8}) {
      // Nothing grouped, so the whole cross section is in the hard channel
      auto m = openmc::gos_oscillator(E, 0.0, I, 0.0, 0.0);
      CHECK(m.s_soft == 0.0);

      double tau = E / openmc::MASS_ELECTRON_EV;
      double gamma = tau + 1.0;
      double beta_sq = 1.0 - 1.0 / (gamma * gamma);
      double bethe = openmc::berger_seltzer_spin_term(tau, 0.5 * tau, false) +
                     std::log(tau * tau * (tau + 2.0) / 2.0) -
                     2.0 * (std::log(I) - std::log(openmc::MASS_ELECTRON_EV));

      // Both in units of the leading constant, which cancels
      constexpr double bohr = openmc::PLANCK_C * openmc::FINE_STRUCTURE /
                              (2.0 * openmc::PI * openmc::MASS_ELECTRON_EV) *
                              1.0e-8;
      constexpr double r_e =
        bohr / (openmc::FINE_STRUCTURE * openmc::FINE_STRUCTURE);
      double k = 2.0 * openmc::PI * 1.0e24 * r_e * r_e *
                 openmc::MASS_ELECTRON_EV / beta_sq;

      CHECK_THAT(m.s_hard, WithinRel(k * bethe, 1.0e-3));
    }
  }
}

// Splitting the same oscillator at a cutoff must not change what it carries:
// the soft and hard parts are one cross section either side of a line, which
// is the property a mixed scheme needs and the reason the model can serve
// single-event transport and condensed history without a seam between them.
TEST_CASE("the oscillator split conserves its moments")
{
  for (bool positron : {false, true}) {
    for (double u_b : {0.0, 60.0, 8979.0}) {
      double w_r = (u_b > 0.0) ? 1.3 * u_b + 40.0 : 90.0;
      for (double E : {1.0e5, 1.0e6, 1.0e8}) {
        auto whole = openmc::gos_oscillator(E, u_b, w_r, 0.3, 0.0, positron);
        for (double w_cc : {1.0e2, 1.0e3, 1.0e4}) {
          auto split = openmc::gos_oscillator(E, u_b, w_r, 0.3, w_cc, positron);
          CHECK_THAT(split.xs_soft + split.xs_hard,
            WithinRel(whole.xs_soft + whole.xs_hard, 1.0e-12));
          CHECK_THAT(split.s_soft + split.s_hard,
            WithinRel(whole.s_soft + whole.s_hard, 1.0e-12));
          CHECK_THAT(split.w2_soft + split.w2_hard,
            WithinRel(whole.w2_soft + whole.w2_hard, 1.0e-12));
        }
      }
    }
  }
}
