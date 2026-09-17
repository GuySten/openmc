#include "openmc/condensed_history.h"

#include <algorithm> // for max, min
#include <cmath>     // for abs, exp, sqrt

#include "openmc/constants.h"
#include "openmc/particle_data.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"

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

MixedStep sample_mixed_step(double xs_hard, double xs_soft, double xs1_soft,
  double stopping_power, double max_loss, double max_deflection,
  double max_distance, uint64_t* seed)
{
  MixedStep step;

  // Whether to group is settled before anything is sampled, from the step this
  // material and this geometry allow rather than from a sampled one. Sampling
  // first and declining the short ones would leave the grouped steps drawn
  // from an exponential with its lower end cut off, which is a longer mean
  // free path to the hard collision than the cross section says.
  double mfp = (xs_hard > 0.0) ? 1.0 / xs_hard : INFTY;
  double s_energy = (stopping_power > 0.0 && max_loss > 0.0)
                      ? max_loss / stopping_power
                      : INFTY;
  double s_angle = (xs1_soft > 0.0 && max_deflection > 0.0)
                     ? max_deflection / xs1_soft
                     : INFTY;
  double reach =
    std::min(std::min(mfp, s_energy), std::min(s_angle, max_distance));
  if (!(reach > 0.0) || xs_soft * reach < MIN_GROUPED_COLLISIONS)
    return step;
  step.grouped = true;

  // Path to the next hard interaction, which is the step the scheme wants.
  // Everything below only shortens it, and shortening costs nothing: the
  // exponential has no memory, so a step cut short and resumed is the same
  // flight as an uncut one.
  double s = (xs_hard > 0.0) ? -std::log(prn(seed)) / xs_hard : INFTY;
  step.ends_in_collision = true;

  // The energy ceiling: see max_loss. It stops the grouped loss from being a
  // large part of what the projectile has, and stops the step from carrying it
  // past its own cutoff.
  if (s_energy < s) {
    s = s_energy;
    step.ends_in_collision = false;
  }

  // The angular ceiling: see max_deflection. One artificial deflection stands
  // in for the grouped ones and is applied at a single point in the step, so
  // how far the step may run is set by how much turning it may cover.
  if (s_angle < s) {
    s = s_angle;
    step.ends_in_collision = false;
  }

  // Geometry has the last word on length
  if (max_distance < s) {
    s = max_distance;
    step.ends_in_collision = false;
  }

  step.length = s;
  step.hinge = s * prn(seed);
  return step;
}

namespace {

//! Transport cutoff of the projectile itself, whatever it is
double own_cutoff(ParticleType type)
{
  int index = type.transport_index();
  return (index == C_NONE) ? 0.0 : settings::energy_cutoff[index];
}

} // namespace

double soft_projectile_headroom(ParticleType type, double E)
{
  return std::max(0.0, E - own_cutoff(type));
}

namespace {

//! Largest share of a step's energy budget one grouped collision may carry
//!
//! The step describes that energy by two moments, and the transfers are
//! distributed as 1/W^2, so the variance sits in the few largest of them. A
//! tenth leaves about ten of them to share it, which is the fewest that makes
//! a mean and a variance mean anything.
constexpr double MAX_SOFT_LOSS_SHARE = 0.1;

} // namespace

double soft_loss_budget(ParticleType type, double E)
{
  return std::min(
    settings::energy_loss_cutoff * E, soft_projectile_headroom(type, E));
}

double soft_collision_cutoff(ParticleType type, double E)
{
  double photon =
    settings::energy_cutoff[ParticleType::photon().transport_index()];
  double electron =
    settings::energy_cutoff[ParticleType::electron().transport_index()];
  return std::max(0.0, std::min(std::min(photon, electron),
                         std::min(soft_projectile_headroom(type, E),
                           MAX_SOFT_LOSS_SHARE * soft_loss_budget(type, E))));
}

double soft_radiative_cutoff(ParticleType type, double E)
{
  double photon =
    settings::energy_cutoff[ParticleType::photon().transport_index()];
  return std::max(
    0.0, std::min(photon, std::min(soft_projectile_headroom(type, E),
                            MAX_SOFT_LOSS_SHARE * soft_loss_budget(type, E))));
}

} // namespace openmc
