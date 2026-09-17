//! \file condensed_history.h
//! Artificial distributions a mixed condensed-history step is carried by

#ifndef OPENMC_CONDENSED_HISTORY_H
#define OPENMC_CONDENSED_HISTORY_H

#include <cstdint>

namespace openmc {

//==============================================================================
//! Artificial angular distribution of the soft deflection over one step
//!
//! Two uniform pieces meeting at \f$\mu_0\f$, the lower one carrying
//! probability \f$a\f$. The shape means nothing: what a class II scheme needs
//! from the grouped collisions is their first two Legendre moments, and the
//! simplest distribution carrying both exactly is the one that should be used,
//! since anything more elaborate would be asserting detail the moments do not
//! contain. This is PENELOPE's choice.
//==============================================================================

struct SoftScattering {
  double mu_0 {1.0}; //!< cosine where the two pieces meet
  double a {0.0};    //!< probability of the piece below mu_0
};

//! Build the artificial distribution from the two moments it must carry
//!
//! Solving \f$\langle \mu \rangle\f$ and \f$\langle \mu^2 \rangle\f$ for the
//! two parameters gives
//! \f[ \mu_0 = \frac{3\langle\mu^2\rangle - 1}{2\langle\mu\rangle}, \qquad
//!     a = \frac{1 + \mu_0 - 2\langle\mu\rangle}{2}. \f]
//!
//! confined to the range over which \f$a\f$ stays a probability. Both moments
//! come back exactly for every pair a real angular distribution can produce,
//! which is \f$\sigma_1 \le \sigma_2 \le 3\sigma_1\f$. Outside it the first
//! is still exact and the second is as close as the form allows, which is the
//! right way round: the first is what sets the transport mean free path.
//!
//! \param[in] mu1 \f$\langle \mu \rangle\f$ over the step
//! \param[in] mu2 \f$\langle \mu^2 \rangle\f$ over the step
SoftScattering soft_scattering(double mu1, double mu2);

//! Sample the deflection the grouped elastic collisions accumulate over a step
//!
//! Over a path \f$s\f$ the moments decay as \f$\langle \mu \rangle =
//! e^{-s/\lambda_1}\f$ and \f$\langle P_2(\mu) \rangle = e^{-s/\lambda_2}\f$,
//! with \f$1/\lambda_\ell = n\sigma_\ell\f$ built from the soft part of the
//! elastic distribution alone.
//!
//! \param[in] s_lambda1 \f$s/\lambda_1\f$, the first transport optical depth
//! \param[in] s_lambda2 \f$s/\lambda_2\f$, the second
//! \param[inout] seed Pseudorandom number seed pointer
//! \return Cosine of the accumulated deflection
double sample_soft_deflection(
  double s_lambda1, double s_lambda2, uint64_t* seed);

//! Sample the energy the grouped collisions take over a step
//!
//! Matching the mean and the variance is the whole content of a restricted
//! stopping power with straggling, so again the simplest distribution carrying
//! both is used. Which one that is depends on how wide the loss is relative to
//! its mean: a uniform distribution can only reach a variance of
//! \f$\langle\omega\rangle^2/3\f$ without going negative, and past that the
//! distribution becomes a uniform piece with an atom at zero -- the step
//! either loses nothing or loses a good deal, which is what a broad straggling
//! distribution physically is.
//!
//! \param[in] mean Mean soft loss over the step in [eV]
//! \param[in] variance Variance of that loss in [eV^2]
//! \param[inout] seed Pseudorandom number seed pointer
//! \return Energy lost in [eV], never negative
double sample_soft_energy_loss(double mean, double variance, uint64_t* seed);

//==============================================================================
//! One mixed condensed-history step
//!
//! The step runs from one hard interaction to the next, with the grouped soft
//! effects applied at a single point inside it. That point is where the
//! spatial accuracy comes from: putting the whole deflection at the end would
//! leave the particle travelling in a straight line for the length of the
//! step and lose the lateral spread, while putting it at the start would
//! overstate it. Uniformly along the step is PENELOPE's random hinge.
//==============================================================================

struct MixedStep {
  double length {0.0}; //!< path the step covers in [cm]
  double hinge {0.0};  //!< distance into it at which the soft effects act
  //! Whether anything is being grouped. False hands the step back to the
  //! single-event transport, and leaves every other field unset.
  bool grouped {false};
  //! Whether a hard interaction waits at the end. False when the energy
  //! ceiling or the geometry cut the step short, and the next one resumes.
  bool ends_in_collision {false};
};

//! Fewest grouped collisions a step must contain to be worth grouping
//!
//! Below this the trade is bad in both directions. A step describes the
//! collisions it swallows by two moments, which is a fair account of a sum of
//! N of them to about \f$1/\sqrt{N}\f$ and no better, so a hundred of them
//! buys a ten per cent description and ten of them a thirty per cent one; and
//! a step that removes ten collisions was not going to be much faster than
//! simulating them anyway.
//!
//! The reasoning fixes the shape of the criterion and measurement fixes the
//! number. At thirty, a 100 keV depth dose in carbon comes back identical to
//! the single-event one -- every step declines to group -- where at ten the
//! two differ by 3.6 standard errors for a speedup of only 1.9. A 1 MeV one
//! keeps a speedup of 4.0 either way and the same 2.1 standard errors, so the
//! low-energy end is what the number is bought from, and it costs the high
//! end a factor of two.
//!
//! This is what decides, step by step and with no input from the user, where a
//! run stops being condensed history and becomes the single-event transport it
//! is built to agree with: in a thin foil, near an interface, wherever the
//! geometry cuts the step short -- and at low energy, where collisions are
//! violent enough that few of them fit under the angular ceiling. A 10 keV
//! electron in carbon fits 4 and is transported one collision at a time; a
//! 1 MeV one fits 150, and a 22 MeV one 16000.
constexpr double MIN_GROUPED_COLLISIONS = 30.0;

//! Sample the next step
//!
//! \param[in] xs_hard Macroscopic hard cross section in [1/cm]
//! \param[in] xs_soft Macroscopic rate of the collisions being grouped, in
//!   [1/cm], which decides only whether grouping is worth it
//! \param[in] xs1_soft First transport cross section of the grouped
//!   deflections, in [1/cm]
//! \param[in] stopping_power Restricted stopping power in [eV/cm]
//! \param[in] max_loss Most the projectile may lose to the grouped collisions
//!   over one step, in [eV]. Two things bound it: a fraction \f$C_2\f$ of the
//!   kinetic energy, which keeps the restricted stopping power evaluated near
//!   the energy it belongs to; and the energy left above the projectile's own
//!   transport cutoff, since a step that carried it past that would take it
//!   beyond the point where it should have stopped -- and, for a positron,
//!   past the energy at which it should have annihilated.
//! \param[in] max_deflection Most the grouped collisions may turn the
//!   projectile over one step, as \f$\langle 1-\mu \rangle\f$. This is what
//!   \f$C_1\f$ names and, in PENELOPE, gets by construction: there the step is
//!   one hard mean free path, so the soft deflection over it is
//!   \f$\sigma_{1,soft}/\sigma_{hard}\f$ by definition. Here the step is a
//!   sampled flight, which runs past a mean free path as often as not, and
//!   the energy ceiling and the geometry cut it short besides -- so the bound
//!   the parameter names has to be applied rather than assumed. It needs no
//!   tabulating to follow the material: \f$\sigma_{1,soft}\f$ already does.
//! \param[in] max_distance Distance beyond which the step cannot usefully
//!   run, normally the distance to the nearest boundary, in [cm]
//! \param[inout] seed Pseudorandom number seed pointer
MixedStep sample_mixed_step(double xs_hard, double xs_soft, double xs1_soft,
  double stopping_power, double max_loss, double max_deflection,
  double max_distance, uint64_t* seed);

} // namespace openmc

#endif // OPENMC_CONDENSED_HISTORY_H
