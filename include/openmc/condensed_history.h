//! \file condensed_history.h
//! Artificial distributions a mixed condensed-history step is carried by

#ifndef OPENMC_CONDENSED_HISTORY_H
#define OPENMC_CONDENSED_HISTORY_H

#include <cstdint>

#include "openmc/particle_data.h"
#include "openmc/vector.h"

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

//! Coarsest a step is ever allowed to be, for either of the two bounds
//!
//! Past this the mixed scheme has left the ground it stands on. A step turning
//! the particle through \f$\langle 1-\mu \rangle = 0.2\f$ has turned it
//! through some 37 degrees, and standing all of that on one artificial
//! deflection at one point of the step is no longer a description of a path;
//! a step taking a fifth of the kinetic energy has moved far enough that the
//! cross sections it was begun with belong to a different particle. PENELOPE
//! caps both parameters here and PenRed enforces it in code, and there is no
//! reason to allow what neither of them does.
constexpr double MAX_STEP_COARSENESS = 0.2;

//! Largest share of a step's energy budget one grouped collision may carry
//!
//! The step describes that energy by two moments, and the transfers are
//! distributed as 1/W^2, so the variance sits in the few largest of them. A
//! tenth leaves about ten of them to share it, which is the fewest that makes
//! a mean and a variance mean anything.
//!
//! It also bounds how far the sampled loss can overshoot its mean, which is
//! what MAX_SOFT_LOSS_OVERSHOOT below is derived from: the two move together
//! and neither can be changed alone.
constexpr double MAX_SOFT_LOSS_SHARE = 0.1;

//! How far past its budget the energy a step actually loses may reach
//!
//! The step's length is chosen so the grouped loss averages no more than the
//! budget, but the sampled loss fluctuates above that mean and the bound on
//! the hard cross section has to cover where it lands, not where it is aimed.
//! The overshoot is bounded because the variance is: over a step of length s
//! the loss has mean sS and variance s(Omega) for the restricted stopping
//! power S and straggling Omega, and
//!
//! \f[ \frac{\Omega}{S} = \frac{\int W^2 d\sigma}{\int W d\sigma}
//!     \le W_{cc}, \f]
//!
//! since every transfer in those integrals is under the soft cutoff -- which
//! is why grouped excitation, whose transfer is a fixed loss rather than a
//! draw from that spectrum, is held under the same ceiling. That
//! cutoff is in turn held under MAX_SOFT_LOSS_SHARE of the budget, so the
//! variance is under a tenth of the mean squared, which puts the two branches
//! of sample_soft_energy_loss() at 1.55 and 1.65 budgets respectively. Two
//! covers both, and the only cost of the margin is a slightly looser bound.
constexpr double MAX_SOFT_LOSS_OVERSHOOT = 2.0;

//! Sample the next step
//!
//! \param[in] xs_hard Macroscopic hard cross section in [1/cm]
//! \param[in] xs_soft Macroscopic rate of the collisions being grouped, in
//!   [1/cm], which decides only whether grouping is worth it
//! \param[in] xs1_soft First transport cross section of the grouped
//!   deflections, in [1/cm]
//! \param[in] stopping_power Restricted stopping power in [eV/cm]
//! \param[in] max_loss Most the projectile may lose to the grouped collisions
//!   over one step, in [eV]. Two things bound it: the energy_loss cutoff as a
//!   fraction of the kinetic energy, which keeps the restricted stopping power
//!   evaluated near the energy it belongs to; and the energy left above the
//!   projectile's own transport cutoff, since a step that carried it past that
//!   would take it beyond the point where it should have stopped -- and, for a
//!   positron, past the energy at which it should have annihilated.
//! \param[in] max_deflection Most the grouped collisions may turn the
//!   projectile over one step, as \f$\langle 1-\mu \rangle\f$. This is the
//!   deflection cutoff, which PENELOPE calls \f$C_1\f$ and gets by
//!   construction: there the step is
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

//! Bound a tabulated hard cross section over the energies one step can reach
//!
//! The flight to the next hard interaction is drawn from this bound rather
//! than from the cross section at the energy the step starts with, because the
//! projectile slows down as it goes. The bound has only to hold: slack costs a
//! few declined interactions, while a shortfall cannot be declined at all and
//! silently undercounts hard interactions, so the two sides of being wrong are
//! not comparable and the scan errs high.
//!
//! Pulled out of the element that owns the table so that it can be tested on
//! one, this being the kind of loop whose off-by-one is invisible in the
//! answer: a bound that is too low by one grid interval still looks like a
//! bound, and only shows up as a slow leak of hard collisions.
//!
//! \param[in] energy Ascending energy grid
//! \param[in] hard_xs Hard cross section tabulated on that grid
//! \param[in] lowest Lowest energy a step beginning at each grid point can
//!   reach, which the caller derives from what the step may lose
//! \param[in] margin Factor the bound is raised by once found
//! \return The bound, one value per grid point
vector<double> step_majorant(const vector<double>& energy,
  const vector<double>& hard_xs, const vector<double>& lowest,
  double margin = 1.001);

//==============================================================================
// What may be grouped
//
// These say which interactions a step is allowed to swallow, and they are
// written for a charged particle rather than for an electron. Nothing in them
// asks what the projectile is made of: they ask what it would emit, and how
// far it is above the energy at which it stops being followed. A heavier
// particle reaches the same rules through the same three calls.
//==============================================================================

//! Energy a grouped event may take from the projectile itself
//!
//! A soft collision must not be able to carry the projectile across the energy
//! at which it stops being transported. Below its own cutoff the projectile
//! would have been killed where it was -- and for a positron, killed means
//! annihilated at rest, which makes two 511 keV photons where an annihilation
//! in flight would have made one of up to \f$T + 1.5 m_e c^2\f$. A grouped
//! event that stepped over that energy would swap one outcome for the other,
//! so the transfer is bounded by how far the projectile is above it.
//!
//! \param[in] type The charged particle being transported
//! \param[in] E Kinetic energy in [eV]
//! \return Headroom in [eV]
double soft_projectile_headroom(ParticleType type, double E);

//! Energy a step is allowed to lose to the grouped collisions
//!
//! A fraction of the kinetic energy, or what is left above the projectile's
//! own transport cutoff, whichever is the smaller.
//!
//! \param[in] type The charged particle being transported
//! \param[in] E Kinetic energy in [eV]
//! \return Budget in [eV]
double soft_loss_budget(ParticleType type, double E);

//! Largest energy transfer a collision may make and still be grouped
//!
//! Two things have to hold. Nothing the collision produces may be lost: a
//! collision transferring \f$W\f$ puts on the stack a knock-on electron of
//! \f$W - B\f$ and, from the vacancy it leaves, fluorescence photons and Auger
//! electrons of at most \f$B\f$, every one of them below \f$W\f$ itself, so a
//! transfer under both the electron and the photon cutoff produces nothing
//! that would have been transported. And the projectile has to survive it,
//! which is soft_projectile_headroom().
//!
//! All three cutoffs therefore bear on the threshold: the photon and electron
//! ones through what the collision emits, the positron one through what the
//! projectile becomes.
//!
//! One more thing bounds it, and it has nothing to do with what is lost. A
//! step describes the energy its grouped collisions take by a mean and a
//! variance, which is a fair account only if many of them contribute. The
//! spectrum of transfers falls as \f$1/W^2\f$, so the variance is carried by
//! the largest of them, and a transfer comparable to the step's whole energy
//! budget would leave that budget in the hands of one or two collisions. So
//! the threshold is also capped at a fraction of soft_loss_budget(). In
//! a photoneutron run this is what binds: an 8 MeV electron cutoff would
//! otherwise let a single grouped collision carry seven times the energy the
//! step was allowed to lose, which is not a description of anything.
//!
//! PENELOPE leaves W_cc to the user and PenRed sets it to a hundredth of the
//! absorption energy, capped at 5 keV, which is the same guard reached from
//! the other end.
//!
//! \param[in] type The charged particle being transported
//! \param[in] E Kinetic energy in [eV]
//! \return Cutoff in [eV]; zero means nothing may be grouped
double soft_collision_cutoff(ParticleType type, double E);

//! Largest bremsstrahlung photon energy that may be grouped
//!
//! A bremsstrahlung collision produces one photon and leaves the projectile in
//! flight, so of what it emits only the photon cutoff bears on it. The
//! projectile bound matters more here than anywhere else: a single photon may
//! carry off nearly the whole kinetic energy, and without that bound a grouped
//! emission could take an electron from just above its cutoff to nearly at
//! rest and then smear the loss along the step.
//!
//! \param[in] type The charged particle being transported
//! \param[in] E Kinetic energy in [eV]
//! \return Cutoff in [eV]
double soft_radiative_cutoff(ParticleType type, double E);

} // namespace openmc

#endif // OPENMC_CONDENSED_HISTORY_H
