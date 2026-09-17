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

} // namespace openmc

#endif // OPENMC_CONDENSED_HISTORY_H
