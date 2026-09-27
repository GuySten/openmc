//! \file distribution_angle.h
//! Angle distribution dependent on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ANGLE_H
#define OPENMC_DISTRIBUTION_ANGLE_H

#include "hdf5.h"

#include "openmc/distribution.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Angle distribution that depends on incident particle energy
//==============================================================================

class AngleDistribution {
public:
  AngleDistribution() = default;

  //! \param[in] group HDF5 group to read the distribution from
  //! \param[in] energy_interp ENDF interpolation rule to apply between the
  //!   tabulated incident energies -- the TAB2 rule, distinct from the rule
  //!   used within each distribution. lin_lin is the default and reproduces
  //!   the historical behaviour, so existing data and callers are unaffected.
  explicit AngleDistribution(
    hid_t group, Interpolation energy_interp = Interpolation::lin_lin);

  //! Sample an angle given an incident particle energy
  //! \param[in] E Particle energy in [eV]
  //! \param[inout] seed pseudorandom number seed pointer
  //! \return Cosine of the angle in the range [-1,1]
  double sample(double E, uint64_t* seed) const;

  //! Sample an angle from the part of the distribution below a quantile
  //!
  //! What a mixed condensed-history scheme needs when a hard elastic
  //! collision ends a step: the deflections it did not group, which are the
  //! large ones. Those sit at the bottom of the cumulative distribution, mu
  //! being measured from backward, so restricting the quantile to
  //! \f$[0, \xi_{max}]\f$ selects exactly them -- with no rejection, which
  //! matters because the hard part can be one ten-thousandth of the whole.
  //!
  //! \param[in] E Particle energy in [eV]
  //! \param[in] xi_max Largest quantile the sample may come from
  //! \param[inout] seed pseudorandom number seed pointer
  //! \return Cosine of the angle in the range [-1,1]
  double sample_restricted(double E, double xi_max, uint64_t* seed) const;

  //! Evaluate the angular PDF at a given energy and cosine
  //! \param[in] E Particle energy in [eV]
  //! \param[in] mu Cosine of the scattering angle
  //! \return Probability density for the scattering cosine
  double evaluate(double E, double mu) const;

  //! Determine whether angle distribution is empty
  //! \return Whether distribution is empty
  bool empty() const { return energy_.empty(); }

  //! Split each tabulated distribution into a soft and a hard part
  //!
  //! A mixed (class II) condensed-history scheme treats deflections larger
  //! than a cutoff \f$\mu_c\f$ as discrete collisions and groups everything
  //! below it into a single artificial deflection per step. Where the cutoff
  //! sits is set the way PENELOPE sets it, by the dimensionless parameter
  //!
  //! \f[ C_1 = \frac{\sigma_{1,\text{soft}}}{\sigma_{\text{hard}}} \f]
  //!
  //! which is the average angular deflection, in the \f$\langle 1-\mu
  //! \rangle\f$ measure, that the grouped collisions accumulate between two
  //! hard ones. Both cross sections carry the same elastic total, so it
  //! cancels and the cutoff depends only on the shape of the distribution.
  //!
  //! \f$C_1 = 0\f$ puts the cutoff at \f$\mu = 1\f$: every collision is
  //! hard, nothing is grouped, and the scheme is exactly the single-event
  //! transport it has to reduce to.
  //!
  //! \param[in] c1 Soft angular deflection per hard collision
  //! \param[out] energy Incident energies the split is tabulated at, in [eV]
  //! \param[out] mu_cut Cosine below which a deflection is hard
  //! \param[out] p_hard Fraction of the elastic cross section that is hard
  //! \param[out] mu1_soft \f$\langle 1 - \mu \rangle\f$ of the soft part,
  //!   per elastic collision, so that \f$\sigma_{1,\text{soft}} =
  //!   \sigma_{el}\f$ times this
  //! \param[out] mu2_soft \f$\langle \frac{3}{2}(1 - \mu^2) \rangle\f$ of
  //!   the soft part, per elastic collision
  void restricted_moments(double c1, vector<double>& energy,
    vector<double>& mu_cut, vector<double>& p_hard, vector<double>& mu1_soft,
    vector<double>& mu2_soft) const;

private:
  //! Shared implementation of sample() and sample_restricted()
  double sample_impl(double E, double xi_max, uint64_t* seed) const;

  vector<double> energy_;
  vector<unique_ptr<Tabular>> distribution_;
  Interpolation energy_interp_ = Interpolation::lin_lin;
};

//==============================================================================
// Non-member functions
//==============================================================================

//! Moments of one tabulated angular density
//!
//! The integrals are exact for each tabulated segment rather than
//! trapezoidal: the integrand is the product of the interpolated density with
//! a polynomial in \f$\mu\f$, and the elastic distributions are so sharply
//! forward-peaked that a quadrature error would show up directly in the step
//! length.
//!
//! \param[in] x Tabulated cosines, ascending
//! \param[in] p Density at each cosine
//! \param[in] histogram Whether the density is a histogram rather than
//!   linearly interpolated
//! \param[out] mu1 \f$\langle 1 - \mu \rangle\f$
//! \param[out] mu2 \f$\langle \frac{3}{2}(1 - \mu^2) \rangle\f$
void angular_moments(const vector<double>& x, const vector<double>& p,
  bool histogram, double& mu1, double& mu2);

//! Soft/hard split of one tabulated angular density
//!
//! Solves \f$\langle 1-\mu \rangle_{\text{soft}}(\mu_c) = C_1 \,
//! P_{\text{hard}}(\mu_c)\f$ for the cutoff. The left side falls to zero and
//! the right side rises to one as \f$\mu_c \to 1\f$, so the ratio decreases
//! monotonically and the root is unique. See
//! AngleDistribution::restricted_moments for the meaning of the arguments.
void restricted_angular_moments(const vector<double>& x,
  const vector<double>& p, bool histogram, double c1, double& mu_cut,
  double& p_hard, double& mu1_soft, double& mu2_soft);

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
