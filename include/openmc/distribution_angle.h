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

  //! Evaluate the angular PDF at a given energy and cosine
  //! \param[in] E Particle energy in [eV]
  //! \param[in] mu Cosine of the scattering angle
  //! \return Probability density for the scattering cosine
  double evaluate(double E, double mu) const;

  //! Mean deflection 1-<mu> that sample() actually produces at this energy
  //
  // Interpolated between the tabulated distributions the same way sample()
  // combines them, so it describes the sampler rather than the underlying
  // data. Lets a caller that knows the correct first moment from elsewhere
  // correct for however coarsely the tables are spaced.
  //
  //! \param[in] E Particle energy in [eV]
  //! \return Mean of 1-mu, or 0 if the distribution is empty
  double mean_deflection(double E) const;

  //! Mean deflection 1-<mu> that sample() actually produces at this energy.
  //
  //! Not the same as mean_deflection(). Under log-log energy interpolation
  //! sample() interpolates the two tables' QUANTILES geometrically, while
  //! mean_deflection() interpolates their MEANS geometrically, and the mean of
  //! a geometric interpolation is not the geometric interpolation of the means
  //! -- Jensen. The gap reaches 10% deep inside a sparse interval. Anything
  //! rescaling the sampled deflection has to divide by this, not by
  //! mean_deflection(), or it corrects against the wrong denominator.
  //!
  //! Evaluated by quadrature over the quantile, so it is for load-time use.
  double sampled_mean_deflection(double E, int n_quantile = 1024) const;

  //! Determine whether angle distribution is empty
  //! \return Whether distribution is empty
  bool empty() const { return energy_.empty(); }

private:
  vector<double> energy_;
  vector<unique_ptr<Tabular>> distribution_;
  vector<double> mean_deflection_; //!< 1-<mu> of each tabulated distribution
  Interpolation energy_interp_ = Interpolation::lin_lin;
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
