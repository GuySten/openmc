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

  //! Determine whether angle distribution is empty
  //! \return Whether distribution is empty
  bool empty() const { return energy_.empty(); }

  //! Legendre transport moments of each tabulated distribution
  //!
  //! Returns \f$\langle 1 - P_\ell(\mu) \rangle\f$ for \f$\ell = 1, 2\f$,
  //! the quantities a condensed-history scheme groups soft collisions by. The
  //! first is what sets the transport mean free path: after a path \f$s\f$ in
  //! a medium of atom density \f$n\f$, \f$\langle\mu\rangle =
  //! \exp(-s n \sigma_{el} \langle 1-\mu \rangle)\f$.
  //!
  //! The integrals are exact for each tabulated segment rather than
  //! trapezoidal: the integrand is the product of the interpolated density
  //! with a polynomial in \f$\mu\f$, and the elastic distributions are so
  //! sharply forward-peaked that a quadrature error would show up directly in
  //! the step length.
  //!
  //! \param[out] energy Incident energies the moments are tabulated at, in [eV]
  //! \param[out] mu1 \f$\langle 1 - \mu \rangle\f$ at each energy
  //! \param[out] mu2 \f$\langle \frac{3}{2}(1 - \mu^2) \rangle\f$ at each
  //! energy
  void transport_moments(
    vector<double>& energy, vector<double>& mu1, vector<double>& mu2) const;

private:
  vector<double> energy_;
  vector<unique_ptr<Tabular>> distribution_;
  Interpolation energy_interp_ = Interpolation::lin_lin;
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
