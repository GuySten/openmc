#ifndef OPENMC_ANGLE_ENERGY_H
#define OPENMC_ANGLE_ENERGY_H

#include <cstdint>

#include "openmc/constants.h"

namespace openmc {

//==============================================================================
//! Abstract type that defines a correlated or uncorrelated angle-energy
//! distribution that is a function of incoming energy. Each derived type must
//! implement a sample() method that returns an outgoing energy and
//! scattering cosine given an incoming energy.
//==============================================================================

class AngleEnergy {
public:
  //! Sample an outgoing energy and scattering cosine
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[out] mu Outgoing cosine with respect to current direction
  //! \param[inout] seed Pseudorandom seed pointer
  virtual void sample(
    double E_in, double& E_out, double& mu, uint64_t* seed) const = 0;

  //! Sample an outgoing energy and evaluate the angular PDF
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[in] mu Scattering cosine with respect to current direction
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[inout] seed Pseudorandom seed pointer
  //! \return Probability density for the scattering cosine
  virtual double sample_energy_and_pdf(
    double E_in, double mu, double& E_out, uint64_t* seed) const = 0;

  //! Upper bound on the outgoing energy that sample() can return
  //!
  //! Returned in the frame the distribution is tabulated in, so a
  //! center-of-mass distribution returns a center-of-mass energy. Used to
  //! decide, before transport begins, whether secondaries can exceed the
  //! available transport data. The bound must be valid -- exceeding it during
  //! transport is a hard error -- and should be as tight as is cheap, since a
  //! loose bound needlessly lowers the maximum photon energy of the problem.
  //!
  //! This is deliberately not pure. Only photonuclear reaction products are
  //! ever bounded, so most subclasses have no reason to implement it, and a
  //! subclass forced to invent an answer is likely to invent a wrong one. The
  //! default says "cannot bound this", which the caller already handles.
  //!
  //! \param[in] E_in Incoming energy in [eV]
  //! \return Maximum outgoing energy in [eV], or INFTY if no bound is known
  virtual double max_energy(double E_in) const { return INFTY; }

  virtual ~AngleEnergy() = default;
};

} // namespace openmc

#endif // OPENMC_ANGLE_ENERGY_H
