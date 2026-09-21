#ifndef OPENMC_ANGLE_ENERGY_H
#define OPENMC_ANGLE_ENERGY_H

#include <cstdint>

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

  //! Sample an outgoing energy and cosine, restricted to outgoing energies
  //! at or above a threshold, and report how much of the distribution that
  //! restriction kept.
  //!
  //! Used where only the high-energy tail of a spectrum can affect the
  //! answer -- photon production feeding photonuclear reactions -- so that
  //! sampling the rest of it, only to discard it, can be avoided. A caller
  //! emits the sampled particle at its weight times the returned mass, which
  //! is unbiased for any score that ignores outgoing energies below the
  //! threshold.
  //!
  //! The default samples the whole distribution and returns 1, which is the
  //! plain unrestricted draw: correct for every subclass, and what a caller
  //! gets wherever restricting is not implemented. An override need not honour
  //! the threshold exactly -- truncating at the nearest tabulated point below
  //! it is enough, since the caller discards what falls short anyway -- but
  //! the mass it returns must be the mass of what it actually sampled from.
  //!
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[in] E_min Lowest outgoing energy worth sampling in [eV]
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[out] mu Outgoing cosine with respect to current direction
  //! \param[inout] seed Pseudorandom seed pointer
  //! \return Probability mass of the restricted range, in (0, 1]
  virtual double sample_above(double E_in, double E_min, double& E_out,
    double& mu, uint64_t* seed) const
  {
    this->sample(E_in, E_out, mu, seed);
    return 1.0;
  }

  //! Upper bound on the outgoing energy that sample() can return
  //!
  //! Returned in the frame the distribution is tabulated in, so a
  //! center-of-mass distribution returns a center-of-mass energy. Used to
  //! decide, before transport begins, whether secondaries can exceed the
  //! available transport data. Must be an attainable bound, not merely a
  //! valid one, or the resulting guard is uselessly loose.
  //!
  //! \param[in] E_in Incoming energy in [eV]
  //! \return Maximum outgoing energy in [eV]
  virtual double max_energy(double E_in) const = 0;

  virtual ~AngleEnergy() = default;
};

} // namespace openmc

#endif // OPENMC_ANGLE_ENERGY_H
