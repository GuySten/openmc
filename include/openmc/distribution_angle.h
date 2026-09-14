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
  explicit AngleDistribution(hid_t group);

  //! Sample an angle given an incident particle energy
  //! \param[in] E Particle energy in [eV]
  //! \param[inout] seed pseudorandom number seed pointer
  //! \return Cosine of the angle in the range [-1,1]
  double sample(double E, uint64_t* seed) const;

  //! Sample an angle, interpolating between tables in log-energy
  //
  // For a distribution tabulated on a sparse, geometric energy grid whose
  // width varies as a power of the energy, sample() is badly biased -- see
  // the implementation. Used for electron elastic scattering; neutron data is
  // tabulated densely enough that sample() is fine and keeps using it.
  //
  //! \param[in] E Particle energy in [eV]
  //! \param[inout] seed pseudorandom number seed pointer
  //! \return Cosine of the angle in the range [-1,1]
  double sample_log_interp(double E, uint64_t* seed) const;

  //! Evaluate the angular PDF at a given energy and cosine
  //! \param[in] E Particle energy in [eV]
  //! \param[in] mu Cosine of the scattering angle
  //! \return Probability density for the scattering cosine
  double evaluate(double E, double mu) const;

  //! Determine whether angle distribution is empty
  //! \return Whether distribution is empty
  bool empty() const { return energy_.empty(); }

private:
  vector<double> energy_;
  vector<unique_ptr<Tabular>> distribution_;
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
