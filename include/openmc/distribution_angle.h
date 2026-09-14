//! \file distribution_angle.h
//! Angle distribution dependent on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ANGLE_H
#define OPENMC_DISTRIBUTION_ANGLE_H

#include "hdf5.h"

#include "openmc/distribution.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! How to interpolate between the tabulated incident energies
//==============================================================================

enum class AngleEnergyInterp {
  //! Choose one bracketing table at random, with a probability linear in
  //! energy. Correct where the tables are closely spaced and the distribution
  //! varies slowly between them, which is the case for neutron data.
  linear_stochastic,

  //! Sample both bracketing tables at a common quantile and interpolate the
  //! deflection geometrically, with a fraction linear in the logarithm of the
  //! energy. For tables spaced geometrically and sparsely, whose width follows
  //! a power of the energy, linear_stochastic is heavily biased toward the
  //! low-energy table across the whole interval; this is not.
  log_correlated,
};

//==============================================================================
//! Angle distribution that depends on incident particle energy
//==============================================================================

class AngleDistribution {
public:
  AngleDistribution() = default;

  //! \param[in] group HDF5 group to read the distribution from
  //! \param[in] interp How to interpolate between tabulated incident energies.
  //!   Defaults to the historical behaviour, so existing data and every
  //!   existing caller are unaffected.
  explicit AngleDistribution(hid_t group,
    AngleEnergyInterp energy_interp = AngleEnergyInterp::linear_stochastic);

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

private:
  vector<double> energy_;
  vector<unique_ptr<Tabular>> distribution_;
  AngleEnergyInterp interp_ = AngleEnergyInterp::linear_stochastic;
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
