#ifndef OPENMC_ELECTROIONIZATION_H
#define OPENMC_ELECTROIONIZATION_H

#include "openmc/endf.h"
#include "openmc/tensor.h"
#include "openmc/vector.h"

#include <hdf5.h>

namespace openmc {

//==============================================================================
//! Knock-on energy spectrum of an electroionization subshell
//!
//! The evaluated spectra are not self-similar, so they cannot be sampled the
//! way a fission or bremsstrahlung spectrum is. Each ends exactly at the
//! kinematic limit \f$(T-B)/2\f$, which is affine in the incident energy, while
//! the low end is pinned near the binding energy and does not scale at all.
//! Remapping such a distribution onto an interpolated outgoing range -- the
//! unit-base transform the general tabulated law applies -- stretches the soft
//! end by the ratio of the two limits, which across the sparse EEDL incident
//! grid inflates the mean energy transfer by a factor of tens.
//!
//! So the two bracketing tables are inverted at one common quantile and the
//! results combined geometrically. That keeps the shape near the fixed lower
//! limit and lets only the upper endpoint move with the incident energy.
//!
//! This lives apart from ContinuousTabular deliberately. It shares none of that
//! class's sampling logic, it needs a density that class has no reason to
//! provide, and the interpolation law it keys on is one neutron data also uses
//! -- so folding it in would put an electron-only algorithm in the path of
//! every neutron secondary energy spectrum.
//==============================================================================

class ElectroionizationSpectrum {
public:
  //! \param[in] group HDF5 group holding `energy` and `distribution`
  explicit ElectroionizationSpectrum(hid_t group);

  //! Sample a knock-on energy
  //!
  //! \param[in] E Incident electron kinetic energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \param[out] density Density of the sampled value in [1/eV], or zero when
  //!   it is not available. It is the pushforward of the sampling quantile,
  //!   not the interpolation of the two tabulated densities, and it comes back
  //!   from here because the inversion has already located the point in both
  //!   tables -- recovering it afterwards would mean searching them again.
  //! \return Knock-on kinetic energy in [eV]
  double sample(double E, uint64_t* seed, double* density = nullptr) const;

private:
  //! Outgoing spectrum tabulated for one incident energy
  struct Table {
    Interpolation interpolation;  //!< within-table law; EEDL uses histogram
    tensor::Tensor<double> e_out; //!< knock-on energies in [eV]
    tensor::Tensor<double> p;     //!< probability density in [1/eV]
    tensor::Tensor<double> c;     //!< cumulative distribution
  };

  //! Invert one table's cumulative distribution
  //!
  //! \param[out] p_local Density at the returned value, for the pushforward
  double invert(int l, double c, double* p_local) const;

  vector<double> energy_; //!< incident energies in [eV]
  vector<Table> distribution_;
};

} // namespace openmc

#endif // OPENMC_ELECTROIONIZATION_H
