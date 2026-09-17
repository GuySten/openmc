#ifndef OPENMC_ELECTROIONIZATION_H
#define OPENMC_ELECTROIONIZATION_H

#include "openmc/endf.h"
#include "openmc/tensor.h"
#include "openmc/vector.h"

#include <functional>

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

  //! Knock-on energy at one quantile of the sampled distribution
  //!
  //! sample() is this with the quantile drawn from the generator, so anything
  //! integrating the spectrum integrates exactly the distribution the
  //! transport samples rather than an interpolation of the tables behind it.
  //!
  //! \param[in] E Incident electron kinetic energy in [eV]
  //! \param[in] xi Quantile in [0, 1]
  //! \param[out] density Density of the sampled value in [1/eV], as in
  //!   sample()
  //! \return Knock-on kinetic energy in [eV]
  double at_quantile(double E, double xi, double* density = nullptr) const;

  //! Moments of the spectrum restricted to knock-on energies below a cutoff
  //!
  //! What a mixed condensed-history scheme needs in order to group the soft
  //! electroionization collisions: how often they happen and how much energy
  //! they carry off. The quantile map is monotonic, so the cutoff is a
  //! quantile and the soft probability is that quantile itself -- which is
  //! also why the hard part can be sampled afterwards by drawing a quantile
  //! above it, with no rejection and no change to the spectrum.
  //!
  //! \param[in] E Incident electron kinetic energy in [eV]
  //! \param[in] e_cut Largest knock-on energy counted as soft, in [eV]
  //! \param[in] weight Acceptance applied to a knock-on energy, for a
  //!   projectile whose spectrum is the tabulated one reweighted. Pass nullptr
  //!   for the electron, whose spectrum is the tabulated one itself.
  //! \param[out] xi_cut Quantile the cutoff sits at, so that 1 - xi_cut is the
  //!   fraction of collisions that stay hard
  //! \param[out] m0 Weighted probability of a soft collision; equal to
  //!   \p xi_cut when \p weight is null
  //! \param[out] m1 \f$\langle E_{knock} \rangle\f$ over the soft part,
  //!   weighted, in [eV]
  //! \param[out] m2 \f$\langle E_{knock}^2 \rangle\f$ over the soft part,
  //!   weighted, in [eV^2]
  void restricted_moments(double E, double e_cut,
    const std::function<double(double)>* weight, double& xi_cut, double& m0,
    double& m1, double& m2) const;

  //! Integrate a function of the knock-on energy over the soft part
  //!
  //! Same quadrature as restricted_moments(), over an arbitrary integrand
  //! rather than the powers of the knock-on energy. What needs it is the
  //! deflection a grouped collision makes, which depends on the density as
  //! well as the energy: the recoil model decides between a close and a
  //! distant collision by comparing the evaluated cross section at that
  //! transfer with the free one.
  //!
  //! \param[in] E Incident electron kinetic energy in [eV]
  //! \param[in] e_cut Largest knock-on energy counted as soft, in [eV]
  //! \param[in] f Integrand, taking the knock-on energy and the density there
  //! \return \f$\int_0^{\xi_{cut}} f \, d\xi\f$, a mean per collision
  //!   times the fraction of collisions that are soft
  double restricted_integral(double E, double e_cut,
    const std::function<double(double, double)>& f) const;

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
