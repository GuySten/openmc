#ifndef OPENMC_BREMSSTRAHLUNG_H
#define OPENMC_BREMSSTRAHLUNG_H

#include "openmc/particle.h"

#include "openmc/tensor.h"

namespace openmc {

//==============================================================================
// Bremsstrahlung classes
//==============================================================================

class BremsstrahlungData {
public:
  // Data
  tensor::Tensor<double> pdf;   //!< Bremsstrahlung energy PDF
  tensor::Tensor<double> cdf;   //!< Bremsstrahlung energy CDF
  tensor::Tensor<double> yield; //!< Photon yield
};

class Bremsstrahlung {
public:
  // Data
  BremsstrahlungData electron;
  BremsstrahlungData positron;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

extern tensor::Tensor<double>
  ttb_e_grid; //! energy T of incident electron in [eV]
extern tensor::Tensor<double>
  ttb_k_grid; //! reduced energy W/T of emitted photon

} // namespace data

//==============================================================================
// Global variables
//==============================================================================

//! Ratio of the positron to the electron bremsstrahlung cross section
//
//! A positron is repelled by the nucleus where an electron is attracted, so it
//! radiates less, the difference vanishing at high energy. The factor is
//! independent of the emitted photon energy, so it scales the cross section
//! and leaves the spectrum alone. Source: F. Salvat, J. M. Fernandez-Varea and
//! J. Sempau, "PENELOPE-2011: A Code System for Monte Carlo Simulation of
//! Electron and Photon Transport", OECD-NEA (2011).
//!
//! \param[in] Z_sq atomic number squared, or its equivalent for a mixture
//! \param[in] E    kinetic energy of the positron in [eV]
double positron_bremsstrahlung_factor(double Z_sq, double E);

void thick_target_bremsstrahlung(Particle& p);
void thick_target_bremsstrahlung(
  Particle& p, ParticleType type, Direction u, double E);

} // namespace openmc

#endif // OPENMC_BREMSSTRAHLUNG_H
