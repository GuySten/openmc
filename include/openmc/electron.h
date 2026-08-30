#ifndef OPENMC_ELECTRON_H
#define OPENMC_ELECTRON_H

#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"
#include "openmc/endf.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/particle.h"
#include "openmc/vector.h"

#include "openmc/tensor.h"
#include <hdf5.h>

#include <string>
#include <unordered_map>
#include <utility> // for pair

namespace openmc {

//==============================================================================
//! Electron interaction data for a single element
//==============================================================================

class ElectronInteraction {
public:
  // Constructors/destructor
  ElectronInteraction(hid_t group);

  // Methods
  void calculate_xs(Particle& p) const;

  double elastic_scatter(double E, uint64_t* seed) const;

  double excitation(double E) const;

  void ionization(Particle& p, int i_shell) const;

  int sample_ionization_shell(Particle& p) const;

  void bremsstrahlung(Particle& p) const;

  // Data members
  std::string name_;      //!< Name of element, e.g. "Zr"
  int Z_;                 //!< Atomic number
  int64_t index_;         //!< Index in data::electroatomic
  int64_t i_photoatomic_; //!< Index in data::photoatomic for this element

  //! For each electroionization subshell, the index of the matching subshell in
  //! the photoatomic PhotonInteraction::shells_ vector. The two lists are not
  //! guaranteed to have the same length or ordering, so they are matched by
  //! ENDF designator rather than by position.
  vector<int> shell_map_;

  // Microscopic cross sections
  tensor::Tensor<double> energy_;
  tensor::Tensor<double> elastic_;
  AngleDistribution elastic_angle_;
  tensor::Tensor<double> ionization_;
  vector<unique_ptr<ContinuousTabular>> ionization_dist_;
  tensor::Tensor<double> excitation_;
  Tabulated1D excitation_energy_loss_;
  tensor::Tensor<double> bremsstrahlung_;
  unique_ptr<ContinuousTabular> bremsstrahlung_dist_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Maps element name to index in data::photoatomic. Owned by photon.cpp;
//! electron data must not write to it.
extern std::unordered_map<std::string, int> element_map;

//! Maps element name to index in data::electroatomic
extern std::unordered_map<std::string, int> electron_map;

extern vector<unique_ptr<ElectronInteraction>> electroatomic;

} // namespace data

namespace detail {

double evaluate_2BN_differential(double T_0, double k_photon, double theta);

double sample_2BN(double T_0, double k_photon, uint64_t* seed);

double sample_schiff_2BS(
  double E_electron, double k_photon, int Z, uint64_t* seed);
} // namespace detail

} // namespace openmc

#endif // OPENMC_ELECTRON_H
