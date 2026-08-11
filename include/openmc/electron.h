#ifndef OPENMC_ELECTRON_H
#define OPENMC_ELECTRON_H

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
  ~ElectronInteraction();

  // Methods
  void calculate_xs(Particle& p) const;

  void elastic_scatter(double E, double* mu, uint64_t* seed) const;

  void excitation(double E_in, double* E_out) const;

  // Data members
  std::string name_; //!< Name of element, e.g. "Zr"
  int Z_;            //!< Atomic number
  int64_t index_;    //!< Index in global elements vector

  // Microscopic cross sections
  tensor::Tensor<double> energy_;
  tensor::Tensor<double> elastic_;
  tensor::Tensor<double> ionization_;
  tensor::Tensor<double> excitation_;
  tensor::Tensor<double> bremsstrahlung_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {
  
//! Photon interaction data for each element
extern std::unordered_map<std::string, int> element_map;
extern vector<unique_ptr<ElectronInteraction>> electroatomic;

} // namespace data

namespace detail {

double  evaluate_2BN_differential(double T_0, double k_photon, double theta);

double  sample_2BN(double T_0, double k_photon, uint64_t* seed);

double  sample_schiff_2BS(double E_electron, double k_photon, int Z, uint64_t* seed);
} // namespace detail  

} // namespace openmc

#endif // OPENMC_ELECTRON_H
