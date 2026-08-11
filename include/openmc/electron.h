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

  void compton_scatter(double alpha, bool doppler, double* alpha_out,
    double* mu, int* i_shell, uint64_t* seed) const;

  double rayleigh_scatter(double alpha, uint64_t* seed) const;

  void pair_production(double alpha, double* E_electron, double* E_positron,
    double* mu_electron, double* mu_positron, uint64_t* seed) const;

  void atomic_relaxation(int i_shell, Particle& p) const;

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

} // namespace openmc

#endif // OPENMC_ELECTRON_H
