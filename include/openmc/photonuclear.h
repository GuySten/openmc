#ifndef OPENMC_PHOTONUCLEAR_H
#define OPENMC_PHOTONUCLEAR_H

#include "openmc/endf.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/particle.h"
#include "openmc/vector.h"

#include "xtensor/xtensor.hpp"
#include <hdf5.h>

#include <string>
#include <unordered_map>
#include <utility> // for pair

namespace openmc {

//==============================================================================
//! Photonuclear interaction data for a single isotope
//==============================================================================


class PhotonuclearInteraction {
public:
  // Constructors/destructor
  PhotonuclearInteraction(hid_t group);
  ~PhotonuclearInteraction();

  // Methods
  void calculate_xs(Particle& p) const;

  // Data members
  std::string name_; //!< Name of nuclide, e.g. "U235"
  int Z_;            //!< Atomic number
  int A_;            //!< Mass number
  int metastable_;   //!< Metastable state
  double awr_;       //!< Atomic weight ratio
  int64_t index_;    //!< Index in the photonuclears array  

  // Microscopic cross sections
  xt::xtensor<double, 1> energy_;
  xt::xtensor<double, 1> disappearance_;
  xt::xtensor<double, 1> heating_;

};

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_photonuclear();

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Photonuclear interaction data for each isotope
extern std::unordered_map<std::string, int> photonuclear_map;
extern vector<unique_ptr<PhotonuclearInteraction>> photonuclears;
extern double photonuclear_energy_min;

} // namespace data

} // namespace openmc

#endif // OPENMC_PHOTONUCLEAR_H
