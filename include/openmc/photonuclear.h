#ifndef OPENMC_PHOTONUCLEAR_H
#define OPENMC_PHOTONUCLEAR_H

#include "openmc/endf.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/particle.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/vector.h"

#include "openmc/tensor.h"
#include <hdf5.h>

#include <string>
#include <unordered_map>
#include <utility> // for pair

namespace openmc {

class PhotonuclearReaction {
public:
  //! Construct reaction from HDF5 data
  //! \param[in] group HDF5 group containing reaction data
  //! \param[in] name Name of the nuclide
  explicit PhotonuclearReaction(hid_t group, std::string name);

  //! Calculate cross section given grid index, interpolation factor
  //
  //! \param[in] i_grid Energy grid index
  //! \param[in] interp_factor Interpolation factor between grid points
  double xs(int64_t i_grid, double interp_factor) const;

  //! Calculate cross section
  //
  //! \param[in] micro Microscopic cross section cache
  double xs(const PhotonuclearMicroXS& micro) const;

  //! \brief Calculate reaction rate based on group-wise flux distribution
  //
  //! \param[in] energy Energy group boundaries in [eV]
  //! \param[in] flux Flux in each energy group (not normalized per eV)
  //! \param[in] grid Nuclide energy grid
  //! \return Reaction rate
  double collapse_rate(span<const double> energy, span<const double> flux,
    const vector<double>& grid) const;

  //! Cross section at a single temperature
  struct TemperatureXS {
    int threshold;
    vector<double> value;
  };

  int mt_;                           //!< ENDF MT value
  double q_value_;                   //!< Reaction Q value in [eV]
  bool scatter_in_cm_;               //!< scattering system in center-of-mass?
  bool redundant_;                   //!< redundant reaction?
  TemperatureXS xs_;                 //!< Cross section
  vector<ReactionProduct> products_; //!< Reaction products
};

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
  tensor::Tensor<double> energy_;
  tensor::Tensor<double> xs_; //!< Cross sections

  vector<unique_ptr<PhotonuclearReaction>> reactions_; //!< Reactions
private:
  void create_derived();

  static int XS_TOTAL;
  static int XS_HEATING;
  static int XS_NEUTRON_PROD;
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
