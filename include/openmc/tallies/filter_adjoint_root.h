#ifndef OPENMC_TALLIES_FILTER_ADJOINT_ROOT_H
#define OPENMC_TALLIES_FILTER_ADJOINT_ROOT_H

#include <cstdint>

#include "openmc/constants.h"
#include "openmc/tallies/filter.h"

namespace openmc {

//==============================================================================
//! Bins the scores of the adjoint side populations' detector trees by the
//! category of the tree's root (prompt fission neutron, delayed group,
//! photoneutron, probe or ray line) and, with a mesh, by the mesh element of
//! the fission event the root came from. Each score is divided by the source
//! weight of its bin in the generation, so that the tally is the detector
//! response per unit source weight. Only the detector trees score it.
//==============================================================================

class AdjointRootFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "adjointroot"; }
  FilterType type() const override { return FilterType::ADJOINT_ROOT; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  //! Index in model::meshes of the origin mesh, C_NONE if there is none
  int32_t mesh() const { return mesh_; }

  //! Set by the side populations once their categories are known
  void set_n_bins(int n_bins) { n_bins_ = n_bins; }

private:
  int32_t mesh_ {C_NONE};
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_ADJOINT_ROOT_H
