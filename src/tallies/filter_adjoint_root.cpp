#include "openmc/tallies/filter_adjoint_root.h"

#include <fmt/core.h>

#include "openmc/adjoint_populations.h"
#include "openmc/error.h"
#include "openmc/mesh.h"
#include "openmc/xml_interface.h"

namespace openmc {

void AdjointRootFilter::from_xml(pugi::xml_node node)
{
  // The bins hold the ID of the origin mesh, or nothing
  if (check_for_node(node, "bins")) {
    auto bins = get_node_array<int32_t>(node, "bins");
    if (bins.size() > 1) {
      fatal_error(
        fmt::format("Adjoint root filter {} takes at most one mesh.", id()));
    }
    if (bins.size() == 1) {
      auto search = model::mesh_map.find(bins[0]);
      if (search == model::mesh_map.end()) {
        fatal_error(fmt::format(
          "Could not find mesh {} specified on adjoint root filter {}.",
          bins[0], id()));
      }
      mesh_ = search->second;
    }
  }
  // The categories are counted by the side populations at initialization
  n_bins_ = 1;
}

void AdjointRootFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  adjpop::detector_bins(match);
}

void AdjointRootFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  int mesh_id = (mesh_ == C_NONE) ? -1 : model::meshes[mesh_]->id_;
  write_dataset(filter_group, "bins", mesh_id);
}

std::string AdjointRootFilter::text_label(int bin) const
{
  return adjpop::detector_bin_label(bin);
}

} // namespace openmc
