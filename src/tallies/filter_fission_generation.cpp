#include "openmc/tallies/filter_fission_generation.h"

#include <stdexcept>

#include <fmt/core.h>

#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// FissionGenerationFilter implementation
//==============================================================================

void FissionGenerationFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<int>(node, "bins");
  this->set_bins(bins);
}

void FissionGenerationFilter::set_bins(span<const int> bins)
{
  bins_.clear();
  bins_.reserve(bins.size());
  map_.clear();

  for (int64_t i = 0; i < bins.size(); ++i) {
    if (bins[i] < 0) {
      throw std::invalid_argument {
        fmt::format("Fission generation filter bin {} is negative.", bins[i])};
    }
    bins_.push_back(bins[i]);
    map_[bins[i]] = i;
  }

  n_bins_ = bins_.size();
}

void FissionGenerationFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  auto search = map_.find(p.fission_generation());
  if (search != map_.end()) {
    match.bins_.push_back(search->second);
    match.weights_.push_back(1.0);
  }
}

void FissionGenerationFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

std::string FissionGenerationFilter::text_label(int bin) const
{
  return fmt::format("Fission Generation {}", bins_[bin]);
}

} // namespace openmc
