#include "openmc/tallies/filter_optical_depth.h"

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/ray.h"
#include "openmc/search.h"
#include "openmc/xml_interface.h"

namespace openmc {

void OpticalDepthFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<double>(node, "bins");
  this->set_bins(bins);
}

void OpticalDepthFilter::set_bins(span<double> bins)
{
  bins_.clear();
  bins_.reserve(bins.size());

  if (bins.size() < 2) {
    throw std::runtime_error {
      "Optical depth filter needs at least two bin edges."};
  }

  for (int64_t i = 0; i < bins.size(); ++i) {
    if (bins[i] < 0.0) {
      throw std::runtime_error {
        "Optical depth bins must not be negative: an optical depth is a "
        "path integral of a non-negative cross section."};
    }
    if (i > 0 && bins[i] <= bins[i - 1]) {
      throw std::runtime_error {
        "Optical depth bins must be monotonically increasing."};
    }
    bins_.push_back(bins[i]);
  }

  n_bins_ = bins_.size() - 1;
}

void OpticalDepthFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // The optical depth of a flight is only defined for a flight, and the only
  // thing that flies one is the next-event estimator. Keying on the estimator
  // makes that explicit rather than assumed, and leaves the filter inert if it
  // is ever reached any other way -- the rule PointFilter follows.
  if (estimator != TallyEstimator::NEXT_EVENT &&
      estimator != TallyEstimator::UNCOLLIDED)
    return;
  const auto& ray = static_cast<const ParticleRay&>(p);

  // Accumulated segment by segment as the ray flew, so it is read back rather
  // than reconstructed from the end points.
  const double tau = ray.traversal_mfp();

  // A contribution outside the binned range is dropped, as it is by any other
  // filter. That is worth watching in a buildup calculation, where the dropped
  // flux is silently missing from the folded answer: compare against the same
  // tally without this filter to see whether the range covers the problem.
  if (tau < bins_.front() || tau > bins_.back())
    return;

  match.bins_.push_back(lower_bound_index(bins_.begin(), bins_.end(), tau));
  match.weights_.push_back(1.0);
}

void OpticalDepthFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

std::string OpticalDepthFilter::text_label(int bin) const
{
  return fmt::format(
    "Optical Depth [{}, {}) mfp", bins_[bin], bins_[bin + 1]);
}

} // namespace openmc
