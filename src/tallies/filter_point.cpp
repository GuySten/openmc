#include "openmc/tallies/filter_point.h"
#include "openmc/tallies/tally_scoring.h"

#include <cmath> // for exp

#include <fmt/core.h>

#include "openmc/math_functions.h"
#include "openmc/ray.h"
#include "openmc/simulation.h"
#include "openmc/xml_interface.h"

namespace openmc {

void PointFilter::from_xml(pugi::xml_node node)
{
  auto bins = get_node_array<double>(node, "bins");

  // Convert to vector of detectors
  vector<std::pair<Position, double>> detectors;
  size_t n = bins.size() / 4;
  for (int i = 0; i < n; ++i) {
    Position pos {bins[4 * i], bins[4 * i + 1], bins[4 * i + 2]};
    detectors.push_back(std::make_pair(pos, bins[4 * i + 3]));
  }
  this->set_detectors(detectors);
}

void PointFilter::set_detectors(span<std::pair<Position, double>> detectors)
{
  // Clear existing detectors
  detectors_.clear();
  detectors_.reserve(detectors.size());

  // Set detectors and number of bins
  for (auto d : detectors) {
    detectors_.push_back(d);
  }
  n_bins_ = detectors_.size();
}

void PointFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // A PointFilter puts its tally on the next-event estimator, and the only
  // thing that scores those is score_point_tally_impl(), which always hands
  // over the ParticleRay it just traced. Keying on the estimator makes that
  // invariant explicit here instead of assumed, and leaves the filter inert
  // if it is ever reached any other way.
  if (estimator != TallyEstimator::NEXT_EVENT)
    return;
  const auto& ray = static_cast<const ParticleRay&>(p);

  // Both quantities are recorded by the flight itself, so they are read back
  // rather than reconstructed: the distance was measured as the ray flew it,
  // and the optical depth was accumulated segment by segment.
  const double distance = ray.total_distance();
  const double attenuation = std::exp(-ray.traversal_mfp());

  int i = 0;
  for (auto [pos, r] : detectors_) {
    if ((ray.r() - pos).norm() < FP_COINCIDENT) {
      match.bins_.push_back(i);
      double weight;
      if (distance > r) {
        weight = attenuation / (distance * distance);
      } else {
        // Inside the exclusion sphere the 1/distance^2 singularity is replaced
        // by its average over a uniform isotropic source in a sphere of radius
        // r, which is 3 (1 - exp(-Sigma_t r)) / (Sigma_t r^3)
        weight = 3.0 * exprel(-ray.macro_xs().total * r) / (r * r);
      }
      match.weights_.push_back(weight);
    }
    ++i;
  }
}

void PointFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  vector<double> detectors;
  for (auto [pos, r] : detectors_) {
    detectors.push_back(pos[0]);
    detectors.push_back(pos[1]);
    detectors.push_back(pos[2]);
    detectors.push_back(r);
  }
  write_dataset(filter_group, "bins", detectors);
}

std::string PointFilter::text_label(int bin) const
{
  auto [pos, r] = detectors_.at(bin);
  return fmt::format("Point: {} {}", pos, r);
}

} // namespace openmc
