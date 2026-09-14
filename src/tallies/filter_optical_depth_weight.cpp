#include "openmc/tallies/filter_optical_depth_weight.h"

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/ray.h"
#include "openmc/xml_interface.h"

namespace openmc {

void OpticalDepthWeightFilter::from_xml(pugi::xml_node node)
{
  n_bins_ = 1;
  if (check_for_node(node, "attenuation")) {
    std::string basis = get_node_value(node, "attenuation", true, true);
    if (basis == "total") {
      basis_ = OpticalDepthBasis::TOTAL;
    } else if (basis == "no-coherent") {
      basis_ = OpticalDepthBasis::NO_COHERENT;
    } else {
      throw std::runtime_error {fmt::format(
        "Unknown attenuation basis '{}' on optical depth filter {}. Use "
        "'total' or 'no-coherent'.",
        basis, id())};
    }
  }
}

void OpticalDepthWeightFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // The optical depth of a flight is only defined for a flight, and the only
  // thing that flies one is the next-event estimator.
  if (estimator != TallyEstimator::NEXT_EVENT &&
      estimator != TallyEstimator::UNCOLLIDED)
    return;
  const auto& ray = static_cast<const ParticleRay&>(p);

  // Accumulated segment by segment as the ray flew, so the depth is the one
  // the flight actually measured -- through voids, through whatever the
  // geometry holds -- and not a cross section times a distance.
  const double tau = (basis_ == OpticalDepthBasis::NO_COHERENT)
                       ? ray.traversal_mfp_excluding_coherent()
                       : ray.traversal_mfp();

  // Weighting rather than selecting: the single bin accumulates sum c_i tau_i.
  match.bins_.push_back(0);
  match.weights_.push_back(tau);
}

void OpticalDepthWeightFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "attenuation",
    basis_ == OpticalDepthBasis::NO_COHERENT ? "no-coherent" : "total");
}

std::string OpticalDepthWeightFilter::text_label(int bin) const
{
  return basis_ == OpticalDepthBasis::NO_COHERENT
           ? "Optical Depth weight (no coherent)"
           : "Optical Depth weight";
}

} // namespace openmc
