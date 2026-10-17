#include "openmc/tallies/filter_musurface.h"

#include <cmath> // for abs, copysign

#include "openmc/search.h"
#include "openmc/surface.h"
#include "openmc/tallies/tally_scoring.h"

namespace openmc {

void MuSurfaceFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  // Get surface normal (and make sure it is a unit vector)
  const auto surf {model::surfaces[p.surface_index()].get()};
  auto n = surf->normal(p.r());
  n /= n.norm();

  // Determine cosine of angle between normal and particle direction. The
  // normal is taken as the surface reports it, so mu carries the sign of the
  // crossing and spans the full [-1, 1] range the bins are defined over.
  // Flipping the normal to face the direction of travel would leave every
  // crossing at mu >= 0, and a bin would then disagree with the sign of the
  // current scored into it.
  double mu = p.u().dot(n);
  if (std::abs(mu) > 1.0)
    mu = std::copysign(1.0, mu);

  // Find matching bin
  if (mu >= bins_.front() && mu <= bins_.back()) {
    auto bin = lower_bound_index(bins_.begin(), bins_.end(), mu);
    match.bins_.push_back(bin);
    match.weights_.push_back(1.0);
  }
}

} // namespace openmc
