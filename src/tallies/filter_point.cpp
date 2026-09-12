#include "openmc/tallies/filter_point.h"
#include "openmc/tallies/tally_scoring.h"

#include <algorithm> // for min
#include <cmath>     // for exp, abs

#include <fmt/core.h>

#include "openmc/bounding_box.h"
#include "openmc/cell.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/math_functions.h"
#include "openmc/ray.h"
#include "openmc/simulation.h"
#include "openmc/surface.h"
#include "openmc/universe.h"
#include "openmc/xml_interface.h"

namespace openmc {

namespace {

//! A geometry state that reports a failure to locate itself instead of
//! aborting, so a detector outside the model is a warning not a fatal error.
class DetectorProbe : public GeometryState {
public:
  void mark_as_lost(const char*) override { lost_ = true; }
  using GeometryState::mark_as_lost;
  bool lost() const { return lost_; }

private:
  bool lost_ {false};
};

//! Shortest distance from a point to an axis-aligned box, zero if inside
double distance_to_box(const BoundingBox& bb, Position c)
{
  double dx = std::max({0.0, bb.min.x - c.x, c.x - bb.max.x});
  double dy = std::max({0.0, bb.min.y - c.y, c.y - bb.max.y});
  double dz = std::max({0.0, bb.min.z - c.z, c.z - bb.max.z});
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

//! The one material of a cell, or false if it has several (distribcell may
//! give each instance its own) and there is no single answer
bool single_material(const Cell& c, int32_t& mat)
{
  if (c.material_.empty())
    return false;
  mat = c.material_.front();
  for (int32_t m : c.material_) {
    if (m != mat)
      return false;
  }
  return true;
}

//! Smallest distance from a point to any surface bounding a cell, or a
//! negative value if the cell does not expose the surfaces that bound it or
//! one of them cannot supply a distance.
//!
//! Only CSGCell overrides Cell::surfaces(); every other kind inherits a base
//! that returns nothing. Reading that as "no surface is in the way" would
//! certify any sphere in a DAGMC model as clear without having examined
//! anything, so the geometry type is checked rather than the list's emptiness
//! -- a CSG cell with no bounding surfaces really is unbounded.
double distance_to_cell_boundary(const Cell& c, Position r)
{
  if (c.geom_type() != GeometryType::CSG)
    return -1.0;

  double closest = INFTY;
  for (int32_t token : c.surfaces()) {
    const Surface& surf {*model::surfaces[std::abs(token) - 1]};
    double d = surf.distance_to_point(r);
    if (d < 0.0)
      return -1.0;
    closest = std::min(closest, d);
  }
  return closest;
}

} // namespace

SphereCheck check_exclusion_sphere(Position pos, double r0, double& nearest)
{
  DetectorProbe probe;
  probe.init_from_r_u(pos, {0.0, 0.0, 1.0});
  if (!exhaustive_find_cell(probe) || probe.lost())
    return SphereCheck::OUTSIDE_MODEL;

  const int innermost = probe.n_coord() - 1;

  // Distance to the nearest boundary of the cell at each level. Levels above
  // the innermost are tracked separately: a sphere reaching one of those can
  // leave the innermost universe altogether, and the cells of that universe
  // then no longer bound where it can go.
  double closest_here = INFTY;
  double closest_above = INFTY;
  for (int level = 0; level <= innermost; ++level) {
    if (probe.coord(level).lattice() != C_NONE)
      return SphereCheck::UNDECIDABLE;

    double d = distance_to_cell_boundary(
      *model::cells[probe.coord(level).cell()], probe.coord(level).r());
    if (d < 0.0)
      return SphereCheck::UNDECIDABLE;

    if (level == innermost) {
      closest_here = d;
    } else {
      closest_above = std::min(closest_above, d);
    }
  }
  nearest = std::min(closest_here, closest_above);

  // The sphere never leaves the cell it started in, and a cell is one material
  if (nearest >= r0)
    return SphereCheck::SINGLE_MATERIAL;

  // It does leave, so the question is what else it reaches. That can only be
  // answered within the innermost universe.
  if (closest_above < r0)
    return SphereCheck::UNDECIDABLE;

  const int32_t mat = probe.material();
  const Position local = probe.coord(innermost).r();
  const Universe& uni {*model::universes[probe.coord(innermost).universe()]};
  for (int32_t i_cell : uni.cells_) {
    const Cell& c {*model::cells[i_cell]};

    // Bounding boxes over-approximate the cell, so a sphere that does not
    // reach the box certainly does not reach the cell
    if (distance_to_box(c.bounding_box(), local) >= r0)
      continue;

    // A neighbour filled by a universe or lattice has no single material of
    // its own to compare against
    if (c.type_ != Fill::MATERIAL)
      return SphereCheck::UNDECIDABLE;

    int32_t neighbour_mat;
    if (!single_material(c, neighbour_mat))
      return SphereCheck::UNDECIDABLE;
    if (neighbour_mat != mat)
      return SphereCheck::MULTIPLE_MATERIALS;
  }

  // Every cell the sphere can reach carries the detector's own material
  return SphereCheck::SINGLE_MATERIAL;
}

void check_point_detector_spheres(const PointFilter& filt)
{
  for (const auto& [pos, r0] : filt.detectors()) {
    if (r0 <= 0.0)
      continue;

    double nearest = INFTY;
    switch (check_exclusion_sphere(pos, r0, nearest)) {
    case SphereCheck::SINGLE_MATERIAL:
      break;
    case SphereCheck::MULTIPLE_MATERIALS:
      warning(fmt::format(
        "Point detector at {} has an exclusion sphere of radius {} that "
        "reaches a cell of a different material, the nearest cell boundary "
        "being {} away. The sphere treatment assumes a single total cross "
        "section throughout and reads it in the cell at the detector, so the "
        "result is biased -- and by the *neighbouring* material's optical "
        "thickness, which the detector's own cell gives no hint of. Reducing "
        "the radius below {} confines the sphere to one cell.",
        pos, r0, nearest, nearest));
      break;
    case SphereCheck::UNDECIDABLE:
      warning(fmt::format(
        "Point detector at {} has an exclusion sphere of radius {} which could "
        "not be shown to hold a single material: it sits in a lattice, is "
        "bounded by a surface with no closed-form distance to a point, or "
        "reaches a cell that is filled by a universe rather than a material. "
        "If it does span more than one material the treatment is biased.",
        pos, r0));
      break;
    case SphereCheck::OUTSIDE_MODEL:
      warning(fmt::format("Point detector at {} is outside the geometry; its "
                          "exclusion sphere could not be checked.",
        pos));
      break;
    }
  }
}

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

void PointFilter::build_detector_bins()
{
  bin_detector_.assign(detectors_.size(), C_NONE);
  for (int bin = 0; bin < detectors_.size(); ++bin) {
    const auto& pos = detectors_[bin].first;
    for (int i = 0; i < model::active_point_detectors.size(); ++i) {
      if (model::active_point_detectors[i] == pos) {
        bin_detector_[bin] = i;
        break;
      }
    }
  }
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

  // The ray says which detector it was aimed at, so bins are selected by
  // comparing indices. This used to compare the ray's end position against
  // every detector, which made a contribution's bin -- and so whether it was
  // counted at all -- depend on roundoff accumulated over the whole flight.
  const int i_detector = ray.detector_index();
  if (i_detector == C_NONE)
    return;

  // Bounded by bin_detector_, which is empty until build_detector_bins() has
  // run for this batch
  for (int bin = 0; bin < bin_detector_.size(); ++bin) {
    if (bin_detector_[bin] != i_detector)
      continue;

    const double r = detectors_[bin].second;
    double weight;
    if (distance > r) {
      weight = attenuation / (distance * distance);
    } else {
      // Inside the exclusion sphere the 1/distance^2 singularity is replaced
      // by its average over a uniform isotropic source in a sphere of radius
      // r, which is 3 (1 - exp(-Sigma_t r)) / (Sigma_t r^3).
      //
      // Sigma_t is taken in the cell holding the detector, which the average
      // assumes holds across the whole sphere. Substituting the mean along
      // the flight, tau/R, looks like a way to tolerate a sphere spanning
      // more than one material, but is not: it characterises the medium out
      // to R while the formula needs it out to r, so it over-corrects for a
      // distant emission and reduces to this value for a near one. Tested
      // against the exact sphere average it lowers the bias when the detector
      // sits in the lighter material and raises it when the detector sits in
      // the denser one, and it makes the weight vary from contribution to
      // contribution -- reintroducing the variance the sphere exists to
      // remove. Keeping the sphere within one material is the remedy.
      weight = 3.0 * exprel(-ray.macro_xs().total * r) / (r * r);
    }

    match.bins_.push_back(bin);
    match.weights_.push_back(weight);
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
