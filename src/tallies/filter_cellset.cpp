#include "openmc/tallies/filter_cellset.h"

#include <numeric> // for accumulate
#include <sstream>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/error.h"
#include "openmc/particle.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Non-member functions
//==============================================================================

std::string cell_set_sense_str(CellSetSense sense)
{
  switch (sense) {
  case CellSetSense::NET:
    return "net";
  case CellSetSense::OUT:
    return "out";
  case CellSetSense::IN:
    return "in";
  }
  UNREACHABLE();
}

CellSetSense cell_set_sense_from_str(const std::string& str)
{
  if (str == "net")
    return CellSetSense::NET;
  if (str == "out")
    return CellSetSense::OUT;
  if (str == "in")
    return CellSetSense::IN;
  fatal_error(fmt::format("Unknown sense '{}' on cell set filter. Valid "
                          "senses are 'net', 'out' and 'in'.",
    str));
}

//==============================================================================
// CellSetFilter implementation
//==============================================================================

void CellSetFilter::from_xml(pugi::xml_node node)
{
  auto cells = get_node_array<int32_t>(node, "bins");
  auto region_sizes = get_node_array<int32_t>(node, "region_sizes");

  // Convert cell IDs to indices of the global cells vector
  for (auto& c : cells) {
    auto search = model::cell_map.find(c);
    if (search == model::cell_map.end()) {
      fatal_error(
        fmt::format("Could not find cell {} specified on tally filter.", c));
    }
    c = search->second;
  }

  this->set_regions(cells, region_sizes);

  if (check_for_node(node, "senses")) {
    vector<CellSetSense> senses;
    for (const auto& str : get_node_array<std::string>(node, "senses")) {
      senses.push_back(cell_set_sense_from_str(str));
    }
    this->set_senses(senses);
  }
}

void CellSetFilter::set_regions(span<int32_t> cells, span<int32_t> region_sizes)
{
  if (region_sizes.empty()) {
    fatal_error("At least one region must be given on a cell set filter.");
  }

  int64_t total =
    std::accumulate(region_sizes.begin(), region_sizes.end(), int64_t {0});
  if (total != static_cast<int64_t>(cells.size())) {
    fatal_error(fmt::format("Cell set filter region sizes sum to {} but {} "
                            "cells were given.",
      total, cells.size()));
  }

  regions_.clear();
  region_cells_.clear();
  regions_.reserve(region_sizes.size());
  region_cells_.reserve(region_sizes.size());

  int64_t offset = 0;
  for (auto size : region_sizes) {
    if (size <= 0) {
      fatal_error("Each region on a cell set filter must contain at least one "
                  "cell.");
    }
    std::unordered_set<int32_t> region;
    vector<int32_t> ordered;
    region.reserve(size);
    ordered.reserve(size);
    for (int32_t i = 0; i < size; ++i) {
      region.insert(cells[offset + i]);
      ordered.push_back(cells[offset + i]);
    }
    regions_.push_back(std::move(region));
    region_cells_.push_back(std::move(ordered));
    offset += size;
  }

  n_bins_ = regions_.size() * senses_.size();
}

void CellSetFilter::set_senses(const vector<CellSetSense>& senses)
{
  if (senses.empty()) {
    fatal_error("At least one sense must be given on a cell set filter.");
  }

  // A repeated sense would silently duplicate bins, which is never intended
  for (int i = 0; i < senses.size(); ++i) {
    for (int j = i + 1; j < senses.size(); ++j) {
      if (senses[i] == senses[j]) {
        fatal_error(fmt::format("Sense '{}' given more than once on a cell set "
                                "filter.",
          cell_set_sense_str(senses[i])));
      }
    }
  }

  senses_ = senses;
  n_bins_ = regions_.size() * senses_.size();
}

bool CellSetFilter::was_inside(const Particle& p, int i_region) const
{
  const auto& region = regions_[i_region];
  for (int i = 0; i < p.n_coord_last(); ++i) {
    if (region.count(p.cell_last(i)) > 0)
      return true;
  }
  return false;
}

bool CellSetFilter::is_inside(const Particle& p, int i_region) const
{
  const auto& region = regions_[i_region];
  for (int i = 0; i < p.n_coord(); ++i) {
    if (region.count(p.coord(i).cell()) > 0)
      return true;
  }
  return false;
}

void CellSetFilter::match_region(
  FilterMatch& match, int i_region, bool leaving) const
{
  int n_senses = senses_.size();
  for (int i_sense = 0; i_sense < n_senses; ++i_sense) {
    int bin = i_region * n_senses + i_sense;
    switch (senses_[i_sense]) {
    case CellSetSense::NET:
      // Sign is carried as a filter weight so that the net current lands in a
      // single bin with a correct variance
      match.bins_.push_back(bin);
      match.weights_.push_back(leaving ? 1.0 : -1.0);
      break;
    case CellSetSense::OUT:
      if (leaving) {
        match.bins_.push_back(bin);
        match.weights_.push_back(1.0);
      }
      break;
    case CellSetSense::IN:
      if (!leaving) {
        match.bins_.push_back(bin);
        match.weights_.push_back(1.0);
      }
      break;
    }
  }
}

void CellSetFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  auto crossing = p.surface_crossing();

  for (int i_region = 0; i_region < regions_.size(); ++i_region) {
    bool from_inside = was_inside(p, i_region);
    bool to_inside = is_inside(p, i_region);

    // At a boundary condition the cells either side of the crossing no longer
    // describe what happened, so take the answer from the boundary instead.
    switch (crossing) {
    case SurfaceCrossing::LEAKED:
      // The particle left the model. The coordinate levels still name the cell
      // it was in, which would otherwise read as a crossing that went nowhere
      // and be discarded, losing the leakage from the region's net current.
      to_inside = false;
      break;
    case SurfaceCrossing::REFLECT_OUT:
      // First half of a reflection: treat it as reaching the boundary
      to_inside = false;
      break;
    case SurfaceCrossing::REFLECT_IN:
      // Second half: treat it as coming back in. Together the two halves give
      // an outgoing and an incoming current of equal size and a net of zero,
      // which is what a reflective boundary physically does.
      from_inside = false;
      break;
    case SurfaceCrossing::PERIODIC:
      // The particle left one face and entered its partner. When the region
      // holds cells at both faces this is a crossing of its boundary twice
      // over, once in each direction, and there is only this one scoring event
      // to record both.
      if (from_inside && to_inside) {
        match_region(match, i_region, true);
        match_region(match, i_region, false);
        continue;
      }
      break;
    case SurfaceCrossing::NORMAL:
      break;
    }

    // Only a crossing with one end inside the region and one end outside is a
    // crossing of that region's boundary. This is what lets a region be a
    // union: a crossing between two of its own cells does not score, and
    // neither does one that misses the region entirely.
    if (from_inside == to_inside)
      continue;

    match_region(match, i_region, from_inside);
  }
}

void CellSetFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);

  vector<int32_t> cell_ids;
  vector<int32_t> region_sizes;
  for (const auto& region : region_cells_) {
    region_sizes.push_back(region.size());
    for (auto c : region) {
      cell_ids.push_back(model::cells[c]->id_);
    }
  }

  vector<std::string> senses;
  for (auto sense : senses_) {
    senses.push_back(cell_set_sense_str(sense));
  }

  write_dataset(filter_group, "bins", cell_ids);
  write_dataset(filter_group, "region_sizes", region_sizes);
  write_dataset(filter_group, "senses", senses);
}

std::string CellSetFilter::text_label(int bin) const
{
  int n_senses = senses_.size();
  int i_region = bin / n_senses;
  int i_sense = bin % n_senses;

  vector<int32_t> ids;
  for (auto c : region_cells_[i_region]) {
    ids.push_back(model::cells[c]->id_);
  }

  std::stringstream out;
  out << "Cell set " << i_region + 1 << " (cells ";
  for (int i = 0; i < ids.size(); ++i) {
    out << (i == 0 ? "" : ", ") << ids[i];
  }
  out << "), " << cell_set_sense_str(senses_[i_sense]);
  return out.str();
}

} // namespace openmc
