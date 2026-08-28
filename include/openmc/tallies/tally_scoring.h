#ifndef OPENMC_TALLIES_TALLY_SCORING_H
#define OPENMC_TALLIES_TALLY_SCORING_H

#include <unordered_map>

#include "openmc/particle.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/tally.h"

namespace openmc {

//==============================================================================
//! An iterator over all combinations of a tally's matching filter bins.
//
//! This iterator handles two distinct tasks.  First, it maps the N-dimensional
//! space created by the indices of N filters onto a 1D sequence.  In other
//! words, it provides a single number that uniquely identifies a combination of
//! bins for many filters.  Second, it handles the task of finding each valid
//! combination of filter bins given that each filter can have 1 or 2 or many
//! bins that are valid for the current tally event.
//==============================================================================

class FilterBinIter {
public:
  //! Construct an iterator over bins that match a given particle's state.
  FilterBinIter(const Tally& tally, Particle& p);

  //! Construct an iterator over all filter bin combinations.
  //
  //! \param end if true, the returned iterator indicates the end of a loop.
  FilterBinIter(
    const Tally& tally, bool end, vector<FilterMatch>* particle_filter_matches);

  bool operator==(const FilterBinIter& other) const
  {
    return index_ == other.index_;
  }

  bool operator!=(const FilterBinIter& other) const
  {
    return !(*this == other);
  }

  FilterBinIter& operator++();

  int64_t index_ {1};
  double weight_ {1.};

  vector<FilterMatch>& filter_matches_;

private:
  void compute_index_weight();

  const Tally& tally_;
};

//==============================================================================
// Non-member functions
//==============================================================================

//! Score tallies using a 1 / Sigma_t estimate of the flux.
//
//! This is triggered after every collision.  It is invalid for tallies that
//! require post-collison information because it can score reactions that didn't
//! actually occur, and we don't a priori know what the outcome will be for
//! reactions that we didn't sample.  It is assumed the material is not void
//! since collisions do not occur in voids.
//
//! \param p The particle being tracked
void score_collision_tally(Particle& p);

//! Score tallies based on a simple count of events (for continuous energy).
//
//! Analog tallies are triggered at every collision, not every event.
//
//! \param p The particle being tracked
void score_analog_tally_ce(Particle& p);

//! Score tallies based on a simple count of events (for multigroup).
//
//! Analog tallies are triggered at every collision, not every event.
//
//! \param p The particle being tracked
void score_analog_tally_mg(Particle& p);

//! Score tallies using a tracklength estimate of the flux.
//
//! This is triggered at every event (surface crossing, lattice crossing, or
//! collision) and thus cannot be done for tallies that require post-collision
//! information.
//
//! \param p The particle being tracked
//! \param distance The distance in [cm] traveled by the particle
void score_tracklength_tally(Particle& p, double distance);

//! Score time filtered tallies using a tracklength estimate of the flux.
//
//! This is triggered at every event (surface crossing, lattice crossing, or
//! collision) and thus cannot be done for tallies that require post-collision
//! information.
//
//! \param p The particle being tracked
//! \param total_distance The distance in [cm] traveled by the particle
void score_timed_tracklength_tally(Particle& p, double total_distance);

//! Score mesh-surface tallies for particle currents.
//
//! \param p The particle being tracked
//! \param tallies A vector of the indices of the tallies to score to
void score_meshsurface_tally(Particle& p, const vector<int>& tallies);

//! Score surface tallies for particle currents.
//
//! \param p The particle being tracked
//! \param tallies A vector of the indices of the tallies to score to
//! \param normal The normal of the surface being crossed
void score_surface_tally(
  Particle& p, const vector<int>& tallies, const Direction& normal);

//! Score the pulse-height tally
//! This is triggered at the end of every particle history
//
//! \param p The particle being tracked
//! \param tallies A vector of the indices of the tallies to score to
void score_pulse_height_tally(Particle& p, const vector<int>& tallies);

//! Commit a super-history's staged, per-fission-event adjoint-tally
//! contributions
//!
//! phi-dagger is a single function of phase space and does not depend on
//! the score. What the score determines is only WHERE phi-dagger is
//! evaluated, via its operator: diagonal operators (inverse-velocity,
//! flux, ordinary reaction rates) evaluate it at the parent, while the
//! fission operator emits at a new energy drawn from chi(E) and so
//! evaluates it at the emitted neutron. This routes each score
//! accordingly and performs the real, atomic commit into each tally's
//! results, then clears the per-super-history staging state.
//!
//! \param p The particle whose staged super-history contributions should
//!   be committed.
//! \param terminal_weight_by_site Realized terminal-generation weight,
//!   keyed by the adjoint_id of the generation-0-emitted neutron it
//!   descends from. A neutron with no entry (or an entry of 0)
//!   contributes nothing -- its subtree died out before the terminal
//!   generation.
//! Record the filter bins matched by a generation-0-emitted fission neutron
//!
//! Must be called as the neutron is created: adjoint tallies use the
//! track-length estimator, so their own scoring call happens in
//! event_advance over the segment BEFORE this collision and cannot supply
//! the bins that fission production belongs in.
//!
//! Evaluates the filters for every adjoint tally at the EMITTED neutron's
//! phase point -- its outgoing energy for EnergyoutFilter, its precursor
//! group for DelayedGroupFilter -- so that siblings born at one collision
//! are separated correctly. Stores the result in
//! p.adjoint_site_filters()[site_id]. Leaves the particle's filter matches
//! invalidated, so any scoring later in this same event recomputes them.
//!
//! \param p The particle at the fission event
//! \param site_id The emitted neutron's adjoint_id
//! \param site The emitted neutron's source site
void record_adjoint_site_filters(
  Particle& p, int site_id, const SourceSite& site);

void commit_adjoint_scores(
  Particle& p, const std::unordered_map<int, double>& terminal_weight_by_site);

} // namespace openmc

#endif // OPENMC_TALLIES_TALLY_SCORING_H
