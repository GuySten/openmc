#ifndef OPENMC_TALLIES_TALLY_SCORING_H
#define OPENMC_TALLIES_TALLY_SCORING_H

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

//! Commit a super-history's staged adjoint-tally contributions
//!
//! Generation 0's contributions to adjoint=True tallies are staged (not
//! committed) during transport -- see the staging block in
//! score_general_ce_nonanalog/score_general_mg -- because the multiplier
//! (realized descendant weight at the final super-history generation)
//! isn't known until the whole super-history concludes. This applies that
//! multiplier and performs the real, atomic commit into each tally's
//! results, then clears the particle's staging buffer.
//!
//! \param p The particle whose staged super-history contributions should
//!   be committed. p.adjoint_weight() must already hold the realized
//!   terminal-generation weight for this super-history.
void commit_adjoint_scores(Particle& p);

} // namespace openmc

#endif // OPENMC_TALLIES_TALLY_SCORING_H
