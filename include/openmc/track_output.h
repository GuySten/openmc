#ifndef OPENMC_TRACK_OUTPUT_H
#define OPENMC_TRACK_OUTPUT_H

#include "openmc/particle.h"

namespace openmc {

//==============================================================================
// Non-member functions
//==============================================================================

//! Open HDF5 track file for writing and create track datatype
void open_track_file();

//! Close HDF5 resources for track file
void close_track_file();

//! Determine whether a given particle should collect/write track information
//
//! \param[in] p  Current particle
//! \return Whether to collect/write track information
bool check_track_criteria(const Particle& p);

//! Create a new track state history for a primary/secondary particle
//
//! \param[in] p  Current particle
void add_particle_track(Particle& p);

//! Store particle's current state
//
//! \param[in] p  Current particle
void write_particle_track(Particle& p);

//! Write full particle state history to HDF5 track file
//
//! \param[in] p  Current particle
void finalize_particle_track(Particle& p);

//! Forget the source particles whose tracks are being written with the shared
//! secondary bank. Called before the source particles of a batch are
//! transported.
void reset_track_roots();

//! Record that the track of a source particle is being written so that the
//! secondaries in its history can be written too when transporting with the
//! shared secondary bank
//
//! \param[in] p  Source particle
void record_track_root(const Particle& p);

//! Gather the source particles recorded by record_track_root() on all ranks.
//! Called after the source particles of a batch have been transported and
//! before any secondaries are.
void synchronize_track_roots();

//! Look up the source particle at the root of a history
//
//! \param[in] root_index  Index of the source particle at the root
//! \return ID of the source particle if its track is being written, -1
//!   otherwise
int64_t track_root_id(int64_t root_index);

} // namespace openmc

#endif // OPENMC_TRACK_OUTPUT_H
