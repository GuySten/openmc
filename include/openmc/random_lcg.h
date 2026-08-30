#ifndef OPENMC_RANDOM_LCG_H
#define OPENMC_RANDOM_LCG_H

#include <cstdint>
#include <initializer_list>

namespace openmc {

//==============================================================================
// Module constants.
//==============================================================================

constexpr int N_STREAMS {4};
constexpr int STREAM_TRACKING {0};
constexpr int STREAM_SOURCE {1};
constexpr int STREAM_URR_PTABLE {2};
constexpr int STREAM_VOLUME {3};
constexpr int64_t DEFAULT_SEED {1};
constexpr uint64_t DEFAULT_STRIDE {152917ULL};

//==============================================================================
//! Generate a pseudo-random number using a linear congruential generator.
//! @param seed Pseudorandom number seed pointer
//! @return A random number between 0 and 1
//==============================================================================

double prn(uint64_t* seed);

//==============================================================================
//! Generate a random number which is 'n' times ahead from the current seed.
//!
//! The result of this function will be the same as the result from calling
//! `prn()` 'n' times, though without the side effect of altering the RNG
//! state.
//! @param n The number of RNG seeds to skip ahead by
//! @param seed Pseudorandom number seed
//! @return A random number between 0 and 1
//==============================================================================

double future_prn(int64_t n, uint64_t seed);

//==============================================================================
//! Set a RNG seed to a unique value based on a unique particle ID by striding
//! the seed.
//! @param id The particle ID
//! @param offset The offset from the master seed to be used (e.g., for creating
//! different streams)
//! @return The initialized seed value
//==============================================================================

//! Fold several small counters into one id fit for init_particle_seeds().
//!
//! init_seed() and init_particle_seeds() separate streams by multiplying the
//! id by prn_stride; they do no mixing and assume the id they are handed is
//! already distinct and unstructured. An id assembled from a generation
//! number, a particle id and an event counter is neither -- consecutive
//! callers differ in one low bit -- so it has to be scrambled first.
//!
//! Uses the splitmix64 finalizer on each component in turn. Boost's
//! hash_combine (in random_ray/source_region.h) is NOT a substitute: it is
//! built for hash buckets, and consecutive inputs flip about 18 of 64 bits
//! against splitmix64's 32. Neighbouring seeds give correlated random
//! sequences, which is the one thing a seed must not do.
//!
//! \param components identity of the thing being seeded, in any fixed order
//! \return non-negative id, well separated for nearby components
int64_t combine_ids(std::initializer_list<int64_t> components);

uint64_t init_seed(int64_t id, int offset);

//==============================================================================
//! Set the RNG seeds to unique values based on the ID of the particle. This
//! function initializes the seeds for all RNG streams of the particle via
//! striding.
//! @param seeds Pseudorandom number seed array
//! @param id The particle ID
//==============================================================================

void init_particle_seeds(int64_t id, uint64_t* seeds);

//==============================================================================
//! Advance the random number seed 'n' times from the current seed. This
//! differs from the future_prn() function in that this function does alter
//! the RNG state.
//! @param seed Pseudorandom number seed pointer
//! @param n The number of RNG seeds to skip ahead by
//==============================================================================

void advance_prn_seed(int64_t n, uint64_t* seed);

//==============================================================================
//! Advance a random number seed 'n' times.
//!
//! This is usually used to skip a fixed number of random numbers (the stride)
//! so that a given particle always has the same starting seed regardless of
//! how many processors are used.
//! @param n The number of RNG seeds to skip ahead by
//! @param seed The starting to seed to advance from
//==============================================================================

uint64_t future_seed(uint64_t n, uint64_t seed);

//==============================================================================
//                               API FUNCTIONS
//==============================================================================

//==============================================================================
//! Get OpenMC's master seed.
//==============================================================================

extern "C" int64_t openmc_get_seed();

//==============================================================================
//! Set OpenMC's master seed.
//! @param new_seed The master seed. All other seeds will be derived from this
//! one.
//==============================================================================

extern "C" void openmc_set_seed(int64_t new_seed);

//==============================================================================
//! Get OpenMC's stride.
//==============================================================================

extern "C" uint64_t openmc_get_stride();

//==============================================================================
//! Set OpenMC's stride.
//! @param new_stride Stride.
//==============================================================================

extern "C" void openmc_set_stride(uint64_t new_stride);

} // namespace openmc
#endif // OPENMC_RANDOM_LCG_H
