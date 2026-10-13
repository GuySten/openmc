#ifndef OPENMC_IFP_H
#define OPENMC_IFP_H

#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"

#include <algorithm> // for copy
#include <cstdint>   // for int32_t

namespace openmc {

//==============================================================================
//! The fission event that emitted a neutron, recorded for IFP.
//
//! The delayed group probabilities scored by ifp-beta-numerator-cond and
//! ifp-beta-numerator-group are a function of the fissioning nuclide, the
//! fission reaction, the incident energy, and the sampled outgoing energy, so
//! those are stored instead of the probabilities themselves and the
//! probabilities are computed when they are scored. The nuclide, reaction and
//! sampled delayed group are packed into one integer and the energies are
//! stored in single precision, which changes the probabilities by less than a
//! part in 10^6, for 12 bytes per record.
//==============================================================================

class IFPBirth {
public:
  //! Record of a neutron that was not born from fission
  IFPBirth() = default;

  //! Record of a fission neutron
  //! \param[in] i_nuclide Index of the fissioning nuclide
  //! \param[in] i_rx Index of the fission reaction in Nuclide::fission_rx_
  //! \param[in] delayed_group Sampled delayed group (0 for prompt)
  //! \param[in] E_in Energy of the neutron causing fission in [eV]
  //! \param[in] E_out Energy of the fission neutron in [eV]
  IFPBirth(
    int i_nuclide, int i_rx, int delayed_group, double E_in, double E_out)
    : id_ {(i_nuclide << 6) | (i_rx << 4) | delayed_group},
      E_in_ {static_cast<float>(E_in)}, E_out_ {static_cast<float>(E_out)}
  {}

  bool from_fission() const { return id_ >= 0; }
  int nuclide() const { return id_ >> 6; }
  int reaction() const { return (id_ >> 4) & 0x3; }
  int delayed_group() const { return from_fission() ? id_ & 0xF : 0; }
  double E_in() const { return E_in_; }
  double E_out() const { return E_out_; }

  bool operator==(const IFPBirth& other) const
  {
    return id_ == other.id_ && E_in_ == other.E_in_ && E_out_ == other.E_out_;
  }

private:
  int32_t id_ {
    -1}; //!< Nuclide (bits 6-30), reaction (4-5), delayed group (0-3)
  float E_in_ {0.0f};  //!< Incident energy in [eV]
  float E_out_ {0.0f}; //!< Outgoing energy in [eV]
};

//! Update a list of values by adding a new value if the size
//! of the list can accomodate the new value or by shifting all
//! values to the left (removing the first value of the list
//! and adding the new value at the end of the list).
//!
//! \param[in] value Value to add to the list
//! \param[in] data Initial version of the list
//! \return Updated list
template<typename T>
vector<T> _ifp(const T& value, const vector<T>& data)
{
  vector<T> updated;
  size_t source_idx = data.size();

  if (source_idx < settings::ifp_n_generation) {
    updated.resize(source_idx + 1);
    for (size_t i = 0; i < source_idx; i++) {
      updated[i] = data[i];
    }
    updated[source_idx] = value;
  } else if (source_idx == settings::ifp_n_generation) {
    updated.resize(source_idx);
    for (size_t i = 0; i < source_idx - 1; i++) {
      updated[i] = data[i + 1];
    }
    updated[source_idx - 1] = value;
  }
  return updated;
}

//==============================================================================
//! One stream of Iterated Fission Probability history data.
//
//! IFP tracks two quantities with identical bookkeeping: the delayed group
//! number of the ancestor neutron (an int) and its lifetime (a double). Each is
//! a per-particle list of the last ifp_n_generation values, held in a source
//! bank and a fission bank, and each is maintained only if a tally asked for
//! it.
//!
//! Bundling the flag with the two banks it governs means every operation is a
//! no-op when the stream is off, so callers do not branch, and the banks cannot
//! be resized independently of the flag.
//==============================================================================

template<typename T>
class IFPStream {
public:
  //! Whether this stream is being maintained
  bool enabled() const { return enabled_; }

  //! Begin maintaining this stream, when a tally requests a dependent score
  void enable() { enabled_ = true; }

  //! Clear the stream, including the flag. Derived from the tallies present in
  //! the model, so it must not persist into the next model in this process.
  void reset()
  {
    enabled_ = false;
    source_bank_.clear();
    fission_bank_.clear();
  }

  vector<vector<T>>& source_bank() { return source_bank_; }
  const vector<vector<T>>& source_bank() const { return source_bank_; }
  vector<vector<T>>& fission_bank() { return fission_bank_; }

  //! Resize both banks
  void resize_banks(int64_t n_source, int64_t n_fission)
  {
    if (!enabled_)
      return;
    source_bank_.resize(n_source);
    fission_bank_.resize(n_fission);
  }

  //! Append a value to the ancestor history of a new fission site
  void store(int64_t i_source, int64_t i_fission, const T& value)
  {
    if (!enabled_)
      return;
    fission_bank_[i_fission] = _ifp(value, source_bank_[i_source]);
  }

  //! Resize a caller-owned buffer for this stream, if it is enabled.
  //
  //! Templated on the container because callers pass both the per-particle
  //! banks (`vector<vector<T>>`) and the flat MPI serialization buffers
  //! (`vector<T>`).
  template<typename V>
  void resize_temp(V& temp, int64_t n) const
  {
    if (enabled_)
      temp.resize(n);
  }

  //! Copy one entry of the fission bank into a caller-owned buffer
  //
  //! \param[in] i_fission Index in this stream's fission bank
  //! \param[in] i_temp Index in the destination buffer
  //! \param[out] out Destination buffer
  void copy_from_fission(
    int64_t i_fission, int64_t i_temp, vector<vector<T>>& out) const
  {
    if (enabled_)
      out[i_temp] = fission_bank_[i_fission];
  }

  //! Copy a temporary buffer into the fission bank
  void copy_temp_to_fission(const vector<T>* temp, size_t n)
  {
    if (enabled_)
      std::copy(temp, temp + n, fission_bank_.data());
  }

  //! Copy a run of a temporary buffer into the source bank
  void copy_temp_to_source(
    int64_t i_temp, int64_t n, int64_t i_source, const vector<vector<T>>& temp)
  {
    if (enabled_)
      std::copy(&temp[i_temp], &temp[i_temp + n], &source_bank_[i_source]);
  }

  //! Copy a whole temporary buffer into the source bank
  void copy_all_temp_to_source(const vector<vector<T>>& temp, int64_t n)
  {
    if (enabled_)
      std::copy(temp.data(), temp.data() + n, source_bank_.begin());
  }

private:
  bool enabled_ {false};
  vector<vector<T>> source_bank_;
  vector<vector<T>> fission_bank_;
};

namespace simulation {

extern IFPStream<int> ifp_delayed_group; //!< Delayed group numbers
extern IFPStream<double> ifp_lifetime;   //!< Neutron lifetimes

//! Fission events that emitted the ancestors. Unlike the other streams, each
//! list ends with the site's own birth, so it holds up to ifp_n_generation + 1
//! records.
extern IFPStream<IFPBirth> ifp_birth;

} // namespace simulation

//! Whether Iterated Fission Probability is in use at all
inline bool ifp_on()
{
  return simulation::ifp_delayed_group.enabled() ||
         simulation::ifp_lifetime.enabled() || simulation::ifp_birth.enabled();
}

//! \brief Iterated Fission Probability (IFP) method.
//!
//! Add the IFP information in the IFP banks using the same index
//! as the one used to append the fission site to the fission bank.
//! Multithreading protection is guaranteed by the index returned by the
//! thread_safe_append call in physics.cpp.
//!
//! \param[in] p Particle
//! \param[in] idx Bank index from the thread_safe_append call in physics.cpp
//! \param[in] birth Fission event that emitted the new fission site, only read
//!   when ifp_birth is enabled
void ifp(const Particle& p, int64_t idx, const IFPBirth& birth);

//! Delayed group of the ancestor whose score a fission of this source particle
//! adds, or -1 if the particle does not yet have ifp_n_generation ancestors
//! \param[in] i_source Index of the particle in the source bank
int ifp_ancestor_delayed_group(int64_t i_source);

//! Whether a source particle has ifp_n_generation ancestors recorded
//! \param[in] i_source Index of the particle in the source bank
bool ifp_history_complete(int64_t i_source);

//! Delayed group probabilities of the ancestor whose score a fission of this
//! particle adds. The ancestor is the same for the whole history, so they are
//! computed once and reused for every fission and tally.
//! \param[in] p Particle
//! \param[out] weights Probabilities for groups 1..MAX_DELAYED_GROUPS
//! \return Whether the particle has ifp_n_generation ancestors recorded
bool ifp_ancestor_dg_weights(const Particle& p, double* weights);

//! Compute the probability that a fission neutron was emitted in each delayed
//! group given the fission event and its sampled outgoing energy.
//!
//! The neutron is sampled from the mixture
//!   f(E_out) = (1 - beta) chi_p(E_out) + sum_g beta_g chi_g(E_out),
//! so the probability of group g given E_out is beta_g chi_g(E_out) / f(E_out).
//! Scoring these probabilities in place of the 0/1 delayed indicator gives the
//! same expected IFP beta numerator, because the progeny of the neutron depend
//! on how it was emitted only through its outgoing energy.
//!
//! \param[in] i_nuclide Index of the fissioning nuclide
//! \param[in] rx Fission reaction that produced the neutron
//! \param[in] E_in Energy of the neutron causing fission in [eV]
//! \param[in] E_out Sampled energy of the fission neutron in [eV]
//! \param[in] delayed_group Sampled delayed group (0 for prompt), used only
//!   when the mixture density vanishes at E_out
//! \param[out] weights Probabilities for groups 1..MAX_DELAYED_GROUPS
void ifp_delayed_group_weights(int i_nuclide, const Reaction& rx, double E_in,
  double E_out, int delayed_group, double* weights);

//! Resize the IFP banks used in the simulation
void resize_simulation_ifp_banks();

//! Clear both streams, including their flags
void reset_ifp_streams();

#ifdef OPENMC_MPI

//! Deserialization information for transfer of IFP data using MPI
struct DeserializationInfo {
  int64_t index_local; //!< local index
  int64_t n;           //!< number of sites sent
};

//! Broadcast the number of generations, determined from the first element on
//! the first processor of whichever stream is enabled.
//!
//! \param[in,out] n_generation Number of generations
//! \param[in] delayed_groups List of delayed group numbers lists
//! \param[in] lifetimes List of lifetimes lists
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes);

//! Send IFP data using MPI.
//!
//! \param[in] idx Index of the first site
//! \param[in] n Number of sites to send
//! \param[in] n_generation Number of generations
//! \param[in] neighbor Index of the neighboring processor
//! \param[in] requests MPI requests
//! \param[in] data List of data lists
//! \param[out] send_data data buffer
template<typename T>
void send_ifp_info(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, const vector<vector<T>>& data,
  vector<T>& send_data)
{
  // Copy data in buffer
  for (int i = idx; i < idx + n; i++) {
    std::copy(
      data[i].begin(), data[i].end(), send_data.begin() + i * n_generation);
  }

  // Send data
  requests.emplace_back();
  MPI_Datatype datatype = mpi::MPITypeMap<T>::mpi_type;
  MPI_Isend(&send_data[n_generation * idx], n_generation * static_cast<int>(n),
    datatype, neighbor, mpi::rank, mpi::intracomm, &requests.back());
}

//! Receive IFP data using MPI.
//!
//! \param[in] idx Index of the first site
//! \param[in] n Number of sites to receive
//! \param[in] n_generation Number of generations
//! \param[in] neighbor Index of the neighboring processor
//! \param[in] requests MPI requests
//! \param[in] data data buffer
//! \param[out] deserialization Information to deserialize the received data
template<typename T>
void receive_ifp_data(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, vector<T>& data,
  vector<DeserializationInfo>& deserialization)
{
  requests.emplace_back();
  MPI_Datatype datatype = mpi::MPITypeMap<T>::mpi_type;
  MPI_Irecv(&data[n_generation * idx], n_generation * static_cast<int>(n),
    datatype, neighbor, neighbor, mpi::intracomm, &requests.back());

  // Deserialization info to reconstruct data later
  DeserializationInfo info = {idx, n};
  deserialization.push_back(info);
}

//! Deserialize IFP information received using MPI into a stream's source bank.
//!
//! \param[in] n_generation Number of generations
//! \param[in] data data to deserialize
//! \param[in,out] stream Stream whose source bank receives the data
//! \param[in] deserialization Information to deserialize the received data
template<typename T>
void deserialize_ifp_info(int n_generation, const vector<T>& data,
  IFPStream<T>& stream, const vector<DeserializationInfo>& deserialization)
{
  if (!stream.enabled())
    return;

  auto& bank = stream.source_bank();
  for (auto info : deserialization) {
    int64_t index_local = info.index_local;
    int64_t n = info.n;

    for (int i = index_local; i < index_local + n; i++) {
      vector<T> data_received(
        data.begin() + n_generation * i, data.begin() + n_generation * (i + 1));
      bank[i] = data_received;
    }
  }
}

#endif

} // namespace openmc

#endif // OPENMC_IFP_H
