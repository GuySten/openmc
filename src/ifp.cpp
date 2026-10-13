#include "openmc/ifp.h"

#include <algorithm> // for copy, fill, max, min
#include <cmath>     // for exp, log, lround

#include <fmt/core.h>

#include "openmc/bank.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/vector.h"

namespace openmc {

namespace simulation {

IFPStream<IFPBirth> ifp_birth {1};
IFPStream<double> ifp_lifetime;
bool ifp_denominator_on {false};

} // namespace simulation

namespace {

//! Incremented whenever the streams are reset, so that cached probabilities
//! never outlive the nuclear data they were computed from
int ifp_cache_epoch {0};

//! Fission channel of an IFP birth record
struct FissionChannel {
  int nuclide;  //!< Index in data::nuclides
  int reaction; //!< Index in Nuclide::fission_rx_
};

vector<FissionChannel> ifp_channels; //!< Channels by index
vector<int> ifp_first_channel;       //!< First channel of each nuclide
constexpr uint32_t IFP_ENERGY_MAX {(1u << 24) - 1}; //!< Largest grid index
double ifp_log_E_min {0.0};  //!< Log of the lowest grid energy
double ifp_log_E_step {0.0}; //!< Log spacing of the energy grid

} // namespace

void build_ifp_fission_channels()
{
  ifp_channels.clear();
  ifp_first_channel.assign(data::nuclides.size(), -1);
  for (int i = 0; i < data::nuclides.size(); ++i) {
    const auto& nuc {data::nuclides[i]};
    if (!nuc->fissionable_)
      continue;
    ifp_first_channel[i] = ifp_channels.size();
    for (int r = 0; r < nuc->fission_rx_.size(); ++r) {
      ifp_channels.push_back({i, r});
    }
  }
  if (ifp_channels.size() > IFP_NOT_FISSION) {
    fatal_error(fmt::format(
      "IFP records can distinguish at most {} fission reactions, but the "
      "model has {}.",
      IFP_NOT_FISSION, ifp_channels.size()));
  }

  // The incident energy grid spans the neutron energies of the data
  int neutron = ParticleType::neutron().transport_index();
  ifp_log_E_min = std::log(data::energy_min[neutron]);
  ifp_log_E_step =
    (std::log(data::energy_max[neutron]) - ifp_log_E_min) / IFP_ENERGY_MAX;
}

IFPBirth make_ifp_birth(int i_nuclide, int i_rx, double E_in, double E_out)
{
  uint32_t channel = ifp_first_channel[i_nuclide] + i_rx;
  double x = (std::log(E_in) - ifp_log_E_min) / ifp_log_E_step;
  uint32_t energy = static_cast<uint32_t>(std::lround(
    std::min(std::max(x, 0.0), static_cast<double>(IFP_ENERGY_MAX))));
  IFPBirth birth;
  birth.bits = (channel << 24) | energy;
  birth.E_out = static_cast<float>(E_out);
  return birth;
}

int ifp_birth_nuclide(const IFPBirth& birth)
{
  return ifp_channels[birth.channel()].nuclide;
}

int ifp_birth_reaction(const IFPBirth& birth)
{
  return ifp_channels[birth.channel()].reaction;
}

double ifp_birth_E_in(const IFPBirth& birth)
{
  return std::exp(
    ifp_log_E_min + (birth.bits & IFP_ENERGY_MAX) * ifp_log_E_step);
}

void ifp(const Particle& p, int64_t idx, const IFPBirth& birth)
{
  simulation::ifp_birth.store(p.current_work(), idx, birth);
  simulation::ifp_lifetime.store(p.current_work(), idx, p.lifetime());
}

bool ifp_history_complete(int64_t i_source)
{
  if (simulation::ifp_birth.enabled())
    return simulation::ifp_birth.complete(i_source);
  return simulation::ifp_lifetime.complete(i_source);
}

const IFPBirth* ifp_ancestor(int64_t i_source)
{
  if (!simulation::ifp_birth.enabled() ||
      !simulation::ifp_birth.complete(i_source))
    return nullptr;
  return &simulation::ifp_birth.source_bank()[i_source][0];
}

bool ifp_in_nuclide_bin(int64_t i_source, int i_nuclide)
{
  if (i_nuclide < 0)
    return true;
  const IFPBirth* ancestor = ifp_ancestor(i_source);
  return ancestor && ancestor->from_fission() &&
         ifp_birth_nuclide(*ancestor) == i_nuclide;
}

void ifp_delayed_group_probabilities(
  const IFPBirth& birth, double* probabilities)
{
  // The ancestor of a particle is the same for its whole history
  thread_local IFPBirth cached_birth;
  thread_local int cached_epoch {-1};
  thread_local double cached[MAX_DELAYED_GROUPS];

  if (cached_epoch == ifp_cache_epoch && birth == cached_birth) {
    std::copy(cached, cached + MAX_DELAYED_GROUPS, probabilities);
    return;
  }

  std::fill(probabilities, probabilities + MAX_DELAYED_GROUPS, 0.0);
  if (birth.from_fission()) {
    const auto& nuc {data::nuclides[ifp_birth_nuclide(birth)]};
    const Reaction& rx {*nuc->fission_rx_[ifp_birth_reaction(birth)]};
    double E_in = ifp_birth_E_in(birth);
    double E_out = birth.E_out;

    // Delayed emission probabilities, following sample_fission_neutron(): the
    // neutron is delayed with probability nu_d / nu_t, and the group is then
    // chosen by comparing a uniform number on [0, nu_d) with the cumulative
    // yields of groups 1 to n - 1, the last group taking any remainder; a
    // group whose cumulative yield passes nu_d is cut there
    int n_group = std::min(nuc->n_precursor_, MAX_DELAYED_GROUPS);
    double nu_t = nuc->nu(E_in, Nuclide::EmissionMode::total);
    double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed);
    if (n_group > 0 && nu_d > 0.0 && nu_t > 0.0) {
      double beta = nu_d / nu_t;
      double prob_group[MAX_DELAYED_GROUPS] = {0.};
      double cumulative = 0.0;
      for (int g = 1; g <= n_group; ++g) {
        double upper =
          (g < n_group)
            ? std::min(nu_d, cumulative + (*rx.products_[g].yield_)(E_in))
            : nu_d;
        prob_group[g - 1] = beta * std::max(0.0, upper - cumulative) / nu_d;
        cumulative = std::max(cumulative, upper);
      }

      // Mixture density of the outgoing energy and its delayed components. An
      // outgoing energy above the data's maximum is resampled, which scales a
      // density by its probability below that energy; this is left out, as
      // it is below 1e-5 in ENDF/B-VII.1 (prompt spectra at E above 13 MeV)
      auto density = [&](int i_product) {
        double f = rx.products_[i_product].energy_pdf(E_in, E_out);
        if (f < 0.0) {
          fatal_error(fmt::format(
            "The outgoing energy density of fission product {} of {} cannot "
            "be evaluated, which the ifp-beta-numerator score requires.",
            i_product, nuc->name_));
        }
        return f;
      };

      double total = (1.0 - beta) * density(0);
      for (int g = 1; g <= n_group; ++g) {
        probabilities[g - 1] = prob_group[g - 1] * density(g);
        total += probabilities[g - 1];
      }

      if (total > 0.0) {
        for (int g = 0; g < n_group; ++g)
          probabilities[g] /= total;
      } else {
        // The densities vanish where a neutron was nonetheless sampled, which
        // single precision rounding of E_out can cause at a table edge; use the
        // probabilities before the outgoing energy is known
        std::copy(prob_group, prob_group + n_group, probabilities);
      }
    }
  }

  std::copy(probabilities, probabilities + MAX_DELAYED_GROUPS, cached);
  cached_birth = birth;
  cached_epoch = ifp_cache_epoch;
}

void resize_simulation_ifp_banks()
{
  // The denominator only needs to know whether a history is complete, which
  // either stream tells
  if (simulation::ifp_denominator_on && !simulation::ifp_birth.enabled() &&
      !simulation::ifp_lifetime.enabled()) {
    simulation::ifp_lifetime.enable();
  }

  if (simulation::ifp_birth.enabled()) {
    // A time cutoff discards delayed neutrons emitted after it, so a
    // neutron's progeny no longer depend on its emission only through its
    // outgoing energy, which the delayed group probabilities assume
    int neutron = ParticleType::neutron().transport_index();
    if (settings::time_cutoff[neutron] < INFTY) {
      fatal_error("The ifp-beta-numerator score cannot be used with a neutron "
                  "time cutoff.");
    }
    build_ifp_fission_channels();
  }

  int64_t n = simulation::work_per_rank;
  simulation::ifp_birth.resize_banks(n, 3 * n);
  simulation::ifp_lifetime.resize_banks(n, 3 * n);
}

void reset_ifp_streams()
{
  simulation::ifp_birth.reset();
  simulation::ifp_lifetime.reset();
  simulation::ifp_denominator_on = false;
  ifp_channels.clear();
  ifp_first_channel.clear();
  ++ifp_cache_epoch;
}

#ifdef OPENMC_MPI
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<IFPBirth>>& births,
  const vector<vector<double>>& lifetimes)
{
  if (mpi::rank == 0) {
    // A births list also holds the site's own birth
    n_generation = simulation::ifp_birth.enabled()
                     ? static_cast<int>(births[0].size()) - 1
                     : static_cast<int>(lifetimes[0].size());
  }
  MPI_Bcast(&n_generation, 1, MPI_INT, 0, mpi::intracomm);
}
#endif

} // namespace openmc
