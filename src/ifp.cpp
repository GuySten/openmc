#include "openmc/ifp.h"

#include <algorithm> // for copy, fill, max, min

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

} // namespace

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
         ancestor->nuclide() == i_nuclide;
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
    const auto& nuc {data::nuclides[birth.nuclide()]};
    const Reaction& rx {*nuc->fission_rx_[birth.reaction()]};
    double E_in = birth.E_in;
    double E_out = birth.E_out;

    // Delayed emission probabilities, following sample_fission_neutron(): the
    // neutron is delayed with probability nu_d / nu_t, and the group is then
    // chosen from the cumulative yields, the last group taking any remainder
    int n_group = std::min(nuc->n_precursor_, MAX_DELAYED_GROUPS);
    double nu_t = nuc->nu(E_in, Nuclide::EmissionMode::total);
    double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed);
    if (n_group > 0 && nu_d > 0.0 && nu_t > 0.0) {
      double beta = nu_d / nu_t;
      double prob_group[MAX_DELAYED_GROUPS] = {0.};
      double prob_used = 0.0;
      for (int g = 1; g <= n_group; ++g) {
        double prob = (g < n_group) ? (*rx.products_[g].yield_)(E_in) / nu_d
                                    : std::max(0.0, 1.0 - prob_used);
        prob_group[g - 1] = beta * prob;
        prob_used += prob;
      }

      // Mixture density of the outgoing energy and its delayed components
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

  int64_t n = simulation::work_per_rank;
  simulation::ifp_birth.resize_banks(n, 3 * n);
  simulation::ifp_lifetime.resize_banks(n, 3 * n);
}

void reset_ifp_streams()
{
  simulation::ifp_birth.reset();
  simulation::ifp_lifetime.reset();
  simulation::ifp_denominator_on = false;
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
