#include "openmc/ifp.h"

#include <algorithm> // for fill, max, min

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

IFPStream<int> ifp_delayed_group;
IFPStream<double> ifp_lifetime;
IFPStream<IFPBirth> ifp_birth;

} // namespace simulation

namespace {

//! Incremented whenever the streams are reset, so that cached probabilities
//! never outlive the nuclear data they were computed from
int ifp_cache_epoch {0};

} // namespace

void ifp(const Particle& p, int64_t idx, const IFPBirth& birth)
{
  simulation::ifp_delayed_group.store(p.current_work(), idx, p.delayed_group());
  simulation::ifp_lifetime.store(p.current_work(), idx, p.lifetime());

  auto& stream = simulation::ifp_birth;
  if (stream.enabled()) {
    // Each list ends with the site's own birth, which is only known here. The
    // list of a neutron that was not born from fission is empty; it gets a
    // record saying so, as the delayed group stream records group 0 for it.
    const auto& parent = stream.source_bank()[p.current_work()];
    auto& updated = stream.fission_bank()[idx];
    if (parent.empty()) {
      updated.assign(1, IFPBirth {});
    } else {
      updated = parent;
    }
    updated.push_back(birth);
    if (updated.size() > settings::ifp_n_generation + 1) {
      updated.erase(updated.begin());
    }
  }
}

int ifp_ancestor_delayed_group(int64_t i_source)
{
  if (simulation::ifp_delayed_group.enabled()) {
    const auto& groups = simulation::ifp_delayed_group.source_bank()[i_source];
    return groups.size() == settings::ifp_n_generation ? groups[0] : -1;
  }
  if (simulation::ifp_birth.enabled()) {
    const auto& births = simulation::ifp_birth.source_bank()[i_source];
    return births.size() == settings::ifp_n_generation + 1
             ? births[0].delayed_group()
             : -1;
  }
  return -1;
}

bool ifp_history_complete(int64_t i_source)
{
  if (simulation::ifp_delayed_group.enabled() ||
      simulation::ifp_birth.enabled()) {
    return ifp_ancestor_delayed_group(i_source) >= 0;
  }
  return simulation::ifp_lifetime.source_bank()[i_source].size() ==
         settings::ifp_n_generation;
}

bool ifp_ancestor_dg_weights(const Particle& p, double* weights)
{
  const auto& births = simulation::ifp_birth.source_bank()[p.current_work()];
  if (births.size() != settings::ifp_n_generation + 1)
    return false;

  // Several fissions and tallies of the same history ask for the same ancestor
  thread_local IFPBirth cached_birth;
  thread_local int cached_epoch {-1};
  thread_local double cached_weights[MAX_DELAYED_GROUPS];

  const IFPBirth& ancestor = births[0];
  if (cached_epoch != ifp_cache_epoch || !(ancestor == cached_birth)) {
    if (ancestor.from_fission()) {
      const auto& rx =
        *data::nuclides[ancestor.nuclide()]->fission_rx_[ancestor.reaction()];
      ifp_delayed_group_weights(ancestor.nuclide(), rx, ancestor.E_in(),
        ancestor.E_out(), ancestor.delayed_group(), cached_weights);
    } else {
      std::fill(cached_weights, cached_weights + MAX_DELAYED_GROUPS, 0.0);
    }
    cached_birth = ancestor;
    cached_epoch = ifp_cache_epoch;
  }
  std::copy(cached_weights, cached_weights + MAX_DELAYED_GROUPS, weights);
  return true;
}

void ifp_delayed_group_weights(int i_nuclide, const Reaction& rx, double E_in,
  double E_out, int delayed_group, double* weights)
{
  std::fill(weights, weights + MAX_DELAYED_GROUPS, 0.0);

  // Delayed emission probabilities, following sample_fission_neutron(): the
  // neutron is delayed with probability nu_d / nu_t, and the group is then
  // chosen from the cumulative yields, the last group taking any remainder
  const auto& nuc {data::nuclides[i_nuclide]};
  int n_group = std::min(nuc->n_precursor_, MAX_DELAYED_GROUPS);
  double nu_t = nuc->nu(E_in, Nuclide::EmissionMode::total);
  double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed);
  if (n_group == 0 || nu_d <= 0.0 || nu_t <= 0.0)
    return;
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
      fatal_error(fmt::format("The outgoing energy density of fission product "
                              "{} of {} cannot be evaluated, which the "
                              "ifp-beta-numerator-cond and "
                              "ifp-beta-numerator-group scores require.",
        i_product, nuc->name_));
    }
    return f;
  };

  double total = (1.0 - beta) * density(0);
  for (int g = 1; g <= n_group; ++g) {
    weights[g - 1] = prob_group[g - 1] * density(g);
    total += weights[g - 1];
  }

  if (total > 0.0) {
    for (int g = 0; g < n_group; ++g)
      weights[g] /= total;
  } else {
    // The tabulated density vanishes where a neutron was nonetheless sampled
    // (e.g. exactly at a table edge); fall back to the analog indicator
    std::fill(weights, weights + MAX_DELAYED_GROUPS, 0.0);
    if (delayed_group > 0)
      weights[delayed_group - 1] = 1.0;
  }
}

void resize_simulation_ifp_banks()
{
  int64_t n = simulation::work_per_rank;
  simulation::ifp_delayed_group.resize_banks(n, 3 * n);
  simulation::ifp_lifetime.resize_banks(n, 3 * n);
  simulation::ifp_birth.resize_banks(n, 3 * n);

  // The birth records also hold the sampled delayed group, which makes the
  // delayed group stream redundant
  if (simulation::ifp_birth.enabled())
    simulation::ifp_delayed_group.reset();
}

void reset_ifp_streams()
{
  simulation::ifp_delayed_group.reset();
  simulation::ifp_lifetime.reset();
  simulation::ifp_birth.reset();
  ++ifp_cache_epoch;
}

#ifdef OPENMC_MPI
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes)
{
  if (mpi::rank == 0) {
    if (simulation::ifp_delayed_group.enabled()) {
      n_generation = static_cast<int>(delayed_groups[0].size());
    } else if (simulation::ifp_lifetime.enabled()) {
      n_generation = static_cast<int>(lifetimes[0].size());
    } else {
      // Only birth records, which are not yet sent between processes
      n_generation = 0;
    }
  }
  MPI_Bcast(&n_generation, 1, MPI_INT, 0, mpi::intracomm);
}
#endif

} // namespace openmc
