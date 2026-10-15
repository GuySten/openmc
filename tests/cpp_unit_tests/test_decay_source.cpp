#include "openmc/chain.h"
#include "openmc/decay_source.h"
#include "openmc/random_lcg.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <pugixml.hpp>

using Catch::Matchers::WithinRel;

namespace {

// Decay chain with branching, two nuclides sharing a half-life, and a stable
// end product:
//
//   A (100 s) --0.6--> B (250 s) --> D (stable)
//             --0.4--> C (100 s) --> D (stable)
const char* CHAIN_XML = R"(
<depletion_chain>
  <nuclide name="A" half_life="100.0" decay_modes="2" reactions="0">
    <decay type="beta-" target="B" branching_ratio="0.6"/>
    <decay type="alpha" target="C" branching_ratio="0.4"/>
  </nuclide>
  <nuclide name="B" half_life="250.0" decay_modes="1" reactions="0">
    <decay type="beta-" target="D" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="C" half_life="100.0" decay_modes="1" reactions="0">
    <decay type="beta-" target="D" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="E" half_life="1.0e12" decay_modes="1" reactions="0">
    <decay type="beta-" target="F" branching_ratio="1.0"/>
    <source type="discrete" particle="electron">
      <parameters>100000.0 300000.0 0.5e-12 0.25e-12</parameters>
    </source>
  </nuclide>
  <nuclide name="F" half_life="10.0" decay_modes="1" reactions="0">
    <decay type="beta-" target="D" branching_ratio="1.0"/>
  </nuclide>
  <nuclide name="D" decay_modes="0" reactions="0">
    <source type="discrete" particle="neutron">
      <parameters>500000.0 1.0</parameters>
    </source>
  </nuclide>
</depletion_chain>
)";

void load_chain()
{
  openmc::free_memory_chain();
  pugi::xml_document doc;
  doc.load_string(CHAIN_XML);
  for (auto node : doc.document_element().children("nuclide")) {
    openmc::data::chain_nuclides.push_back(
      std::make_unique<openmc::ChainNuclide>(node));
    openmc::data::chain_nuclides.back()->read_decay_source_data(node);
  }
}

int index(const std::string& name)
{
  return openmc::data::chain_nuclide_map.at(name);
}

const double LAMBDA_100 = std::log(2.0) / 100.0;
const double LAMBDA_250 = std::log(2.0) / 250.0;

} // namespace

TEST_CASE("Decay inventory of a single nuclide at constant power")
{
  load_chain();
  openmc::DecaySourceSettings s;
  s.timesteps = {300.0};
  s.power = {1.0};

  auto inventory = openmc::decay_inventory(index("B"), s);
  double expected = -std::expm1(-LAMBDA_250 * 300.0) / LAMBDA_250;
  REQUIRE_THAT(inventory.at(index("B")), WithinRel(expected, 1e-12));

  // Stable nuclides do not emit and are not part of the inventory
  REQUIRE(inventory.count(index("D")) == 0);
  openmc::free_memory_chain();
}

TEST_CASE("Decay inventory with branching and equal decay constants")
{
  load_chain();
  openmc::DecaySourceSettings s;
  s.timesteps = {400.0};
  s.power = {1.0};
  const double T = 400.0;
  const double la = LAMBDA_100;
  const double lb = LAMBDA_250;

  auto inventory = openmc::decay_inventory(index("A"), s);

  // Parent at constant production rate
  double n_a = -std::expm1(-la * T) / la;
  REQUIRE_THAT(inventory.at(index("A")), WithinRel(n_a, 1e-12));

  // Two-member Bateman solution integrated over the irradiation
  double n_b = 0.6 * la / (lb - la) *
               (-std::expm1(-la * T) / la - -std::expm1(-lb * T) / lb);
  REQUIRE_THAT(inventory.at(index("B")), WithinRel(n_b, 1e-10));

  // Daughter with the same decay constant as its parent, for which the
  // Bateman solution is singular. The exact solution is the integral of
  // lambda*t*exp(-lambda*t).
  double n_c = 0.4 * (1.0 - std::exp(-la * T) * (1.0 + la * T)) / la;
  REQUIRE_THAT(inventory.at(index("C")), WithinRel(n_c, 1e-7));
  openmc::free_memory_chain();
}

TEST_CASE("Decay inventory with a power history")
{
  load_chain();
  // Irradiation, shutdown, and operation at twice the initial power. The
  // inventory is normalized to the final power.
  openmc::DecaySourceSettings s;
  s.timesteps = {200.0, 150.0, 50.0};
  s.power = {1.0, 0.0, 2.0};

  auto inventory = openmc::decay_inventory(index("B"), s);
  double lb = LAMBDA_250;
  double expected = (0.5 * -std::expm1(-lb * 200.0) * std::exp(-lb * 200.0) +
                      -std::expm1(-lb * 50.0)) /
                    lb;
  REQUIRE_THAT(inventory.at(index("B")), WithinRel(expected, 1e-12));
  openmc::free_memory_chain();
}

TEST_CASE("Decay inventory at saturation")
{
  load_chain();
  // After many half-lives, the decay rate of each nuclide equals its
  // production rate
  openmc::DecaySourceSettings s;
  s.timesteps = {1.0e6};
  s.power = {1.0};

  auto inventory = openmc::decay_inventory(index("A"), s);
  REQUIRE_THAT(LAMBDA_100 * inventory.at(index("A")), WithinRel(1.0, 1e-12));
  REQUIRE_THAT(LAMBDA_250 * inventory.at(index("B")), WithinRel(0.6, 1e-12));
  REQUIRE_THAT(LAMBDA_100 * inventory.at(index("C")), WithinRel(0.4, 1e-7));
  openmc::free_memory_chain();
}

TEST_CASE("Decay sources never emit neutrons")
{
  load_chain();
  // Delayed neutrons come from the fission data, so neutron decay sources in
  // the chain are not used
  const auto& d = *openmc::data::chain_nuclides[index("D")];
  REQUIRE(d.source(openmc::ParticleType::neutron()) == nullptr);
  openmc::free_memory_chain();
}

namespace {

// Mean of the delays sampled for the trajectory ending at a nuclide
double mean_delay(const std::vector<openmc::DecayTrajectory>& trajectories,
  int nuclide, const openmc::DecayHistory& h, double* min_delay = nullptr,
  double* max_delay = nullptr)
{
  constexpr int n = 200000;
  uint64_t seed = openmc::init_seed(1, 0);
  for (const auto& traj : trajectories) {
    if (traj.nuclide != nuclide)
      continue;
    double sum = 0.0;
    double lo = 1e300;
    double hi = -1e300;
    for (int i = 0; i < n; ++i) {
      double t = openmc::sample_decay_delay(traj, h, &seed);
      sum += t;
      lo = std::min(lo, t);
      hi = std::max(hi, t);
    }
    if (min_delay)
      *min_delay = lo;
    if (max_delay)
      *max_delay = hi;
    return sum / n;
  }
  return -1.0;
}

} // namespace

TEST_CASE("Delay of decay particles")
{
  load_chain();
  openmc::DecaySourceSettings s;
  s.timesteps = {300.0};
  s.power = {1.0};
  openmc::DecayHistory h(s);
  const double T = 300.0;

  // Single nuclide: exponential distribution truncated at the start of the
  // irradiation
  auto traj_b = openmc::find_decay_trajectories(index("B"), h);
  double lb = LAMBDA_250;
  double expected = 1.0 / lb - T * std::exp(-lb * T) / -std::expm1(-lb * T);
  double lo, hi;
  double mean = mean_delay(traj_b, index("B"), h, &lo, &hi);
  REQUIRE_THAT(mean, WithinRel(expected, 0.01));
  REQUIRE(lo >= 0.0);
  REQUIRE(hi <= T);

  // Daughter: the delay follows the Bateman response of the parent-daughter
  // pair, whose mean is computed by quadrature
  auto traj_a = openmc::find_decay_trajectories(index("A"), h);
  double la = LAMBDA_100;
  double num = 0.0;
  double den = 0.0;
  constexpr int n = 100000;
  for (int i = 0; i < n; ++i) {
    double t = (i + 0.5) * T / n;
    double k = (std::exp(-la * t) - std::exp(-lb * t)) / (lb - la);
    num += t * k;
    den += k;
  }
  mean = mean_delay(traj_a, index("B"), h);
  REQUIRE_THAT(mean, WithinRel(num / den, 0.01));
  openmc::free_memory_chain();
}

TEST_CASE("Delay of decay particles with a shutdown")
{
  load_chain();
  // Atoms decaying at the end of the history were produced either in the last
  // step or before the shutdown, never during it
  openmc::DecaySourceSettings s;
  s.timesteps = {200.0, 150.0, 50.0};
  s.power = {1.0, 0.0, 2.0};
  openmc::DecayHistory h(s);
  auto traj = openmc::find_decay_trajectories(index("A"), h);

  uint64_t seed = openmc::init_seed(2, 0);
  int n_before = 0;
  int n_during = 0;
  for (int i = 0; i < 100000; ++i) {
    double t = openmc::sample_decay_delay(traj[0], h, &seed);
    if (t > 50.0 && t < 200.0)
      ++n_during;
    if (t >= 200.0)
      ++n_before;
  }
  REQUIRE(n_during == 0);

  // Fraction produced before the shutdown
  double la = LAMBDA_100;
  double before = 0.5 * std::exp(-la * 200.0) * -std::expm1(-la * 200.0);
  double last = -std::expm1(-la * 50.0);
  REQUIRE_THAT(n_before / 1.0e5, WithinRel(before / (before + last), 0.05));
  openmc::free_memory_chain();
}

TEST_CASE("Decay inventory with an infinitely long irradiation")
{
  load_chain();
  // Default settings correspond to an infinitely long irradiation at constant
  // power, where each nuclide decays as fast as it is produced
  openmc::DecaySourceSettings s;
  openmc::DecayHistory h(s);
  REQUIRE(h.infinite);

  auto inventory = openmc::decay_inventory(index("A"), s);
  REQUIRE_THAT(LAMBDA_100 * inventory.at(index("A")), WithinRel(1.0, 1e-12));
  REQUIRE_THAT(LAMBDA_250 * inventory.at(index("B")), WithinRel(0.6, 1e-12));
  REQUIRE_THAT(LAMBDA_100 * inventory.at(index("C")), WithinRel(0.4, 1e-7));

  // The delay is the sum of the times spent in each nuclide
  auto trajectories = openmc::find_decay_trajectories(index("A"), h);
  double mean = mean_delay(trajectories, index("B"), h);
  REQUIRE_THAT(mean, WithinRel(1.0 / LAMBDA_100 + 1.0 / LAMBDA_250, 0.01));
  openmc::free_memory_chain();
}

TEST_CASE("Long-lived nuclides are treated as stable")
{
  load_chain();
  // E has a half-life above the default cutoff of 1000 years, so neither it
  // nor its short-lived daughter F decays during operation
  openmc::DecaySourceSettings s;
  REQUIRE(openmc::decay_inventory(index("E"), s).empty());

  // With a longer cutoff, the infinitely long irradiation brings both into
  // equilibrium
  s.half_life_cutoff = 1.0e13;
  auto inventory = openmc::decay_inventory(index("E"), s);
  REQUIRE_THAT(
    inventory.at(index("E")) * std::log(2.0) / 1.0e12, WithinRel(1.0, 1e-12));
  REQUIRE_THAT(
    inventory.at(index("F")) * std::log(2.0) / 10.0, WithinRel(1.0, 1e-7));
  openmc::free_memory_chain();
}

TEST_CASE("Energy emission rate of decay sources")
{
  load_chain();
  const auto& e = *openmc::data::chain_nuclides[index("E")];
  REQUIRE_THAT(e.source_energy(openmc::ParticleType::electron()),
    WithinRel(100000.0 * 0.5e-12 + 300000.0 * 0.25e-12, 1e-12));
  REQUIRE(e.source_energy(openmc::ParticleType::photon()) == 0.0);
  openmc::free_memory_chain();
}
