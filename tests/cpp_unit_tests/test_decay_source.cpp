#include "openmc/chain.h"
#include "openmc/decay_source.h"
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
  <nuclide name="D" decay_modes="0" reactions="0"/>
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
