#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

#include "openmc/atomic_mass.h"
#include "openmc/constants.h"
#include "openmc/nuclide.h"

using Catch::Approx;

TEST_CASE("Nuclide constructed without cross section data")
{
  using namespace openmc;

  SECTION("identity and mass come from the nuclide name")
  {
    Nuclide nuc {"Al27"};
    REQUIRE(nuc.name_ == "Al27");
    REQUIRE(nuc.Z_ == 13);
    REQUIRE(nuc.A_ == 27);
    REQUIRE(nuc.metastable_ == 0);

    // Compared against the atomic weight ratio an ENDF/B-VII.1 neutron data
    // file carries, rather than against the table this is derived from, so
    // that the substitution itself is what is being checked
    REQUIRE(nuc.awr_ == Approx(26.7497).epsilon(1.0e-5));
  }

  SECTION("metastable states are recognized")
  {
    Nuclide nuc {"Am242_m1"};
    REQUIRE(nuc.Z_ == 95);
    REQUIRE(nuc.A_ == 242);
    REQUIRE(nuc.metastable_ == 1);

    // The mass table only holds ground states, and the ~48 keV excitation of
    // Am242m1 is some 2e-7 of its mass, far below the tolerance here
    REQUIRE(nuc.awr_ == Approx(239.9801).epsilon(1.0e-5));
  }

  SECTION("H1 gets the atomic mass rather than the bare proton mass")
  {
    Nuclide nuc {"H1"};
    REQUIRE(nuc.Z_ == 1);
    REQUIRE(nuc.A_ == 1);
    REQUIRE(nuc.awr_ == Approx(0.999167).epsilon(1.0e-5));
    REQUIRE(nuc.awr_ > MASS_PROTON / MASS_NEUTRON);
  }

  SECTION("no cross section data is present")
  {
    Nuclide nuc {"U235"};
    REQUIRE(nuc.awr_ == Approx(233.0248).epsilon(1.0e-5));
    REQUIRE(nuc.grid_.empty());
    REQUIRE(nuc.xs_.empty());
    REQUIRE(nuc.reactions_.empty());
    REQUIRE(nuc.kTs_.empty());
    REQUIRE_FALSE(nuc.fissionable_);

    // Every reaction lookup has to miss, since there are no reactions to find.
    // Left uninitialized these are garbage that passes the >= 0 test callers
    // use before indexing the empty reactions_ vector.
    for (auto index : nuc.reaction_index_) {
      REQUIRE(static_cast<int>(index) == C_NONE);
    }
  }

  SECTION("an elemental evaluation carries the natural atomic weight")
  {
    // Some libraries provide C0 in place of C12 and C13
    Nuclide nuc {"C0"};
    REQUIRE(nuc.Z_ == 6);
    REQUIRE(nuc.A_ == 0);

    // Natural carbon, not C12
    REQUIRE(nuc.awr_ == Approx(11.9079).epsilon(1.0e-5));
    REQUIRE(nuc.awr_ > atomic_mass(6, 12) / MASS_NEUTRON);
  }

  SECTION("names that carry no usable mass are rejected")
  {
    REQUIRE_THROWS_AS(Nuclide {"not a nuclide"}, std::runtime_error);

    // Tc has no naturally occurring isotopes, so no elemental weight either
    REQUIRE_THROWS_AS(Nuclide {"Tc0"}, std::runtime_error);

    // A number too large for the PDG parser is reported, not thrown past
    REQUIRE_THROWS_AS(Nuclide {"99999999999"}, std::runtime_error);

    // A rejected name leaves nothing registered behind
    REQUIRE(data::nuclide_map.find("Tc0") == data::nuclide_map.end());
  }

  SECTION("only canonical nuclide names are accepted")
  {
    // A data library names nuclides in GNDS form. Particle aliases and bare
    // PDG numbers are not nuclide names, and accepting them here would make a
    // model run only when neutron data is being skipped.
    for (const auto* name : {"alpha", "h1", "proton", "d", "t", "he4",
           "1000130270", "pdg:1000130270"}) {
      REQUIRE_THROWS_AS(Nuclide {name}, std::runtime_error);
    }

    // The canonical spellings of those same nuclides are fine
    REQUIRE_NOTHROW(Nuclide {"He4"});
    REQUIRE_NOTHROW(Nuclide {"H1"});
    REQUIRE_NOTHROW(Nuclide {"Al27"});
  }
}
