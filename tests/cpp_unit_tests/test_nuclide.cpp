#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openmc/atomic_mass.h"
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
    REQUIRE(nuc.awr_ == Approx(atomic_mass(13, 27) / MASS_NEUTRON));
  }

  SECTION("metastable states are recognized")
  {
    Nuclide nuc {"Am242_m1"};
    REQUIRE(nuc.Z_ == 95);
    REQUIRE(nuc.A_ == 242);
    REQUIRE(nuc.metastable_ == 1);

    // The mass table only holds ground states
    REQUIRE(nuc.awr_ == Approx(atomic_mass(95, 242) / MASS_NEUTRON));
  }

  SECTION("H1 gets the atomic mass rather than the bare proton mass")
  {
    Nuclide nuc {"H1"};
    REQUIRE(nuc.Z_ == 1);
    REQUIRE(nuc.A_ == 1);
    REQUIRE(nuc.awr_ == Approx(atomic_mass(1, 1) / MASS_NEUTRON));
    REQUIRE(nuc.awr_ > MASS_PROTON / MASS_NEUTRON);
  }

  SECTION("no cross section data is present")
  {
    Nuclide nuc {"U235"};
    REQUIRE(nuc.grid_.empty());
    REQUIRE(nuc.xs_.empty());
    REQUIRE(nuc.reactions_.empty());
    REQUIRE_FALSE(nuc.fissionable_);
  }
}
