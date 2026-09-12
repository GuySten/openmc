#include <memory>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <pugixml.hpp>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/geometry.h"
#include "openmc/surface.h"
#include "openmc/tallies/filter_point.h"
#include "openmc/universe.h"

using namespace openmc;

namespace {

//! A root cell bounded by a sphere of radius 3, filled by a universe split in
//! two by a sphere of radius 1. Nesting is deliberate: the bounding surface of
//! each coordinate level has to be measured against, not just the innermost
//! cell's, and anything beyond r = 3 is outside the model.
class SphereFixture {
public:
  //! \param outer material outside r = 1
  //! \param inclusion add a small bounded cell of material 9 at x = 2,
  //!   far enough out that a small sphere cannot reach its bounding box
  explicit SphereFixture(int32_t outer = 9, bool inclusion = false)
    : root_universe_ {model::root_universe},
      n_coord_levels_ {model::n_coord_levels}
  {
    clear();
    model::n_coord_levels = 2;
    model::root_universe = 0;

    add_sphere(1, "0 0 0 1");
    add_sphere(2, "0 0 0 3");
    if (inclusion)
      add_sphere(3, "2 0 0 0.3");

    // materials: 7 inside r = 1, and either 7 or 9 outside, per the test
    model::cells.push_back(make_cell(1, 0, 1, "-2"));     // root, r < 3
    model::cells.push_back(make_cell(2, 1, -1, "-1", 7)); // inside r = 1
    model::cells.push_back(
      make_cell(3, 1, -1, inclusion ? "+1 +3" : "+1", outer)); // outside r = 1
    if (inclusion)
      model::cells.push_back(make_cell(4, 1, -1, "-3", 9)); // island at x = 2
    for (int i = 0; i < model::cells.size(); ++i)
      model::cell_map[model::cells[i]->id_] = i;

    add_universe(0, {0});
    if (inclusion)
      add_universe(1, {1, 2, 3});
    else
      add_universe(1, {1, 2});
  }

  ~SphereFixture()
  {
    clear();
    model::root_universe = root_universe_;
    model::n_coord_levels = n_coord_levels_;
  }

private:
  static void clear()
  {
    model::cells.clear();
    model::cell_map.clear();
    model::universes.clear();
    model::universe_map.clear();
    model::surfaces.clear();
    model::surface_map.clear();
  }

  static void add_sphere(int id, const char* coeffs)
  {
    pugi::xml_document doc;
    auto node = doc.append_child("surface");
    node.append_attribute("id") = id;
    node.append_attribute("type") = "sphere";
    node.append_attribute("coeffs") = coeffs;
    model::surfaces.push_back(std::make_unique<SurfaceSphere>(node));
    model::surface_map[id] = model::surfaces.size() - 1;
  }

  static void add_universe(int id, std::vector<int32_t> cells)
  {
    auto u = std::make_unique<Universe>();
    u->id_ = id;
    u->cells_ = std::move(cells);
    u->n_instances_ = 1;
    model::universes.push_back(std::move(u));
    model::universe_map[id] = model::universes.size() - 1;
  }

  static std::unique_ptr<CSGCell> make_cell(
    int id, int universe, int fill, const char* region, int32_t material = -1)
  {
    pugi::xml_document doc;
    auto node = doc.append_child("cell");
    node.append_attribute("id") = id;
    node.append_attribute("universe") = universe;
    if (fill >= 0) {
      const auto fill_value {std::to_string(fill)};
      node.append_child("fill").text() = fill_value.c_str();
    } else {
      node.append_child("material").text() = "void";
    }
    if (region[0] != '\0')
      node.append_child("region").text() = region;

    auto cell = std::make_unique<CSGCell>(node);
    if (fill < 0) {
      cell->type_ = Fill::MATERIAL;
      cell->material_ = {material};
      cell->sqrtkT_.push_back(0.0);
      cell->density_mult_.push_back(1.0);
    } else {
      cell->type_ = Fill::UNIVERSE;
    }
    return cell;
  }

  int root_universe_;
  int n_coord_levels_;
};

} // anonymous namespace

TEST_CASE_METHOD(
  SphereFixture, "Exclusion sphere material uniformity is decided exactly")
{
  double nearest = INFTY;

  SECTION("a sphere well inside its cell is confined")
  {
    // at the centre: r = 1 is 1.0 away, the root cell's r = 3 is 3.0 away
    REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 0.5, nearest) ==
            SphereCheck::SINGLE_MATERIAL);
    REQUIRE(nearest == Catch::Approx(1.0));
  }

  SECTION("a sphere reaching a different material is reported")
  {
    REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 2.0, nearest) ==
            SphereCheck::MULTIPLE_MATERIALS);
    REQUIRE(nearest == Catch::Approx(1.0));
  }

  SECTION("the decision turns exactly on the radius")
  {
    // off-centre: nearest boundary is r = 1 at a distance of 0.4
    Position p {0.6, 0.0, 0.0};
    REQUIRE(check_exclusion_sphere(p, 0.399, nearest) ==
            SphereCheck::SINGLE_MATERIAL);
    REQUIRE(nearest == Catch::Approx(0.4));
    REQUIRE(check_exclusion_sphere(p, 0.401, nearest) ==
            SphereCheck::MULTIPLE_MATERIALS);
  }

  SECTION("surfaces from every coordinate level are measured")
  {
    // at r = 1.5 the inner sphere is 0.5 away and the root cell's outer
    // sphere, one level up, is 1.5 away -- the inner one governs
    REQUIRE(check_exclusion_sphere({1.5, 0.0, 0.0}, 0.4, nearest) ==
            SphereCheck::SINGLE_MATERIAL);
    REQUIRE(nearest == Catch::Approx(0.5));

    // at r = 2.8 the level-up surface is the closer of the two at 0.2, even
    // though the level the detector sits in has nothing nearer than 1.8. A
    // sphere of 0.4 can therefore leave the innermost universe, past which
    // its cells no longer bound where the sphere reaches
    REQUIRE(check_exclusion_sphere({2.8, 0.0, 0.0}, 0.4, nearest) ==
            SphereCheck::UNDECIDABLE);
    REQUIRE(nearest == Catch::Approx(0.2));
  }

  SECTION("a detector outside the model is reported, not fatal")
  {
    REQUIRE(check_exclusion_sphere({0.0, 0.0, 50.0}, 0.1, nearest) ==
            SphereCheck::OUTSIDE_MODEL);
  }
}

TEST_CASE_METHOD(SphereFixture,
  "A sphere crossing into the same material is not flagged", "[same]")
{
  // This fixture is the default: material 9 outside r = 1, differing from the
  // 7 inside, so crossing the boundary really does mix materials
  double nearest = INFTY;
  REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 2.0, nearest) ==
          SphereCheck::MULTIPLE_MATERIALS);
  REQUIRE(nearest == Catch::Approx(1.0));
}

TEST_CASE("A cell boundary within one material does not matter")
{
  // Same geometry, but the surface at r = 1 now merely subdivides a single
  // material. The sphere still leaves its cell, so a cell-based test would
  // flag it; a material-based one must not.
  SphereFixture fixture {7};
  double nearest = INFTY;
  REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 2.0, nearest) ==
          SphereCheck::SINGLE_MATERIAL);
  REQUIRE(nearest == Catch::Approx(1.0));

  // and one that does not even reach the boundary is still fine
  REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 0.5, nearest) ==
          SphereCheck::SINGLE_MATERIAL);
}

TEST_CASE("Cells out of the sphere's reach are not counted against it")
{
  // Material 7 everywhere nearby, plus a small island of material 9 centred
  // at x = 2 with radius 0.3, so its bounding box starts 1.7 from the origin.
  SphereFixture fixture {7, true};
  double nearest = INFTY;

  // A sphere of 1.5 leaves its cell at r = 1 but cannot reach the island, so
  // every cell it can touch is material 7
  REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 1.5, nearest) ==
          SphereCheck::SINGLE_MATERIAL);
  REQUIRE(nearest == Catch::Approx(1.0));

  // A sphere of 2.0 does reach the island's bounding box, so the answer
  // changes -- the two differ only in whether that box is within reach
  REQUIRE(check_exclusion_sphere({0.0, 0.0, 0.0}, 2.0, nearest) ==
          SphereCheck::MULTIPLE_MATERIALS);
}
