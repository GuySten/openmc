#include <cstdio>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/hdf5_interface.h"
#include "openmc/particle.h"
#include "openmc/photonuclear.h"
#include "openmc/vector.h"

using namespace openmc;
using Catch::Matchers::WithinAbs;

namespace {

void write_reaction(hid_t rxs_group, int mt, bool redundant, int threshold,
  const vector<double>& xs)
{
  hid_t group = create_group(rxs_group, "reaction_" + std::to_string(mt));
  write_attribute(group, "Q_value", 0.0);
  write_attribute(group, "mt", mt);
  write_attribute(group, "center_of_mass", 1);
  write_attribute(group, "redundant", redundant ? 1 : 0);
  write_dataset(group, "xs", xs);
  hid_t dset = open_dataset(group, "xs");
  write_attribute(dset, "threshold_idx", threshold);
  close_dataset(dset);
  close_group(group);
}

} // namespace

// The data below mimic IAEA PD-2019 Ta181, where the lowest grid point lies
// below every reaction threshold and a reaction is already nonzero at its
// threshold point. PhotonuclearReaction::xs() is zero across the interval just
// below a threshold, so the derived total must be as well or reaction sampling
// cannot reach its cutoff there.
TEST_CASE("Photonuclear total matches the sum of reactions near thresholds")
{
  const std::string filename = "test_photonuclear_threshold.h5";
  {
    hid_t file = file_open(filename, 'w');
    hid_t group = create_group(file, "Ta181");
    write_attribute(group, "Z", 73);
    write_attribute(group, "A", 181);
    write_attribute(group, "metastable", 0);
    write_attribute(group, "atomic_weight_ratio", 179.4);
    write_dataset(group, "energy",
      vector<double> {4.658e6, 5.0e6, 6.0e6, 7.0e6, 7.0e6, 8.0e6, 9.0e6});

    hid_t rxs_group = create_group(group, "reactions");
    // Nonzero at its threshold, with a grid interval below it
    write_reaction(rxs_group, 102, false, 1,
      vector<double> {2.855e-3, 3.0e-3, 3.5e-3, 3.5e-3, 4.0e-3, 4.5e-3});
    // Threshold at a later point, again nonzero there
    write_reaction(
      rxs_group, 5, false, 2, vector<double> {0.1, 0.2, 0.2, 0.3, 0.4});
    // Threshold at the second of two coincident grid points
    write_reaction(rxs_group, 16, false, 4, vector<double> {0.05, 0.1, 0.2});
    // Heating is redundant and so excluded from the total
    write_reaction(rxs_group, 301, true, 1,
      vector<double> {1.0e3, 2.0e3, 3.0e3, 4.0e3, 5.0e3, 6.0e3});
    close_group(rxs_group);

    data::photonuclears.push_back(make_unique<PhotonuclearInteraction>(group));
    close_group(group);
    file_close(file);
  }
  std::remove(filename.c_str());

  const auto& nuc = *data::photonuclears.back();
  Particle p;
  const auto& micro = p.photonuclear_xs(nuc.index_);

  for (double E : {4.7e6, 4.898e6, 4.99e6, 5.0e6, 5.5e6, 5.99e6, 6.0e6, 6.5e6,
         6.99e6, 7.0e6, 7.5e6, 8.0e6, 8.5e6, 8.99e6}) {
    p.E() = E;
    nuc.calculate_xs(p);

    double total = 0.0;
    double heating = 0.0;
    for (const auto& rx : nuc.reactions_) {
      if (rx->mt_ == 301)
        heating += rx->xs(micro);
      if (!rx->redundant_)
        total += rx->xs(micro);
    }

    INFO("E = " << E);
    REQUIRE_THAT(micro.total, WithinAbs(total, 1e-15));
    REQUIRE_THAT(micro.heating, WithinAbs(heating, 1e-9));
  }

  // Below the lowest threshold nothing can occur
  p.E() = 4.9e6;
  nuc.calculate_xs(p);
  REQUIRE(micro.total == 0.0);

  data::photonuclears.clear();
}
