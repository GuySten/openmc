#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/hdf5_interface.h"
#include "openmc/search.h"

using namespace openmc;

// Simple log-log interpolation for cross sections
double interpolate_xs(const std::vector<double>& energy,
                      const std::vector<double>& xs,
                      double E)
{
  // Handle exact matches and boundary cases
  if (E <= energy.front()) return xs.front();
  if (E >= energy.back()) return xs.back();

  // Find the right interval: energy[i-1] <= E < energy[i]
  size_t i = 1;
  while (i < energy.size() && energy[i] < E) {
    i++;
  }

  // Check for exact match
  if (std::abs(energy[i] - E) < 1e-10 * E) {
    return xs[i];
  }
  if (std::abs(energy[i-1] - E) < 1e-10 * E) {
    return xs[i-1];
  }

  double E1 = energy[i-1];
  double E2 = energy[i];
  double xs1 = xs[i-1];
  double xs2 = xs[i];

  // Log-log interpolation (handles Gamow-like falloff better)
  if (xs1 > 0 && xs2 > 0) {
    double f = std::log(E/E1) / std::log(E2/E1);
    return xs1 * std::exp(f * std::log(xs2/xs1));
  } else {
    // Linear interpolation fallback for zero values
    double f = (E - E1) / (E2 - E1);
    return xs1 + f * (xs2 - xs1);
  }
}

TEST_CASE("Test Deuteron HDF5 Data")
{
  // Path to test data file
  std::string filename = "data/deuteron_H2.h5";

  // Open HDF5 file
  hid_t file_id = file_open(filename, 'r');
  REQUIRE(file_id >= 0);

  // Open the H2 group
  hid_t group_id = open_group(file_id, "H2");
  REQUIRE(group_id >= 0);

  // Read attributes
  int Z, A, metastable;
  double awr;
  read_attribute(group_id, "Z", Z);
  read_attribute(group_id, "A", A);
  read_attribute(group_id, "metastable", metastable);
  read_attribute(group_id, "atomic_weight_ratio", awr);

  std::cout << "\n=== Deuteron-on-H2 Data ===" << std::endl;
  std::cout << "Target: Z=" << Z << ", A=" << A << ", AWR=" << awr << std::endl;

  REQUIRE(Z == 1);
  REQUIRE(A == 2);
  REQUIRE(metastable == 0);
  REQUIRE_THAT(awr, Catch::Matchers::WithinRel(1.996256, 1e-4));

  // Read energy grid
  std::vector<double> energy;
  read_dataset(group_id, "energy", energy);
  std::cout << "Energy grid: " << energy.size() << " points, "
            << energy.front() << " to " << energy.back() << " eV" << std::endl;

  // Read MT numbers
  std::vector<int> mt_list;
  read_dataset(group_id, "MT", mt_list);
  std::cout << "Reactions: ";
  for (int mt : mt_list) std::cout << mt << " ";
  std::cout << std::endl;

  REQUIRE(mt_list.size() == 3);
  REQUIRE(mt_list[0] == 2);    // elastic
  REQUIRE(mt_list[1] == 50);   // d,d -> n + He3
  REQUIRE(mt_list[2] == 600);  // d,d -> p + t

  // Read Q-values
  std::vector<double> q_values;
  read_dataset(group_id, "Q_value", q_values);
  std::cout << "Q-values (MeV): ";
  for (double q : q_values) std::cout << q/1e6 << " ";
  std::cout << std::endl;

  REQUIRE_THAT(q_values[0], Catch::Matchers::WithinAbs(0.0, 1e-6));  // elastic
  REQUIRE_THAT(q_values[1]/1e6, Catch::Matchers::WithinRel(3.269, 0.01));  // n+He3
  REQUIRE_THAT(q_values[2]/1e6, Catch::Matchers::WithinRel(4.033, 0.01));  // p+t

  // Read cross section matrix
  xt::xtensor<double, 2> xs_matrix;
  read_dataset(group_id, "xs", xs_matrix);
  std::cout << "XS matrix shape: " << xs_matrix.shape()[0] << " x "
            << xs_matrix.shape()[1] << std::endl;

  // Extract columns for each reaction
  std::vector<double> xs_elastic(energy.size());
  std::vector<double> xs_n_he3(energy.size());
  std::vector<double> xs_p_t(energy.size());

  for (size_t i = 0; i < energy.size(); i++) {
    xs_elastic[i] = xs_matrix(i, 0);
    xs_n_he3[i] = xs_matrix(i, 1);
    xs_p_t[i] = xs_matrix(i, 2);
  }

  // Test cross sections at specific energies
  std::cout << "\n=== Cross Section Values ===" << std::endl;

  // Test energies: 10 keV, 100 keV, 1 MeV
  std::vector<double> test_E = {1e4, 1e5, 1e6};

  // Expected values from Python (approximate)
  std::vector<double> expected_elastic = {1.0, 1.0, 1.0};
  std::vector<double> expected_n_he3 = {8.82e-6, 1.63e-2, 9.36e-2};
  std::vector<double> expected_p_t = {9.02e-6, 1.54e-2, 7.75e-2};

  std::cout << "\nElastic (MT=2):" << std::endl;
  for (size_t i = 0; i < test_E.size(); i++) {
    double xs = interpolate_xs(energy, xs_elastic, test_E[i]);
    std::cout << "  E=" << test_E[i]/1e6 << " MeV: " << xs << " b"
              << " (expected ~" << expected_elastic[i] << ")" << std::endl;
    REQUIRE_THAT(xs, Catch::Matchers::WithinRel(expected_elastic[i], 0.01));
  }

  std::cout << "\nd+d -> n + He3 (MT=50):" << std::endl;
  for (size_t i = 0; i < test_E.size(); i++) {
    double xs = interpolate_xs(energy, xs_n_he3, test_E[i]);
    std::cout << "  E=" << test_E[i]/1e6 << " MeV: " << xs << " b"
              << " (expected ~" << expected_n_he3[i] << ")" << std::endl;
    REQUIRE_THAT(xs, Catch::Matchers::WithinRel(expected_n_he3[i], 0.05));
  }

  std::cout << "\nd+d -> p + t (MT=600):" << std::endl;
  for (size_t i = 0; i < test_E.size(); i++) {
    double xs = interpolate_xs(energy, xs_p_t, test_E[i]);
    std::cout << "  E=" << test_E[i]/1e6 << " MeV: " << xs << " b"
              << " (expected ~" << expected_p_t[i] << ")" << std::endl;
    REQUIRE_THAT(xs, Catch::Matchers::WithinRel(expected_p_t[i], 0.05));
  }

  // Verify fusion cross section ratio (should be ~50/50 at high energy)
  double ratio = interpolate_xs(energy, xs_n_he3, 1e6) /
                 interpolate_xs(energy, xs_p_t, 1e6);
  std::cout << "\nFusion branch ratio (n+He3)/(p+t) at 1 MeV: " << ratio << std::endl;
  REQUIRE_THAT(ratio, Catch::Matchers::WithinRel(1.2, 0.1));  // ~1.2 at 1 MeV

  close_group(group_id);
  file_close(file_id);

  std::cout << "\n=== All tests passed! ===" << std::endl;
}
