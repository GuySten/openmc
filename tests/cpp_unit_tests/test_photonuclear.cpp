#include <cmath>
#include <cstdio>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/particle.h"
#include "openmc/photonuclear.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/secondary_nbody.h"
#include "openmc/tensor.h"
#include "openmc/vector.h"

using namespace openmc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

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
         6.99e6, 7.0e6, 7.5e6, 8.0e6, 8.5e6, 8.99e6, 9.0e6}) {
    p.E() = E;
    nuc.calculate_xs(p);

    double total = 0.0;
    for (const auto& rx : nuc.reactions_) {
      if (!rx->redundant_)
        total += rx->xs(micro);
    }

    INFO("E = " << E);
    REQUIRE_THAT(micro.total, WithinAbs(total, 1e-15));
  }

  // A photon exactly at the highest grid point is evaluated at the upper end of
  // the last interval
  p.E() = 9.0e6;
  nuc.calculate_xs(p);
  REQUIRE(micro.index_grid == 5);
  REQUIRE(micro.interp_factor == 1.0);

  // Below the lowest threshold nothing can occur
  p.E() = 4.9e6;
  nuc.calculate_xs(p);
  REQUIRE(micro.total == 0.0);

  data::photonuclears.clear();
}

// When the highest grid point is repeated, a photon exactly at it must not
// select the zero-width interval between the repeated points or index past the
// end of the grid.
TEST_CASE("Photonuclear cross section at a repeated highest grid point")
{
  const std::string filename = "test_photonuclear_top.h5";
  {
    hid_t file = file_open(filename, 'w');
    hid_t group = create_group(file, "Ta180");
    write_attribute(group, "Z", 73);
    write_attribute(group, "A", 180);
    write_attribute(group, "metastable", 0);
    write_attribute(group, "atomic_weight_ratio", 178.4);
    write_dataset(group, "energy", vector<double> {1.0e6, 2.0e6, 3.0e6, 3.0e6});

    hid_t rxs_group = create_group(group, "reactions");
    write_reaction(rxs_group, 5, false, 0, vector<double> {1.0, 2.0, 3.0, 4.0});
    close_group(rxs_group);

    data::photonuclears.push_back(make_unique<PhotonuclearInteraction>(group));
    close_group(group);
    file_close(file);
  }
  std::remove(filename.c_str());

  const auto& nuc = *data::photonuclears.back();
  Particle p;
  const auto& micro = p.photonuclear_xs(nuc.index_);

  p.E() = 3.0e6;
  nuc.calculate_xs(p);
  REQUIRE(micro.index_grid == 1);
  REQUIRE(micro.interp_factor == 1.0);
  REQUIRE(micro.total == 3.0);
  REQUIRE(nuc.reactions_[0]->xs(micro) == 3.0);

  data::photonuclears.clear();
}

namespace {

NBodyPhaseSpace make_nbody(const std::string& filename, const char* particle)
{
  hid_t file = file_open(filename, 'w');
  hid_t group = create_group(file, "nbody");
  write_attribute(group, "n_particles", 3);
  write_attribute(group, "total_mass", 2.9877);
  write_attribute(group, "atomic_weight_ratio", 2.9896);
  write_attribute(group, "q_value", -6.2575e6);
  if (particle)
    write_attribute(group, "particle", particle);
  NBodyPhaseSpace dist {group};
  close_group(group);
  file_close(file);
  std::remove(filename.c_str());
  return dist;
}

} // namespace

// The maximum N-body energy is (Ap-1)/Ap times the kinetic energy available in
// the center-of-mass system. A neutron carries A/(A+1) of its energy into that
// system; a massless photon loses only the center-of-mass recoil E^2/(2Mc^2).
TEST_CASE("N-body phase space energy limit depends on the incident particle")
{
  const double Ap = 2.9877;
  const double A = 2.9896;
  const double Q = -6.2575e6;
  const double E = 20.0e6;

  // Data written before the incident particle was recorded are neutron data
  auto legacy = make_nbody("test_nbody_legacy.h5", nullptr);
  REQUIRE_THAT(legacy.max_energy(E),
    WithinRel((Ap - 1.0) / Ap * (A / (A + 1.0) * E + Q), 1e-14));

  auto neutron = make_nbody("test_nbody_neutron.h5", "neutron");
  REQUIRE(neutron.max_energy(E) == legacy.max_energy(E));

  auto photon = make_nbody("test_nbody_photon.h5", "photon");
  double E_cm = E - E * E / (2.0 * MASS_NEUTRON_EV * A) + Q;
  REQUIRE_THAT(photon.max_energy(E), WithinRel((Ap - 1.0) / Ap * E_cm, 1e-14));

  // Sampled energies never exceed the limit
  uint64_t seed = 1;
  for (int i = 0; i < 1000; ++i) {
    double E_out;
    double mu;
    photon.sample(E, E_out, mu, &seed);
    REQUIRE(E_out >= 0.0);
    REQUIRE(E_out <= photon.max_energy(E));
  }
}

namespace {

// A Kalbach-Mann distribution with constant parameters R and A at two incident
// energies, and outgoing energies uniform on [0, 1 MeV]
KalbachMann make_kalbach_mann(
  const std::string& filename, const char* particle, double R, double A)
{
  hid_t file = file_open(filename, 'w');
  hid_t group = create_group(file, "km");
  if (particle)
    write_attribute(group, "particle", particle);

  write_dataset(group, "energy", vector<double> {1.0e7, 2.0e7});
  hid_t dset = open_dataset(group, "energy");
  // One region with histogram interpolation: breakpoints, then laws
  vector<int> interpolation {2, 1};
  hsize_t dims[] {2, 1};
  write_attr(
    dset, 2, dims, "interpolation", H5T_NATIVE_INT, interpolation.data());
  close_dataset(dset);

  // Rows are outgoing energy, PDF, CDF, R and A
  auto eout = tensor::zeros<double>({5, 4});
  for (int j = 0; j < 4; ++j) {
    bool upper = (j % 2 == 1);
    eout(0, j) = upper ? 1.0e6 : 0.0;
    eout(1, j) = upper ? 0.0 : 1.0e-6;
    eout(2, j) = upper ? 1.0 : 0.0;
    eout(3, j) = R;
    eout(4, j) = A;
  }
  write_dataset(group, "distribution", eout);
  dset = open_dataset(group, "distribution");
  write_attribute(dset, "offsets", vector<int> {0, 2});
  write_attribute(dset, "interpolation", vector<int> {1, 1});
  write_attribute(dset, "n_discrete_lines", vector<int> {0, 0});
  close_dataset(dset);

  KalbachMann dist {group};
  close_group(group);
  file_close(file);
  std::remove(filename.c_str());
  return dist;
}

// Check that the angular PDF is normalized and agrees with the cosines that
// sample() produces
void check_kalbach_mann_pdf(const KalbachMann& dist)
{
  const double E_in = 1.5e7;
  uint64_t seed = 1;
  double E_out;

  // Integrate the PDF over each of n_bins equal cosine bins with Simpson's rule
  const int n_bins = 20;
  const int n_sub = 20;
  const double width = 2.0 / n_bins;
  vector<double> prob(n_bins, 0.0);
  for (int b = 0; b < n_bins; ++b) {
    double h = width / n_sub;
    for (int k = 0; k <= n_sub; ++k) {
      double mu = -1.0 + b * width + k * h;
      double w = (k == 0 || k == n_sub) ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
      prob[b] +=
        w * h / 3.0 * dist.sample_energy_and_pdf(E_in, mu, E_out, &seed);
    }
  }
  double total = 0.0;
  for (double p : prob)
    total += p;
  REQUIRE_THAT(total, WithinAbs(1.0, 1e-8));

  const int n = 400000;
  vector<int> counts(n_bins, 0);
  for (int i = 0; i < n; ++i) {
    double mu;
    dist.sample(E_in, E_out, mu, &seed);
    int b = std::min(static_cast<int>((mu + 1.0) / width), n_bins - 1);
    ++counts[b];
  }
  for (int b = 0; b < n_bins; ++b) {
    double expected = n * prob[b];
    double sigma = std::sqrt(expected * (1.0 - prob[b]));
    INFO("bin " << b << ": expected " << expected << ", sampled " << counts[b]);
    REQUIRE(std::abs(counts[b] - expected) < 5.0 * sigma);
  }
}

} // namespace

// For an incident photon the Kalbach-Mann angular distribution is isotropic
// with probability 1 - R and proportional to exp(A mu) with probability R,
// rather than the symmetric form used for an incident neutron. The PDF has to
// describe the distribution that sample() draws from.
TEST_CASE("Kalbach-Mann angular PDF matches sampling")
{
  SECTION("Incident neutron")
  {
    check_kalbach_mann_pdf(
      make_kalbach_mann("test_km_neutron.h5", nullptr, 0.4, 1.5));
  }
  SECTION("Incident photon")
  {
    check_kalbach_mann_pdf(
      make_kalbach_mann("test_km_photon.h5", "photon", 0.4, 1.5));
  }
}
