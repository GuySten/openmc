#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openmc/buildup.h"
#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"

using namespace openmc;

namespace {

//! Write a table in buff's format, so the reader is tested against the layout
//! it will actually meet rather than against a mock of it. The values are made
//! up; what is real is the shape, the units (buff works in MeV) and the
//! attributes that say what convention the depths are counted in.
//!
//! scattered[i][k][j] = E_i * X_k^(j + 1) for the bins that carry anything,
//! which is a straight line of known slope through (ln X, ln B) and through
//! (ln E, ln B), so it pins both interpolations exactly. Bin 0 is left empty
//! throughout, standing for a bin no scatter can reach.
void write_table(const std::string& path, const std::string& attenuation = "total",
  const std::string& compton = "bound",
  const std::string& format = "buff-buildup-table-2")
{
  const std::vector<double> source_energies {1.0, 2.0};        // MeV
  const std::vector<double> depths {0.5, 1.0, 2.0, 4.0};       // mfp
  const std::vector<double> ratio_edges {0.125, 0.25, 0.5, 1.02}; // E_out/E_in
  const std::vector<double> mu_total {0.0707, 0.0493};         // cm^2/g

  std::vector<double> scattered(
    source_energies.size() * depths.size() * (ratio_edges.size() - 1));
  for (int i = 0; i < source_energies.size(); ++i)
    for (int k = 0; k < depths.size(); ++k)
      for (int j = 0; j < ratio_edges.size() - 1; ++j) {
        int n = (i * depths.size() + k) * (ratio_edges.size() - 1) + j;
        scattered[n] =
          (j == 0) ? 0.0 : source_energies[i] * std::pow(depths[k], j + 1);
      }

  hid_t file = file_open(path, 'w');
  write_attribute(file, "material", std::string {"water"});
  write_attribute(file, "format", format);
  write_attribute(file, "attenuation", attenuation);
  write_attribute(file, "compton", compton);
  write_dataset(file, "source_energies", source_energies);
  write_dataset(file, "depths", depths);
  write_dataset(file, "ratio_edges", ratio_edges);
  write_dataset(file, "mu_total", mu_total);
  hsize_t shape[3] {source_energies.size(), depths.size(), ratio_edges.size() - 1};
  write_double(file, 3, shape, "scattered", scattered.data(), false);
  file_close(file);
}

} // namespace

TEST_CASE("Buildup table reads buff's layout")
{
  const std::string path {"test_buildup_table.h5"};
  write_table(path);
  BuildupTable table(path);

  REQUIRE(table.material() == "water");
  REQUIRE(table.n_bins() == 3);

  // buff works in MeV, OpenMC in eV; the outgoing grid is dimensionless
  REQUIRE(table.source_energies()[0] == Approx(1.0e6));
  REQUIRE(table.ratio_edges().back() == Approx(1.02));
  // A bin is scored at the geometric mean of its edges, which is where buff
  // evaluates a response when it builds the table
  REQUIRE(table.ratio_centres()[0] == Approx(std::sqrt(0.125 * 0.25)));
  REQUIRE(table.mu_total()[1] == Approx(0.0493));

  // The bins follow the source energy, which is what lets one table serve a
  // continuum source
  vector<double> energies;
  table.bin_energies(1.5e6, energies);
  REQUIRE(energies[0] == Approx(std::sqrt(0.125 * 0.25) * 1.5e6));

  std::remove(path.c_str());
}

TEST_CASE("Buildup table interpolates the source energy")
{
  const std::string path {"test_buildup_source.h5"};
  write_table(path);
  BuildupTable table(path);
  vector<double> spectrum;

  REQUIRE(table.covers(1.0e6));
  REQUIRE(table.covers(1.5e6));
  REQUIRE_FALSE(table.covers(0.5e6));
  REQUIRE_FALSE(table.covers(3.0e6));

  // Exact on a tabulated source energy
  table.scattered(2.0e6, 2.0, spectrum);
  REQUIRE(spectrum[1] == Approx(2.0 * std::pow(2.0, 2)));

  // The stored value is proportional to the source energy, so a logarithmic
  // interpolation between them is exact. This is the lookup that storing the
  // outgoing energy as a fraction of the source makes possible at all: in
  // absolute energy the bins slide out from under each other and there is
  // nothing to interpolate.
  table.scattered(std::sqrt(2.0) * 1.0e6, 2.0, spectrum);
  REQUIRE(spectrum[1] == Approx(std::sqrt(2.0) * std::pow(2.0, 2)));
  REQUIRE(spectrum[2] == Approx(std::sqrt(2.0) * std::pow(2.0, 3)));
  REQUIRE(spectrum[0] == 0.0);

  std::remove(path.c_str());
}

TEST_CASE("Buildup table interpolates depth logarithmically")
{
  const std::string path {"test_buildup_depth.h5"};
  write_table(path);
  BuildupTable table(path);
  vector<double> spectrum;

  // Exact on a tabulated depth
  table.scattered(2.0e6, 2.0, spectrum);
  REQUIRE(spectrum[0] == 0.0); // a bin nothing reaches stays empty
  REQUIRE(spectrum[1] == Approx(2.0 * std::pow(2.0, 2)));
  REQUIRE(spectrum[2] == Approx(2.0 * std::pow(2.0, 3)));

  // A power of the depth is a straight line through (ln X, ln B), so the
  // interpolation is exact between tabulated depths as well
  table.scattered(1.0e6, 3.0, spectrum);
  REQUIRE(spectrum[1] == Approx(std::pow(3.0, 2)));
  REQUIRE(spectrum[2] == Approx(std::pow(3.0, 3)));

  // Below the shallowest depth the line is continued: the scattered flux is
  // proportional to depth as the depth goes to zero, so (ln X, ln B) is a
  // line there and the extrapolation is asymptotically exact
  table.scattered(1.0e6, 0.05, spectrum);
  REQUIRE(spectrum[1] == Approx(std::pow(0.05, 2)));

  // Above the deepest it is held flat instead: nothing bounds a buildup
  // factor from above, and a ray that deep is attenuated to nothing anyway
  table.scattered(1.0e6, 400.0, spectrum);
  REQUIRE(spectrum[2] == Approx(std::pow(4.0, 3)));

  std::remove(path.c_str());
}

TEST_CASE("Buildup table refuses what it cannot be read against")
{
  const std::string path {"test_buildup_refuse.h5"};

  // OpenMC attenuates photons on the total cross section. A table whose
  // depths are counted in anything else is wrong by exp of the difference,
  // which reaches 10% of the coefficient for iron near 100 keV.
  write_table(path, "no-coherent");
  REQUIRE_THROWS(BuildupTable {path});

  write_table(path, "total", "free");
  REQUIRE_THROWS(BuildupTable {path});

  write_table(path, "total", "bound", "buff-buildup-table-1");
  REQUIRE_THROWS(BuildupTable {path});

  std::remove(path.c_str());
}
