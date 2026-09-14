#include "openmc/buildup.h"

#include <algorithm> // for lower_bound, is_sorted
#include <cmath>     // for log, exp

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"

namespace openmc {

namespace {

//! Read a string attribute written either way.
//!
//! h5py writes string attributes as variable-length UTF-8 by default, which is
//! what buff's tables carry and what read_attribute() cannot read: it sizes a
//! fixed-length type from the attribute, gets the eight bytes of a pointer,
//! and HDF5 refuses the conversion. Rather than make every producer of a table
//! write HDF5 the way C++ likes it, read both.
std::string read_string_attribute(hid_t obj, const char* name)
{
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0)
    return {};
  hid_t type = H5Aget_type(attr);
  hid_t mem_type = H5Tcopy(H5T_C_S1);
  std::string value;

  if (H5Tis_variable_str(type) > 0) {
    H5Tset_size(mem_type, H5T_VARIABLE);
    H5Tset_cset(mem_type, H5Tget_cset(type));
    char* buffer {nullptr};
    if (H5Aread(attr, mem_type, &buffer) >= 0 && buffer) {
      value = buffer;
      H5free_memory(buffer);
    }
  } else {
    size_t n = H5Tget_size(type);
    H5Tset_size(mem_type, n);
    H5Tset_cset(mem_type, H5Tget_cset(type));
    std::string buffer(n, '\0');
    if (H5Aread(attr, mem_type, &buffer[0]) >= 0)
      value = buffer.c_str(); // a fixed-length string is NUL-padded
  }

  H5Tclose(mem_type);
  H5Tclose(type);
  H5Aclose(attr);
  return value;
}

} // namespace

BuildupTable::BuildupTable(const std::string& path) : path_ {path}
{
  hid_t file = file_open(path, 'r');

  std::string format = read_string_attribute(file, "format");
  if (format != "buff-buildup-table-2") {
    file_close(file);
    // Version 1 stored the outgoing energy in MeV rather than as a fraction
    // of the source energy, and cannot be reinterpreted: its grid was never
    // aligned to any particular source energy.
    fatal_error(fmt::format("{} is not a buff buildup table (format attribute "
                            "is '{}', expected 'buff-buildup-table-2')",
      path, format));
  }

  material_ = read_string_attribute(file, "material");

  // OpenMC attenuates photons on the total cross section: incoherent with the
  // bound scattering function, coherent in full, photoelectric and pair
  // production. A table built on any other convention counts its depths in
  // different mean free paths, and handing it OpenMC's optical depth is wrong
  // by exp of the difference -- 10% of the coefficient for iron near 100 keV.
  // The classic buildup-factor convention that drops coherent scattering is
  // the usual way to arrive here, and it is a rebuild, not a rescale: the
  // transport behind the table excluded it too.
  std::string attenuation = read_string_attribute(file, "attenuation");
  std::string compton = read_string_attribute(file, "compton");
  if (attenuation != "total" || compton != "bound") {
    file_close(file);
    fatal_error(
      fmt::format("buildup table {} was built with attenuation='{}' and "
                  "compton='{}', which is not the total cross section OpenMC "
                  "attenuates photons on. Rebuild it with buff's defaults "
                  "(compton='bound', coherent='transport').",
        path, attenuation, compton));
  }

  read_dataset(file, "source_energies", source_energies_);
  read_dataset(file, "depths", depths_);
  read_dataset(file, "ratio_edges", ratio_edges_);
  read_dataset(file, "scattered", scattered_);
  if (object_exists(file, "mu_total")) {
    read_dataset(file, "mu_total", mu_total_);
  }
  if (object_exists(file, "interval_bounds")) {
    vector<int64_t> bounds;
    read_dataset(file, "interval_bounds", bounds);
    for (int64_t b : bounds)
      interval_bounds_.push_back(static_cast<int>(b));
  }
  file_close(file);

  // buff works in MeV throughout; OpenMC works in eV. The outgoing grid is
  // dimensionless and needs no conversion.
  for (auto& E : source_energies_)
    E *= 1.0e6;

  int n_energy = source_energies_.size();
  int n_depth = depths_.size();
  int n_bin = ratio_edges_.size() - 1;
  const auto& shape = scattered_.shape();
  if (shape.size() != 3 || static_cast<int>(shape[0]) != n_energy ||
      static_cast<int>(shape[1]) != n_depth ||
      static_cast<int>(shape[2]) != n_bin) {
    fatal_error(fmt::format("buildup table {} has a scattered array that does "
                            "not match its own axes",
      path));
  }
  if (n_depth < 2 || n_bin < 1) {
    fatal_error(fmt::format(
      "buildup table {} needs at least two depths and one outgoing bin", path));
  }
  if (!std::is_sorted(depths_.begin(), depths_.end()) || depths_.front() <= 0.0) {
    fatal_error(fmt::format(
      "buildup table {} has depths that are not ascending and positive", path));
  }
  if (!mu_total_.empty() && mu_total_.size() != source_energies_.size()) {
    fatal_error(fmt::format(
      "buildup table {} has one mu_total per axis it does not have", path));
  }

  // Where in each bin it is scored. buff folds a response at the geometric
  // mean of a bin's edges when it tabulates, so reading the table back at any
  // other point would answer a different question than the table was checked
  // against.
  ratio_centres_.reserve(n_bin);
  for (int j = 0; j < n_bin; ++j)
    ratio_centres_.push_back(std::sqrt(ratio_edges_[j] * ratio_edges_[j + 1]));

  log_depths_.reserve(n_depth);
  for (double X : depths_)
    log_depths_.push_back(std::log(X));

  // Depth is interpolated as a straight line through (ln X, ln B) bin by bin,
  // so the logarithms are taken once here rather than at every lookup. A bin
  // is either positive at every depth or zero at every depth -- a bin above
  // the source energy can never be reached, and one below it is reached at
  // any depth -- so a single flag per (energy, bin) decides, and the zeros
  // need no logarithm.
  log_scattered_.resize({static_cast<size_t>(n_energy),
    static_cast<size_t>(n_depth), static_cast<size_t>(n_bin)});
  for (int i = 0; i < n_energy; ++i) {
    for (int j = 0; j < n_bin; ++j) {
      bool carries = scattered_(i, 0, j) > 0.0;
      for (int k = 0; k < n_depth; ++k) {
        double value = scattered_(i, k, j);
        if (carries && !(value > 0.0)) {
          fatal_error(fmt::format("buildup table {} has a bin that vanishes at "
                                  "some depths but not others, which the "
                                  "logarithmic depth interpolation cannot read",
            path));
        }
        log_scattered_(i, k, j) = carries ? std::log(value) : 0.0;
      }
    }
  }
}

void BuildupTable::interval(double E, int& lo, int& hi) const
{
  lo = 0;
  hi = source_energies_.size();
  for (int bound : interval_bounds_) {
    if (source_energies_[bound] <= E)
      lo = bound;
    else {
      hi = bound;
      break;
    }
  }
}

void BuildupTable::bin_energies(double E, vector<double>& energies) const
{
  energies.resize(ratio_centres_.size());
  for (int j = 0; j < ratio_centres_.size(); ++j)
    energies[j] = ratio_centres_[j] * E;
}

double BuildupTable::mu_total(double E) const
{
  if (mu_total_.empty())
    return 0.0;
  int n = source_energies_.size();
  int k = std::lower_bound(source_energies_.begin(), source_energies_.end(), E) -
          source_energies_.begin();
  k = std::max(1, std::min(k, n - 1));
  double f = std::log(E / source_energies_[k - 1]) /
             std::log(source_energies_[k] / source_energies_[k - 1]);
  return std::exp((1.0 - f) * std::log(mu_total_[k - 1]) +
                  f * std::log(mu_total_[k]));
}

void BuildupTable::depth_slice(
  int i_energy, double depth, vector<double>& spectrum) const
{
  int n_bin = this->n_bins();
  int n_depth = depths_.size();
  spectrum.assign(n_bin, 0.0);

  // Below the shallowest tabulated depth the straight line is continued
  // rather than cut off. It is the right thing to continue: the scattered
  // flux is proportional to the depth as the depth goes to zero -- one
  // scatter, one chance to have it -- so (ln X, ln B) is a line of slope one
  // there, and the extrapolation is asymptotically exact. Beyond the deepest
  // it is held flat instead, because nothing bounds the growth of a buildup
  // factor from above; a ray that far in is attenuated by exp(-X) and
  // contributes nothing worth being wrong about.
  double log_depth = std::log(depth);
  int k = std::lower_bound(depths_.begin(), depths_.end(), depth) -
          depths_.begin();
  k = std::max(1, std::min(k, n_depth - 1));
  if (depth >= depths_.back())
    log_depth = log_depths_.back();

  double span = log_depths_[k] - log_depths_[k - 1];
  double f = (log_depth - log_depths_[k - 1]) / span;

  for (int j = 0; j < n_bin; ++j) {
    if (!(scattered_(i_energy, 0, j) > 0.0))
      continue;
    double lo = log_scattered_(i_energy, k - 1, j);
    double hi = log_scattered_(i_energy, k, j);
    spectrum[j] = std::exp(lo + f * (hi - lo));
  }
}

void BuildupTable::scattered(
  double E, double depth, vector<double>& spectrum) const
{
  int lo, hi;
  this->interval(E, lo, hi);

  // The two tabulated source energies straddling this one, inside the
  // interval. An absorption edge is a real discontinuity -- above one,
  // fluorescence opens a channel of photons more penetrating than the source
  // photons that made them -- so the stencil never spans one.
  int upper = std::lower_bound(source_energies_.begin() + lo,
                source_energies_.begin() + hi, E) - source_energies_.begin();
  upper = std::max(lo + 1, std::min(upper, hi - 1));

  if (hi - lo < 2) {
    this->depth_slice(lo, depth, spectrum);
    return;
  }

  int lower = upper - 1;
  double f = std::log(E / source_energies_[lower]) /
             std::log(source_energies_[upper] / source_energies_[lower]);
  f = std::max(0.0, std::min(1.0, f));

  this->depth_slice(lower, depth, spectrum);
  vector<double> other;
  this->depth_slice(upper, depth, other);

  for (int j = 0; j < spectrum.size(); ++j) {
    double a = spectrum[j], b = other[j];
    if (a > 0.0 && b > 0.0) {
      spectrum[j] = std::exp((1.0 - f) * std::log(a) + f * std::log(b));
    } else {
      // A bin only one side reaches is faded in linearly rather than dropped:
      // it is the edge of the spectrum moving, and it carries little.
      spectrum[j] = (1.0 - f) * a + f * b;
    }
  }
}

} // namespace openmc
