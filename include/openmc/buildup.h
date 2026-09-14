#ifndef OPENMC_BUILDUP_H
#define OPENMC_BUILDUP_H

#include <string>

#include "openmc/tensor.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Energy-dependent buildup factors tabulated by buff.
//!
//! A buildup factor is the ratio of the total flux to the uncollided flux at
//! the same point of the same infinite medium. buff tabulates it resolved in
//! outgoing energy: G(E_in, X, y) is the scattered flux arriving in the bin at
//! y = E_out/E_in per unit uncollided flux, for a source of energy E_in seen
//! through X mean free paths. The uncollided beam itself is not in the table
//! -- it is a delta of weight one at E_in, which the transport code already
//! has.
//!
//! That resolution is the point. A table of scalar buildup factors answers one
//! response on one energy mesh; this one is folded by whatever the tally asks
//! for, because each outgoing bin can be scored at its own energy.
//!
//! The outgoing energy is a *fraction* of the source energy, which is what
//! lets one table serve a continuum source. Stored in absolute energy the bins
//! slide out from under each other as the source energy moves -- a bin
//! populated at one source energy is empty at the next -- so interpolating
//! them is 100% wrong at any spacing, and snapping to the nearest tabulated
//! source energy is no better: at 40 mfp the scattered term is 99.9% of the
//! flux, so nothing dilutes the error, and 2% spacing already costs 3.3%. In
//! the ratio the Compton edge stays put; 5% spacing costs 1.6% per bin and
//! 0.44% folded into a response.
//!
//! The depths are counted in mean free paths of the total attenuation
//! coefficient, and the table carries that coefficient so that a code
//! measuring its own optical depth can check it measures the same thing.
//==============================================================================

class BuildupTable {
public:
  //! Read a buff-buildup-table HDF5 file.
  //! \param[in] path Path to the file
  explicit BuildupTable(const std::string& path);

  //==========================================================================
  // Lookup

  //! Whether a source energy is inside the tabulated range.
  //! \param[in] E Source energy in eV
  bool covers(double E) const
  {
    return E >= source_energies_.front() && E <= source_energies_.back();
  }

  //! Scattered flux per unit uncollided flux, bin by bin.
  //!
  //! Both axes are interpolated logarithmically. Absorption edges are not
  //! crossed: above one, fluorescence opens a channel of photons more
  //! penetrating than the source photons that made them, so the buildup
  //! factor is genuinely discontinuous, and the interpolation is confined to
  //! the interval holding the source energy.
  //!
  //! \param[in] E Source energy in eV, anywhere inside the tabulated range
  //! \param[in] depth Optical depth in mean free paths at the source energy
  //! \param[out] spectrum Resized to n_bins(); the scattered buildup factor
  //!   of each outgoing bin, so that its sum plus one is the number buildup
  //!   factor
  void scattered(double E, double depth, vector<double>& spectrum) const;

  //! Energy at which each outgoing bin should be scored, eV.
  //!
  //! \param[in] E Source energy in eV
  //! \param[out] energies Resized to n_bins()
  void bin_energies(double E, vector<double>& energies) const;

  //==========================================================================
  // Accessors

  //! Outgoing bin edges as a fraction of the source energy, ascending
  const vector<double>& ratio_edges() const { return ratio_edges_; }

  //! Where in each bin it is scored, as a fraction of the source energy. The
  //! geometric mean of the bin's edges, matching where buff folds a response
  //! when it builds the table.
  const vector<double>& ratio_centres() const { return ratio_centres_; }

  //! Tabulated source energies in eV, ascending
  const vector<double>& source_energies() const { return source_energies_; }

  //! Tabulated depths in mean free paths, ascending
  const vector<double>& depths() const { return depths_; }

  //! Total mass attenuation coefficient in cm^2/g at each source energy, on
  //! the convention the depths are counted in.
  const vector<double>& mu_total() const { return mu_total_; }

  //! Total mass attenuation coefficient at an arbitrary source energy, cm^2/g
  double mu_total(double E) const;

  //! Material the table was built for, as buff named it
  const std::string& material() const { return material_; }

  const std::string& path() const { return path_; }

  int n_bins() const { return ratio_centres_.size(); }

private:
  std::string path_;
  std::string material_;
  vector<double> source_energies_; //!< eV, ascending
  vector<double> depths_;          //!< mean free paths, ascending
  vector<double> log_depths_;      //!< natural logs of depths_
  vector<double> ratio_edges_;     //!< E_out/E_in, ascending
  vector<double> ratio_centres_;   //!< E_out/E_in, geometric bin means
  vector<double> mu_total_;        //!< cm^2/g at each source energy
  //! Indices into source_energies_ at which the table is discontinuous --
  //! absorption edges. Interpolation never crosses one.
  vector<int> interval_bounds_;

  //! (source energies, depths, bins), and its logarithm where positive
  tensor::Tensor<double> scattered_;
  tensor::Tensor<double> log_scattered_;

  //! One tabulated source energy's spectrum, interpolated in depth
  void depth_slice(int i_energy, double depth, vector<double>& spectrum) const;

  //! Index range of the edge-bounded interval holding a source energy
  void interval(double E, int& lo, int& hi) const;
};

} // namespace openmc

#endif // OPENMC_BUILDUP_H
