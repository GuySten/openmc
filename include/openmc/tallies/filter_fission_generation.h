#ifndef OPENMC_TALLIES_FILTER_FISSION_GENERATION_H
#define OPENMC_TALLIES_FILTER_FISSION_GENERATION_H

#include <unordered_map>

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins events by the number of fissions in the particle's ancestry since its
//! source particle: 0 for the source particles and their non-fission
//! secondaries, 1 for the neutrons and photons of their fissions, and so on.
//==============================================================================

class FissionGenerationFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~FissionGenerationFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "fissiongeneration"; }
  FilterType type() const override { return FilterType::FISSION_GENERATION; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int>& bins() const { return bins_; }
  void set_bins(span<const int> bins);

protected:
  //----------------------------------------------------------------------------
  // Data members

  vector<int> bins_;

  std::unordered_map<int, int> map_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_FISSION_GENERATION_H
