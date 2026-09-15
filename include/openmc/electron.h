#ifndef OPENMC_ELECTRON_H
#define OPENMC_ELECTRON_H

#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"
#include "openmc/endf.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/particle.h"
#include "openmc/vector.h"

#include "openmc/tensor.h"
#include <hdf5.h>

#include <string>
#include <unordered_map>
#include <utility> // for pair

namespace openmc {

//==============================================================================
//! Electron interaction data for a single element
//==============================================================================

class ElectronInteraction {
public:
  // Constructors/destructor
  ElectronInteraction(hid_t group);

  // Methods
  void calculate_xs(Particle& p) const;

  double elastic_scatter(double E, uint64_t* seed) const;

  //! Mean deflection 1-<mu> at a given energy, from the transport-corrected
  //! elastic cross section with the in-peak contribution removed.
  double mean_deflection(double E) const;

  //! Factor that puts the sampled deflection onto that mean, interpolated
  //! from the table built by compute_mean_deflection().
  double elastic_rescale(double E) const;

  //! Tabulate the mean deflection of the large-angle distribution on the
  //! energy grid. Called once, from the constructor.
  void compute_mean_deflection();

  double excitation(double E) const;

  void ionization(Particle& p, int i_shell) const;

  int sample_ionization_shell(Particle& p) const;

  void bremsstrahlung(Particle& p) const;

  //! Factor that puts the sampled bremsstrahlung photon energy onto the
  //! tabulated mean, interpolated from the table built by
  //! compute_brems_rescale(). Unity when the evaluation carries no BREML
  //! block to anchor against.
  double brems_rescale(double E) const;

  //! Tabulate that factor on the energy grid. Called once, from the
  //! constructor, and only after bremsstrahlung_dist_ exists.
  void compute_brems_rescale();

  // Data members
  std::string name_;      //!< Name of element, e.g. "Zr"
  int Z_;                 //!< Atomic number
  int64_t index_;         //!< Index in data::electroatomic
  int64_t i_photoatomic_; //!< Index in data::photoatomic for this element

  //! For each electroionization subshell, the index of the matching subshell in
  //! the photoatomic PhotonInteraction::shells_ vector. The two lists are not
  //! guaranteed to have the same length or ordering, so they are matched by
  //! ENDF designator rather than by position.
  vector<int> shell_map_;

  // Microscopic cross sections
  tensor::Tensor<double> energy_;
  tensor::Tensor<double> elastic_;
  //! Transport-corrected elastic cross section, on the dense energy grid the
  //! sparse angular tables cannot supply between themselves. This is the first
  //! moment of the *total* elastic cross section, so the in-peak part has to be
  //! taken back out of it before it describes the tabulated large-angle
  //! distribution -- see mean_deflection().
  tensor::Tensor<double> elastic_transport_;
  //! Total elastic cross section: the tabulated large-angle part in elastic_
  //! plus the forward peak the evaluation leaves to an analytic form. Equal to
  //! elastic_ below the energy at which the peak opens up (1.75 MeV in Al, 3
  //! MeV in Fe, 8 MeV in U).
  tensor::Tensor<double> elastic_total_;
  //! Mean deflection 1-<mu> of the tabulated large-angle distribution, on the
  //! energy grid: the transport cross section less the forward peak's share of
  //! it, over the large-angle cross section. Built once by
  //! compute_mean_deflection() rather than per collision.
  vector<double> elastic_deflection_;
  //! Ratio of the tabulated mean deflection to the one the angular
  //! distribution actually samples, on the energy grid. Built once by
  //! compute_mean_deflection(); the quadrature behind it is far too expensive
  //! to repeat per collision.
  vector<double> elastic_rescale_;
  AngleDistribution elastic_angle_;
  tensor::Tensor<double> ionization_;
  vector<unique_ptr<ContinuousTabular>> ionization_dist_;
  tensor::Tensor<double> excitation_;
  Tabulated1D excitation_energy_loss_;
  tensor::Tensor<double> bremsstrahlung_;
  unique_ptr<ContinuousTabular> bremsstrahlung_dist_;
  //! Average energy of the emitted bremsstrahlung photon, from the BREML block
  //! at JXS(26). The photon spectra are tabulated on only nine incident
  //! energies spanning ten decades, and between them the sampled mean drifts
  //! several percent off this curve; BREML gives the first moment on a grid
  //! dense enough to interpolate. Empty if the evaluation has no such block.
  Tabulated1D brems_mean_energy_;
  //! Whether brems_mean_energy_ was populated from the evaluation.
  bool has_brems_mean_energy_ {false};
  //! Ratio of that tabulated mean to the one the spectrum actually samples, on
  //! the energy grid. Built once by compute_brems_rescale(); the quadrature
  //! behind it is far too expensive to repeat per collision. Empty when there
  //! is no BREML block, which the sampling reads as "leave the photon alone".
  vector<double> brems_rescale_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Maps element name to index in data::photoatomic. Owned by photon.cpp;
//! electron data must not write to it.
extern std::unordered_map<std::string, int> element_map;

//! Maps element name to index in data::electroatomic
extern std::unordered_map<std::string, int> electron_map;

extern vector<unique_ptr<ElectronInteraction>> electroatomic;

} // namespace data

namespace detail {

double evaluate_2BN_differential(double T_0, double k_photon, double theta);

double sample_2BN(double T_0, double k_photon, uint64_t* seed);

double sample_schiff_2BS(
  double E_electron, double k_photon, int Z, uint64_t* seed);
} // namespace detail

} // namespace openmc

#endif // OPENMC_ELECTRON_H
