//! \file distribution_energy.h
//! Energy distributions that depend on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ENERGY_H
#define OPENMC_DISTRIBUTION_ENERGY_H

#include "hdf5.h"
#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/vector.h"

namespace openmc {

//===============================================================================
//! Abstract class defining an energy distribution that is a function of the
//! incident energy of a projectile. Each derived type must implement a sample()
//! function that returns a sampled outgoing energy given an incoming energy
//===============================================================================

class EnergyDistribution {
public:
  virtual double sample(double E, uint64_t* seed) const = 0;
  virtual ~EnergyDistribution() = default;
};

//===============================================================================
//! Discrete photon energy distribution
//===============================================================================

class DiscretePhoton : public EnergyDistribution {
public:
  explicit DiscretePhoton(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  int primary_flag_; //!< Indicator of whether the photon is a primary or
                     //!< non-primary photon.
  double energy_;    //!< Photon energy or binding energy
  double A_;         //!< Atomic weight ratio of the target nuclide
};

//===============================================================================
//! Level inelastic scattering distribution
//===============================================================================

class LevelInelastic : public EnergyDistribution {
public:
  explicit LevelInelastic(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  double threshold_;  //!< Energy threshold in lab, (A + 1)/A * |Q|
  double mass_ratio_; //!< (A/(A+1))^2
};

//===============================================================================
//! An energy distribution represented as a tabular distribution with histogram
//! or linear-linear interpolation. This corresponds to ACE law 4, which NJOY
//! produces for a number of ENDF energy distributions.
//===============================================================================

class ContinuousTabular : public EnergyDistribution {
public:
  //! \param[in] group HDF5 group to read from
  //! \param[in] unit_base Whether to remap the sampled outgoing energy onto
  //!   the interpolated [E_1, E_K] range of the bracketing tables. Correct
  //!   when the whole distribution scales with the incident energy, as for
  //!   fission and bremsstrahlung spectra. Must be false for distributions
  //!   anchored at a fixed lower limit -- notably electroionization, whose
  //!   knock-on spectrum is pinned at the subshell binding energy and falls as
  //!   1/T^2, so that only the upper endpoint scales. Stretching such a
  //!   distribution inflates the mean energy transfer by the ratio of the
  //!   endpoints, which across the sparse EEDL incident-energy grid can be a
  //!   factor of tens.
  explicit ContinuousTabular(hid_t group, bool unit_base = true);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  bool unit_base_; //!< Remap onto the interpolated [E_1, E_K] range?

  //! Outgoing energy for a single incoming energy
  struct CTTable;

  //! Invert the cumulative distribution of a single table
  //!
  //! \param[in] l Index of the table
  //! \param[in] r1 Cumulative probability in [0,1)
  //! \param[out] discrete Whether the sample fell on a discrete line
  //! \return Outgoing energy in the table's own scale, in [eV]
  double sample_table(int l, double r1, bool& discrete) const;

  //! Outgoing energy for a single incoming energy
  struct CTTable {
    Interpolation interpolation;  //!< Interpolation law
    int n_discrete;               //!< Number of of discrete energies
    tensor::Tensor<double> e_out; //!< Outgoing energies in [eV]
    tensor::Tensor<double> p;     //!< Probability density
    tensor::Tensor<double> c;     //!< Cumulative distribution
  };

  int n_region_;                        //!< Number of inteprolation regions
  vector<int> breakpoints_;             //!< Breakpoints between regions
  vector<Interpolation> interpolation_; //!< Interpolation laws
  vector<double> energy_;               //!< Incident energy in [eV]
  vector<CTTable> distribution_; //!< Distributions for each incident energy
};

//===============================================================================
//! Evaporation spectrum corresponding to ACE law 9 and ENDF File 5, LF=9.
//===============================================================================

class Evaporation : public EnergyDistribution {
public:
  explicit Evaporation(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_;          //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Maxwell fission spectrum.
//! This corresponds to ACE law 7 and ENDF File 5, LF=7.
//===============================================================================

class MaxwellEnergy : public EnergyDistribution {
public:
  explicit MaxwellEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D theta_; //!< Incoming energy dependent parameter
  double u_;          //!< Restriction energy
};

//===============================================================================
//! Energy distribution of neutrons emitted from a Watt fission spectrum. This
//! corresponds to ACE law 11 and ENDF File 5, LF=11.
//===============================================================================

class WattEnergy : public EnergyDistribution {
public:
  explicit WattEnergy(hid_t group);

  //! Sample energy distribution
  //! \param[in] E Incident particle energy in [eV]
  //! \param[inout] seed Pseudorandom number seed pointer
  //! \return Sampled energy in [eV]
  double sample(double E, uint64_t* seed) const override;

private:
  Tabulated1D a_; //!< Energy-dependent 'a' parameter
  Tabulated1D b_; //!< Energy-dependent 'b' parameter
  double u_;      //!< Restriction energy
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ENERGY_H
