//! \file charged_particle.h
//! \brief Incident charged particle data

#ifndef OPENMC_CHARGED_PARTICLE_H
#define OPENMC_CHARGED_PARTICLE_H

#include <string>
#include <unordered_map>

#include <hdf5.h>

#include "openmc/memory.h"
#include "openmc/particle_data.h"
#include "openmc/reaction_product.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Incident charged particle cross section data
//!
//! This class stores cross section data for incident charged particles
//! (protons, deuterons, tritons, helions, alphas) on a specific target nuclide.
//==============================================================================

class IncidentChargedParticle {
public:
  //============================================================================
  // Constructors
  IncidentChargedParticle(hid_t group);

  //============================================================================
  // Methods

  //! Look up cross section at given energy
  //!
  //! \param[in] i_reaction  Index of reaction in reactions array
  //! \param[in] E  Energy in [eV]
  //! \return Cross section in [b]
  double xs(int i_reaction, double E) const;

  //! Get total cross section at given energy
  //!
  //! \param[in] E  Energy in [eV]
  //! \return Total cross section in [b]
  double total_xs(double E) const;

  //============================================================================
  // Data members

  std::string name_;       //!< Full name (e.g., "deuteron_H3")
  std::string projectile_; //!< Projectile type (e.g., "deuteron")
  std::string target_;     //!< Target nuclide name (e.g., "H3")
  int Z_;                  //!< Target atomic number
  int A_;                  //!< Target mass number
  int metastable_;         //!< Target metastable state
  double awr_;             //!< Target atomic weight ratio

  vector<double> energy_;        //!< Energy grid in [eV]
  vector<int> MT_;               //!< Reaction MT numbers
  vector<double> Q_value_;       //!< Q-values in [eV]
  vector<ParticleType> ejectile_; //!< Ejectile particle type for each reaction
  vector<double> xs_;            //!< Cross sections [n_energy x n_reactions], flattened

  //! Products for each reaction (indexed by reaction index, not MT)
  //! Each reaction can have multiple products with energy-dependent yields
  vector<vector<ReactionProduct>> products_;

  int n_energy_;    //!< Number of energy points
  int n_reactions_; //!< Number of reactions

private:
  //! Find energy index and interpolation factor
  //!
  //! \param[in] E  Energy in [eV]
  //! \param[out] i  Lower energy index
  //! \param[out] f  Interpolation factor
  void find_energy(double E, int& i, double& f) const;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Map from charged particle data name to index
extern std::unordered_map<std::string, int> charged_particle_map;

//! Vector of all charged particle data
extern vector<unique_ptr<IncidentChargedParticle>> charged_particles;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

//! Read charged particle data from HDF5 file
//!
//! \param[in] filename  Path to HDF5 file
void read_charged_particle_data(const std::string& filename);

//! Clear all charged particle data
void charged_particles_clear();

//! Get particle type from string name
//!
//! \param[in] name  Particle name (e.g., "proton", "deuteron")
//! \return ParticleType enum value
ParticleType particle_type_from_string(const std::string& name);

//! Get string name from particle type
//!
//! \param[in] type  ParticleType enum value
//! \return Particle name string
std::string particle_type_to_string(ParticleType type);

} // namespace openmc

#endif // OPENMC_CHARGED_PARTICLE_H
