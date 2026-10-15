//! \file chain.h
//! \brief Depletion chain and associated information

#ifndef OPENMC_CHAIN_H
#define OPENMC_CHAIN_H

#include <cmath>
#include <string>
#include <unordered_map>

#include "pugixml.hpp"

#include "openmc/angle_energy.h" // for AngleEnergy
#include "openmc/array.h"
#include "openmc/distribution.h" // for UPtrDist
#include "openmc/memory.h"       // for unique_ptr
#include "openmc/particle_type.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Data for a nuclide in the depletion chain
//==============================================================================

class ChainNuclide {
public:
  // Types
  struct Product {
    std::string name;       //!< Reaction product name
    double branching_ratio; //!< Branching ratio
  };

  //! Independent fission product yields at one incident neutron energy
  struct FissionYields {
    double energy;            //!< Incident neutron energy in [eV]
    vector<Product> products; //!< Products with yields per fission
  };

  // Constructors, destructors
  ChainNuclide(pugi::xml_node node);
  ~ChainNuclide();

  //! Read the decay modes, fission product yields, and electron and positron
  //! sources that are needed only by decay sources
  //! \param node XML node of the nuclide in the chain file
  void read_decay_source_data(pugi::xml_node node);

  //! Compute the decay constant for the nuclide
  //! \return Decay constant in [1/s], zero for a stable nuclide
  double decay_constant() const
  {
    return half_life_ > 0.0 ? std::log(2.0) / half_life_ : 0.0;
  }

  const std::string& name() const { return name_; }
  double decay_energy() const { return decay_energy_; }
  const Distribution* photon_energy() const { return photon_energy_.get(); }

  //! Decay source of a given particle type
  //! \param type Particle type
  //! \return Energy distribution whose integral is the emission rate in
  //!   [particles/s/atom], or nullptr if the nuclide does not emit it
  const Distribution* source(ParticleType type) const;

  //! Energy emission rate of the decay source of a given particle type.
  //! Available after read_decay_source_data() is called.
  //! \param type Particle type
  //! \return Integral of energy times the source in [eV/s/atom]
  double source_energy(ParticleType type) const;

  const std::unordered_map<int, vector<Product>>& reaction_products() const
  {
    return reaction_products_;
  }

  //! Decay modes leading to other nuclides with their branching ratios.
  //! Available after read_decay_source_data() is called.
  const vector<Product>& decay_products() const { return decay_products_; }

  //! Independent fission product yields, resolving borrowed yields.
  //! Available after read_decay_source_data() is called.
  const vector<FissionYields>& fission_yields() const;

private:
  // Data members
  std::string name_;          //!< Name of nuclide
  double half_life_ {0.0};    //!< Half-life in [s]
  double decay_energy_ {0.0}; //!< Decay energy in [eV]
  std::unordered_map<int, vector<Product>>
    reaction_products_;                  //!< Map of MT to reaction products
  vector<Product> decay_products_;       //!< Decay targets and branching ratios
  vector<FissionYields> fission_yields_; //!< Fission product yields
  std::string fission_yield_parent_;     //!< Nuclide whose yields are borrowed
  UPtrDist photon_energy_;               //!< Decay photon energy distribution
  UPtrDist electron_energy_;             //!< Decay electron energy distribution
  UPtrDist positron_energy_;             //!< Decay positron energy distribution
  array<double, 3> source_energy_ {}; //!< Energy emission rates of the photon,
                                      //!< electron, and positron sources
};

//==============================================================================
// Angle-energy distribution for decay photon
//==============================================================================

class DecayPhotonAngleEnergy : public AngleEnergy {
public:
  explicit DecayPhotonAngleEnergy(const Distribution* dist)
    : photon_energy_(dist)
  {}

  //! Sample distribution for an angle and energy
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[out] mu Outgoing cosine with respect to current direction
  //! \param[inout] seed Pseudorandom seed pointer
  void sample(
    double E_in, double& E_out, double& mu, uint64_t* seed) const override;

  //! Sample an outgoing energy and evaluate the angular PDF
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[in] mu Scattering cosine with respect to current direction
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[inout] seed Pseudorandom seed pointer
  //! \return Probability density for the scattering cosine
  double sample_energy_and_pdf(
    double E_in, double mu, double& E_out, uint64_t* seed) const override;

private:
  const Distribution* photon_energy_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

extern std::unordered_map<std::string, int> chain_nuclide_map;
extern vector<unique_ptr<ChainNuclide>> chain_nuclides;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void read_chain_file_xml();

//! Read the data needed by decay sources for nuclides already in the chain
void read_chain_decay_source_data();

void free_memory_chain();

} // namespace openmc

#endif // OPENMC_CHAIN_H
