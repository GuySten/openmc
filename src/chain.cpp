//! \file chain.cpp
//! \brief Depletion chain and associated information

#include "openmc/chain.h"

#include <cstdlib> // for getenv
#include <memory>  // for make_unique
#include <string>  // for stod

#include <fmt/core.h>
#include <pugixml.hpp>

#include "openmc/distribution.h" // for distribution_from_xml
#include "openmc/error.h"
#include "openmc/reaction.h"
#include "openmc/xml_interface.h" // for get_node_value

namespace openmc {

//==============================================================================
// ChainNuclide implementation
//==============================================================================

ChainNuclide::ChainNuclide(pugi::xml_node node)
{
  name_ = get_node_value(node, "name");
  if (check_for_node(node, "half_life")) {
    half_life_ = std::stod(get_node_value(node, "half_life"));
  }
  if (check_for_node(node, "decay_energy")) {
    decay_energy_ = std::stod(get_node_value(node, "decay_energy"));
  }

  // Read reactions to store MT -> product map
  for (pugi::xml_node reaction_node : node.children("reaction")) {
    std::string rx_name = get_node_value(reaction_node, "type");
    if (!reaction_node.attribute("target"))
      continue;
    std::string rx_target = get_node_value(reaction_node, "target");
    double branching_ratio = 1.0;
    if (reaction_node.attribute("branching_ratio")) {
      branching_ratio =
        std::stod(get_node_value(reaction_node, "branching_ratio"));
    }
    int mt = reaction_mt(rx_name);
    reaction_products_[mt].push_back({rx_target, branching_ratio});
  }

  // Read decay modes that lead to another nuclide. Spontaneous fission is
  // skipped since its products are not given by the decay mode.
  for (pugi::xml_node decay_node : node.children("decay")) {
    if (!decay_node.attribute("target"))
      continue;
    std::string decay_type = get_node_value(decay_node, "type");
    if (decay_type.find("sf") != std::string::npos)
      continue;
    double branching_ratio =
      std::stod(get_node_value(decay_node, "branching_ratio"));
    if (branching_ratio > 0.0) {
      decay_products_.push_back(
        {get_node_value(decay_node, "target"), branching_ratio});
    }
  }

  // Read independent fission product yields, or the name of the nuclide whose
  // yields are borrowed
  pugi::xml_node fpy_node = node.child("neutron_fission_yields");
  if (fpy_node) {
    if (check_for_node(fpy_node, "parent")) {
      fission_yield_parent_ = get_node_value(fpy_node, "parent");
    }
    for (pugi::xml_node yield_node : fpy_node.children("fission_yields")) {
      FissionYields yields;
      yields.energy = std::stod(get_node_value(yield_node, "energy"));
      auto products = get_node_array<std::string>(yield_node, "products");
      auto data = get_node_array<double>(yield_node, "data");
      if (products.size() != data.size()) {
        fatal_error(fmt::format(
          "Fission yield products and data for {} in the chain file have "
          "different lengths.",
          name_));
      }
      for (size_t i = 0; i < products.size(); ++i) {
        yields.products.push_back({products[i], data[i]});
      }
      fission_yields_.push_back(std::move(yields));
    }
  }

  for (pugi::xml_node source_node : node.children("source")) {
    // Skip sources without data
    if (check_for_node(source_node, "parameters") &&
        get_node_value(source_node, "parameters", false, true).empty())
      continue;
    auto particle = get_node_value(source_node, "particle");
    if (particle == "photon") {
      photon_energy_ = distribution_from_xml(source_node);
    } else if (particle == "electron") {
      electron_energy_ = distribution_from_xml(source_node);
    } else if (particle == "positron") {
      positron_energy_ = distribution_from_xml(source_node);
    }
  }

  // Set entry in mapping
  data::chain_nuclide_map[name_] = data::chain_nuclides.size();
}

ChainNuclide::~ChainNuclide()
{
  data::chain_nuclide_map.erase(name_);
}

const Distribution* ChainNuclide::source(ParticleType type) const
{
  if (type == ParticleType::photon()) {
    return photon_energy_.get();
  } else if (type == ParticleType::electron()) {
    return electron_energy_.get();
  } else if (type == ParticleType::positron()) {
    return positron_energy_.get();
  }
  return nullptr;
}

const vector<ChainNuclide::FissionYields>& ChainNuclide::fission_yields() const
{
  if (!fission_yield_parent_.empty()) {
    auto it = data::chain_nuclide_map.find(fission_yield_parent_);
    if (it == data::chain_nuclide_map.end()) {
      fatal_error(fmt::format("Fission product yields for {} borrow from {}, "
                              "which is not present in the chain file.",
        name_, fission_yield_parent_));
    }
    return data::chain_nuclides[it->second]->fission_yields_;
  }
  return fission_yields_;
}

//==============================================================================
// DecayPhotonAngleEnergy implementation
//==============================================================================

void DecayPhotonAngleEnergy::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  E_out = photon_energy_->sample(seed).first;
  mu = Uniform(-1., 1.).sample(seed).first;
}

double DecayPhotonAngleEnergy::sample_energy_and_pdf(
  double E_in, double mu, double& E_out, uint64_t* seed) const
{
  E_out = photon_energy_->sample(seed).first;
  return 0.5;
}

//==============================================================================
// Global variables
//==============================================================================

namespace data {

std::unordered_map<std::string, int> chain_nuclide_map;
vector<unique_ptr<ChainNuclide>> chain_nuclides;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void read_chain_file_xml()
{
  free_memory_chain();

  char* chain_file_path = std::getenv("OPENMC_CHAIN_FILE");
  if (!chain_file_path) {
    return;
  }

  write_message(5, "Reading chain file: {}...", chain_file_path);

  pugi::xml_document doc;
  auto result = doc.load_file(chain_file_path);
  if (!result) {
    fatal_error(
      fmt::format("Error processing chain file: {}", chain_file_path));
  }

  // Get root element
  pugi::xml_node root = doc.document_element();

  for (auto node : root.children("nuclide")) {
    data::chain_nuclides.push_back(std::make_unique<ChainNuclide>(node));
  }
}

void free_memory_chain()
{
  data::chain_nuclides.clear();
  data::chain_nuclide_map.clear();
}

} // namespace openmc
