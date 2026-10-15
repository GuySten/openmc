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

  for (pugi::xml_node source_node : node.children("source")) {
    auto particle = get_node_value(source_node, "particle");
    if (particle == "photon") {
      photon_energy_ = distribution_from_xml(source_node);
      break;
    }
  }

  // Set entry in mapping
  data::chain_nuclide_map[name_] = data::chain_nuclides.size();
}

ChainNuclide::~ChainNuclide()
{
  data::chain_nuclide_map.erase(name_);
}

namespace {

//! Integral of the energy times the distribution of a decay source
//! \param node XML node of the distribution
//! \return Energy emission rate in [eV/s/atom]
double source_energy_integral(pugi::xml_node node)
{
  std::string type = get_node_value(node, "type", true, true);
  if (type == "mixture") {
    double result = 0.0;
    for (pugi::xml_node pair : node.children("pair")) {
      double p = std::stod(pair.attribute("probability").value());
      result += p * source_energy_integral(pair.child("dist"));
    }
    return result;
  }

  auto params = get_node_array<double>(node, "parameters");
  std::size_t n = params.size() / 2;
  const double* x = params.data();
  const double* p = params.data() + n;
  double result = 0.0;
  if (type == "discrete") {
    for (std::size_t i = 0; i < n; ++i) {
      result += x[i] * p[i];
    }
  } else if (type == "tabular") {
    std::string interp = check_for_node(node, "interpolation")
                           ? get_node_value(node, "interpolation")
                           : "histogram";
    for (std::size_t i = 0; i + 1 < n; ++i) {
      double dx = x[i + 1] - x[i];
      if (interp == "histogram") {
        result += p[i] * 0.5 * (x[i + 1] + x[i]) * dx;
      } else {
        // Exact for linear-linear interpolation, an approximation otherwise
        result +=
          dx / 6.0 *
          ((2.0 * x[i] + x[i + 1]) * p[i] + (x[i] + 2.0 * x[i + 1]) * p[i + 1]);
      }
    }
  } else {
    fatal_error(fmt::format(
      "Unsupported decay source distribution type '{}' in chain file.", type));
  }
  return result;
}

} // namespace

void ChainNuclide::read_decay_source_data(pugi::xml_node node)
{
  decay_products_.clear();
  fission_yields_.clear();
  fission_yield_parent_.clear();
  source_energy_.fill(0.0);

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
      source_energy_[0] = source_energy_integral(source_node);
    } else if (particle == "electron") {
      electron_energy_ = distribution_from_xml(source_node);
      source_energy_[1] = source_energy_integral(source_node);
    } else if (particle == "positron") {
      positron_energy_ = distribution_from_xml(source_node);
      source_energy_[2] = source_energy_integral(source_node);
    }
  }
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

double ChainNuclide::source_energy(ParticleType type) const
{
  if (type == ParticleType::photon()) {
    return source_energy_[0];
  } else if (type == ParticleType::electron()) {
    return source_energy_[1];
  } else if (type == ParticleType::positron()) {
    return source_energy_[2];
  }
  return 0.0;
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

void read_chain_decay_source_data()
{
  char* chain_file_path = std::getenv("OPENMC_CHAIN_FILE");
  if (!chain_file_path || data::chain_nuclides.empty()) {
    return;
  }

  pugi::xml_document doc;
  auto result = doc.load_file(chain_file_path);
  if (!result) {
    fatal_error(
      fmt::format("Error processing chain file: {}", chain_file_path));
  }

  for (auto node : doc.document_element().children("nuclide")) {
    auto it = data::chain_nuclide_map.find(get_node_value(node, "name"));
    if (it != data::chain_nuclide_map.end()) {
      data::chain_nuclides[it->second]->read_decay_source_data(node);
    }
  }
}

void free_memory_chain()
{
  data::chain_nuclides.clear();
  data::chain_nuclide_map.clear();
}

} // namespace openmc
