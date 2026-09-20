#include "openmc/material.h"

#include <algorithm> // for min, max, sort, fill
#include <cassert>
#include <cmath>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "openmc/tensor.h"

#include "openmc/capi.h"
#include "openmc/condensed_history.h"
#include "openmc/container_util.h"
#include "openmc/cross_sections.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/gos.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/photon.h"
#include "openmc/photonuclear.h"
#include "openmc/random_dist.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/string_utils.h"
#include "openmc/thermal.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace model {

std::unordered_map<int32_t, int32_t> material_map;
vector<unique_ptr<Material>> materials;

} // namespace model

//==============================================================================
// Material implementation
//==============================================================================

Material::Material(pugi::xml_node node)
{
  index_ = model::materials.size(); // Avoids warning about narrowing

  if (check_for_node(node, "id")) {
    this->set_id(std::stoi(get_node_value(node, "id")));
  } else {
    fatal_error("Must specify id of material in materials XML file.");
  }

  if (check_for_node(node, "name")) {
    name_ = get_node_value(node, "name");
  }

  if (check_for_node(node, "cfg")) {
    auto cfg = get_node_value(node, "cfg");
    write_message(
      5, "NCrystal config string for material #{}: '{}'", this->id(), cfg);
    ncrystal_mat_ = NCrystalMat(cfg);
  }

  if (check_for_node(node, "depletable")) {
    depletable_ = get_node_value_bool(node, "depletable");
  }

  bool sum_density {false};
  pugi::xml_node density_node = node.child("density");
  std::string units;
  if (density_node) {
    units = get_node_value(density_node, "units");
    if (units == "sum") {
      sum_density = true;
    } else if (units == "macro") {
      if (check_for_node(density_node, "value")) {
        density_ = std::stod(get_node_value(density_node, "value"));
      } else {
        density_ = 1.0;
      }
    } else {
      double val = std::stod(get_node_value(density_node, "value"));
      if (val <= 0.0) {
        fatal_error("Need to specify a positive density on material " +
                    std::to_string(id_) + ".");
      }

      if (units == "g/cc" || units == "g/cm3") {
        density_ = -val;
      } else if (units == "kg/m3") {
        density_ = -1.0e-3 * val;
      } else if (units == "atom/b-cm") {
        density_ = val;
      } else if (units == "atom/cc" || units == "atom/cm3") {
        density_ = 1.0e-24 * val;
      } else {
        fatal_error("Unknown units '" + units + "' specified on material " +
                    std::to_string(id_) + ".");
      }
    }
  } else {
    fatal_error("Must specify <density> element in material " +
                std::to_string(id_) + ".");
  }

  if (node.child("element")) {
    fatal_error(
      "Unable to add an element to material " + std::to_string(id_) +
      " since the element option has been removed from the xml input. "
      "Elements can only be added via the Python API, which will expand "
      "elements into their natural nuclides.");
  }

  // =======================================================================
  // READ AND PARSE <nuclide> TAGS

  // Check to ensure material has at least one nuclide
  if (!check_for_node(node, "nuclide") &&
      !check_for_node(node, "macroscopic")) {
    fatal_error("No macroscopic data or nuclides specified on material " +
                std::to_string(id_));
  }

  // Create list of macroscopic x/s based on those specified, just treat
  // them as nuclides. This is all really a facade so the user thinks they
  // are entering in macroscopic data but the code treats them the same
  // as nuclides internally.
  // Get pointer list of XML <macroscopic>
  auto node_macros = node.children("macroscopic");
  int num_macros = std::distance(node_macros.begin(), node_macros.end());

  vector<std::string> names;
  vector<double> densities;
  if (settings::run_CE && num_macros > 0) {
    fatal_error("Macroscopic can not be used in continuous-energy mode.");
  } else if (num_macros > 1) {
    fatal_error("Only one macroscopic object permitted per material, " +
                std::to_string(id_));
  } else if (num_macros == 1) {
    pugi::xml_node node_nuc = *node_macros.begin();

    // Check for empty name on nuclide
    if (!check_for_node(node_nuc, "name")) {
      fatal_error("No name specified on macroscopic data in material " +
                  std::to_string(id_));
    }

    // store nuclide name
    std::string name = get_node_value(node_nuc, "name", false, true);
    names.push_back(name);

    // Set density for macroscopic data
    if (units == "macro") {
      densities.push_back(density_);
    } else {
      fatal_error("Units can only be macro for macroscopic data " + name);
    }
  } else {
    // Create list of nuclides based on those specified
    for (auto node_nuc : node.children("nuclide")) {
      // Check for empty name on nuclide
      if (!check_for_node(node_nuc, "name")) {
        fatal_error(
          "No name specified on nuclide in material " + std::to_string(id_));
      }

      // store nuclide name
      std::string name = get_node_value(node_nuc, "name", false, true);
      names.push_back(name);

      // Check if no atom/weight percents were specified or if both atom and
      // weight percents were specified
      if (units == "macro") {
        densities.push_back(density_);
      } else {
        bool has_ao = check_for_node(node_nuc, "ao");
        bool has_wo = check_for_node(node_nuc, "wo");

        if (!has_ao && !has_wo) {
          fatal_error(
            "No atom or weight percent specified for nuclide: " + name);
        } else if (has_ao && has_wo) {
          fatal_error("Cannot specify both atom and weight percents for a "
                      "nuclide: " +
                      name);
        }

        // Copy atom/weight percents
        if (has_ao) {
          densities.push_back(std::stod(get_node_value(node_nuc, "ao")));
        } else {
          densities.push_back(-std::stod(get_node_value(node_nuc, "wo")));
        }
      }
    }
  }

  // =======================================================================
  // READ AND PARSE <isotropic> element

  vector<std::string> iso_lab;
  if (check_for_node(node, "isotropic")) {
    iso_lab = get_node_array<std::string>(node, "isotropic");
  }

  // ========================================================================
  // COPY NUCLIDES TO ARRAYS IN MATERIAL

  // allocate arrays in Material object
  auto n = names.size();
  nuclide_.reserve(n);
  atom_density_ = tensor::Tensor<double>({n});
  if (settings::photon_transport)
    element_.reserve(n);

  for (int i = 0; i < n; ++i) {
    const auto& name {names[i]};

    // Check that this nuclide is listed in the nuclear data library
    // (cross_sections.xml for CE and the MGXS HDF5 for MG)
    if (settings::run_mode != RunMode::PLOTTING) {
      LibraryKey key {Library::Type::neutron, name};
      if (data::library_map.find(key) == data::library_map.end()) {
        fatal_error("Could not find nuclide " + name +
                    " in the "
                    "nuclear data library.");
      }
    }

    // If this nuclide hasn't been encountered yet, we need to add its name
    // and alias to the nuclide_dict
    if (data::nuclide_map.find(name) == data::nuclide_map.end()) {
      int index = data::nuclide_map.size();
      data::nuclide_map[name] = index;
      nuclide_.push_back(index);
    } else {
      nuclide_.push_back(data::nuclide_map[name]);
    }

    // If the corresponding element hasn't been encountered yet and photon
    // transport will be used, we need to add its symbol to the element_dict
    if (settings::photon_transport) {
      std::string element = to_element(name);

      // Make sure photon cross section data is available
      if (settings::run_mode != RunMode::PLOTTING) {
        LibraryKey key {Library::Type::photon, element};
        if (data::library_map.find(key) == data::library_map.end()) {
          fatal_error(
            "Could not find element " + element + " in cross_sections.xml.");
        }
      }

      if (data::element_map.find(element) == data::element_map.end()) {
        int index = data::element_map.size();
        data::element_map[element] = index;
        element_.push_back(index);
      } else {
        element_.push_back(data::element_map[element]);
      }
    }

    // Copy atom/weight percent
    atom_density_(i) = densities[i];
  }

  if (settings::run_CE) {
    // By default, isotropic-in-lab is not used
    if (iso_lab.size() > 0) {
      p0_.resize(n);

      // Apply isotropic-in-lab treatment to specified nuclides
      for (int j = 0; j < n; ++j) {
        for (const auto& nuc : iso_lab) {
          if (names[j] == nuc) {
            p0_[j] = true;
            break;
          }
        }
      }
    }
  }

  // Check to make sure either all atom percents or all weight percents are
  // given
  if (!((atom_density_ >= 0.0).all() || (atom_density_ <= 0.0).all())) {
    fatal_error(
      "Cannot mix atom and weight percents in material " + std::to_string(id_));
  }

  // Determine density if it is a sum value
  if (sum_density)
    density_ = atom_density_.sum();

  if (check_for_node(node, "temperature")) {
    temperature_ = std::stod(get_node_value(node, "temperature"));
  }

  if (check_for_node(node, "volume")) {
    volume_ = std::stod(get_node_value(node, "volume"));
  }

  // =======================================================================
  // READ AND PARSE <sab> TAG FOR THERMAL SCATTERING DATA
  if (settings::run_CE) {
    // Loop over <sab> elements

    vector<std::string> sab_names;
    for (auto node_sab : node.children("sab")) {
      // Determine name of thermal scattering table
      if (!check_for_node(node_sab, "name")) {
        fatal_error("Need to specify <name> for thermal scattering table.");
      }
      std::string name = get_node_value(node_sab, "name");
      sab_names.push_back(name);

      // Read the fraction of nuclei affected by this thermal scattering table
      double fraction = 1.0;
      if (check_for_node(node_sab, "fraction")) {
        fraction = std::stod(get_node_value(node_sab, "fraction"));
      }

      // Check that the thermal scattering table is listed in the
      // cross_sections.xml file
      if (settings::run_mode != RunMode::PLOTTING) {
        LibraryKey key {Library::Type::thermal, name};
        if (data::library_map.find(key) == data::library_map.end()) {
          fatal_error("Could not find thermal scattering data " + name +
                      " in cross_sections.xml file.");
        }
      }

      // Determine index of thermal scattering data in global
      // data::thermal_scatt array
      int index_table;
      if (data::thermal_scatt_map.find(name) == data::thermal_scatt_map.end()) {
        index_table = data::thermal_scatt_map.size();
        data::thermal_scatt_map[name] = index_table;
      } else {
        index_table = data::thermal_scatt_map[name];
      }

      // Add entry to thermal tables vector. For now, we put the nuclide index
      // as zero since we don't know which nuclides the table is being applied
      // to yet (this is assigned in init_thermal)
      thermal_tables_.push_back({index_table, 0, fraction});
    }
  }
}

Material::~Material()
{
  model::material_map.erase(id_);
}

Material& Material::clone()
{
  std::unique_ptr<Material> mat = std::make_unique<Material>();

  // set all other parameters to whatever the calling Material has
  mat->name_ = name_;
  mat->nuclide_ = nuclide_;
  mat->element_ = element_;
  mat->ncrystal_mat_ = ncrystal_mat_.clone();
  mat->atom_density_ = atom_density_;
  mat->density_ = density_;
  mat->density_gpcc_ = density_gpcc_;
  mat->volume_ = volume_;
  mat->fissionable() = fissionable_;
  mat->depletable() = depletable_;
  mat->p0_ = p0_;
  mat->mat_nuclide_index_ = mat_nuclide_index_;
  mat->thermal_tables_ = thermal_tables_;
  mat->temperature_ = temperature_;

  if (ttb_)
    mat->ttb_ = std::make_unique<Bremsstrahlung>(*ttb_);

  mat->index_ = model::materials.size();
  mat->set_id(C_NONE);
  model::materials.push_back(std::move(mat));
  return *model::materials.back();
}

void Material::finalize()
{
  // Set fissionable if any nuclide is fissionable
  if (settings::run_CE) {
    for (const auto& i_nuc : nuclide_) {
      if (data::nuclides[i_nuc]->fissionable_) {
        fissionable_ = true;
        break;
      }
    }

    // Generate material bremsstrahlung data for electrons and positrons
    if (settings::photon_transport &&
        settings::electron_treatment == ElectronTreatment::TTB) {
      this->init_bremsstrahlung();
    }

    // Build the oscillator model used by the inelastic angular partition
    if (settings::electron_transport) {
      this->init_electron_oscillators();
    }

    // Assign thermal scattering tables
    this->init_thermal();
  }

  // Normalize density
  this->normalize_density();
}

void Material::normalize_density()
{
  bool percent_in_atom = (atom_density_(0) >= 0.0);
  bool density_in_atom = (density_ >= 0.0);

  for (int i = 0; i < nuclide_.size(); ++i) {
    // determine atomic weight ratio
    int i_nuc = nuclide_[i];
    double awr = settings::run_CE ? data::nuclides[i_nuc]->awr_
                                  : data::mg.nuclides_[i_nuc].awr;

    // if given weight percent, convert all values so that they are divided
    // by awr. thus, when a sum is done over the values, it's actually
    // sum(w/awr)
    if (!percent_in_atom)
      atom_density_(i) = -atom_density_(i) / awr;
  }

  // determine normalized atom percents. if given atom percents, this is
  // straightforward. if given weight percents, the value is w/awr and is
  // divided by sum(w/awr)
  atom_density_ /= atom_density_.sum();

  // Change density in g/cm^3 to atom/b-cm. Since all values are now in
  // atom percent, the sum needs to be re-evaluated as 1/sum(x*awr)
  if (!density_in_atom) {
    double sum_percent = 0.0;
    for (int i = 0; i < nuclide_.size(); ++i) {
      int i_nuc = nuclide_[i];
      double awr = settings::run_CE ? data::nuclides[i_nuc]->awr_
                                    : data::mg.nuclides_[i_nuc].awr;
      sum_percent += atom_density_(i) * awr;
    }
    sum_percent = 1.0 / sum_percent;
    density_ = -density_ * N_AVOGADRO / MASS_NEUTRON * sum_percent;
  }

  // Calculate nuclide atom densities
  atom_density_ *= density_;

  // Calculate density in [g/cm^3] and charge density in [e/b-cm]
  density_gpcc_ = 0.0;
  charge_density_ = 0.0;
  for (int i = 0; i < nuclide_.size(); ++i) {
    int i_nuc = nuclide_[i];
    double awr = settings::run_CE ? data::nuclides[i_nuc]->awr_ : 1.0;
    int z = settings::run_CE ? data::nuclides[i_nuc]->Z_ : 0.0;
    density_gpcc_ += atom_density_(i) * awr * MASS_NEUTRON / N_AVOGADRO;
    charge_density_ += atom_density_(i) * z;
  }
}

void Material::init_thermal()
{
  vector<ThermalTable> tables;

  std::unordered_set<int> already_checked;
  for (const auto& table : thermal_tables_) {
    // Make sure each S(a,b) table only gets checked once
    if (already_checked.find(table.index_table) != already_checked.end()) {
      continue;
    }
    already_checked.insert(table.index_table);

    // In order to know which nuclide the S(a,b) table applies to, we need
    // to search through the list of nuclides for one which has a matching
    // name
    bool found = false;
    for (int j = 0; j < nuclide_.size(); ++j) {
      const auto& name {data::nuclides[nuclide_[j]]->name_};
      if (contains(data::thermal_scatt[table.index_table]->nuclides_, name)) {
        tables.push_back({table.index_table, j, table.fraction});
        found = true;
      }
    }

    // Check to make sure thermal scattering table matched a nuclide
    if (!found) {
      fatal_error("Thermal scattering table " +
                  data::thermal_scatt[table.index_table]->name_ +
                  " did not match any nuclide on material " +
                  std::to_string(id_));
    }
  }

  // Make sure each nuclide only appears in one table.
  for (int j = 0; j < tables.size(); ++j) {
    for (int k = j + 1; k < tables.size(); ++k) {
      if (tables[j].index_nuclide == tables[k].index_nuclide) {
        int index = nuclide_[tables[j].index_nuclide];
        auto name = data::nuclides[index]->name_;
        fatal_error(
          name + " in material " + std::to_string(id_) +
          " was found "
          "in multiple thermal scattering tables. Each nuclide can appear in "
          "only one table per material.");
      }
    }
  }

  // If there are multiple S(a,b) tables, we need to make sure that the
  // entries in i_sab_nuclides are sorted or else they won't be applied
  // correctly in the cross_section module.
  std::sort(tables.begin(), tables.end(), [](ThermalTable a, ThermalTable b) {
    return a.index_nuclide < b.index_nuclide;
  });

  // Update the list of thermal tables
  thermal_tables_ = tables;
}

Material::OscillatorTable Material::oscillator_table() const
{
  // Average electron number and average atomic weight
  double electron_density = 0.0;
  double mass_density = 0.0;

  // Log of the mean excitation energy of the material
  double log_I = 0.0;

  // Effective number of conduction electrons in the material
  double n_conduction = 0.0;

  // Oscillator strength and square of the binding energy for each oscillator
  // in material
  vector<double> f;
  vector<double> e_b_sq;

  for (int i = 0; i < element_.size(); ++i) {
    const auto& elm = *data::elements[element_[i]];
    double awr = data::nuclides[nuclide_[i]]->awr_;

    // Get atomic density of nuclide given atom/weight percent
    double atom_density =
      (atom_density_[0] > 0.0) ? atom_density_[i] : -atom_density_[i] / awr;

    electron_density += atom_density * elm.Z_;
    mass_density += atom_density * awr * MASS_NEUTRON;
    log_I += atom_density * elm.Z_ * std::log(elm.I_);

    for (int j = 0; j < elm.n_electrons_.size(); ++j) {
      if (elm.n_electrons_[j] < 0) {
        n_conduction -= elm.n_electrons_[j] * atom_density;
        continue;
      }
      e_b_sq.push_back(elm.ionization_energy_[j] * elm.ionization_energy_[j]);
      f.push_back(elm.n_electrons_[j] * atom_density);
    }
  }
  log_I /= electron_density;
  n_conduction /= electron_density;
  for (auto& f_i : f)
    f_i /= electron_density;

  // Get density in g/cm^3 if it is given in atom/b-cm
  double density = (density_ < 0.0) ? -density_ : mass_density / N_AVOGADRO;

  // Calculate the square of the plasma energy
  double e_p_sq =
    PLANCK_C * PLANCK_C * PLANCK_C * N_AVOGADRO * electron_density * density /
    (2.0 * PI * PI * FINE_STRUCTURE * MASS_ELECTRON_EV * mass_density);

  OscillatorTable osc;
  osc.f = std::move(f);
  osc.e_b_sq = std::move(e_b_sq);
  osc.e_p_sq = e_p_sq;
  osc.n_conduction = n_conduction;
  osc.log_I = log_I;
  osc.electron_density = electron_density;

  // Get the Sternheimer adjustment factor
  osc.rho = sternheimer_adjustment(
    osc.f, osc.e_b_sq, e_p_sq, n_conduction, log_I, 1.0e-6, 100);

  return osc;
}

void Material::collision_stopping_power(double* s_col, bool positron)
{
  auto osc = this->oscillator_table();
  const auto& f = osc.f;
  const auto& e_b_sq = osc.e_b_sq;
  double e_p_sq = osc.e_p_sq;
  double n_conduction = osc.n_conduction;
  double log_I = osc.log_I;
  double electron_density = osc.electron_density;
  double rho = osc.rho;

  // Classical electron radius in cm
  constexpr double CM_PER_ANGSTROM {1.0e-8};
  constexpr double r_e =
    CM_PER_ANGSTROM * PLANCK_C / (2.0 * PI * FINE_STRUCTURE * MASS_ELECTRON_EV);

  // Constant in expression for collision stopping power
  constexpr double BARN_PER_CM_SQ {1.0e24};
  double c =
    BARN_PER_CM_SQ * 2.0 * PI * r_e * r_e * MASS_ELECTRON_EV * electron_density;

  // Loop over incident charged particle energies
  for (int i = 0; i < data::ttb_e_grid.size(); ++i) {
    double E = data::ttb_e_grid(i);

    // Get the density effect correction
    double delta =
      density_effect(f, e_b_sq, e_p_sq, n_conduction, rho, E, 1.0e-6, 100);

    // Square of the ratio of the speed of light to the velocity of the charged
    // particle
    double beta_sq = E * (E + 2.0 * MASS_ELECTRON_EV) /
                     ((E + MASS_ELECTRON_EV) * (E + MASS_ELECTRON_EV));

    double tau = E / MASS_ELECTRON_EV;

    double F;
    if (positron) {
      double t = tau + 2.0;
      F = std::log(4.0) - (beta_sq / 12.0) * (23.0 + 14.0 / t + 10.0 / (t * t) +
                                               4.0 / (t * t * t));
    } else {
      F = (1.0 - beta_sq) *
          (1.0 + tau * tau / 8.0 - (2.0 * tau + 1.0) * std::log(2.0));
    }

    // Calculate the collision stopping power for this energy
    s_col[i] =
      c / beta_sq *
      (2.0 * (std::log(E) - log_I) + std::log(1.0 + tau / 2.0) + F - delta);
  }
}

void Material::init_electron_oscillators()
{
  // Every table below lives on this grid, and everything the charged-particle
  // transport reads is interpolated on it. Without it the tables would be
  // empty and the transport would run on zeros -- no density effect, no
  // collision stopping power to hold the grouped channel to -- which is a
  // wrong answer rather than a missing one, so it is refused here.
  if (data::brems_e_grid.size() < 2) {
    fatal_error("Electron transport needs the bremsstrahlung energy grid from "
                "the photon library, which was not read. Check that the "
                "library carries bremsstrahlung data for every element in "
                "material " +
                std::to_string(id_) + ".");
  }

  auto osc = this->oscillator_table();

  // Tabulate the density-effect correction. It enters the distant transverse
  // term of the angular partition below, and the Newton solve behind it is far
  // too expensive to repeat at every collision.
  auto n_e = data::brems_e_grid.size();
  density_effect_ = tensor::Tensor<double>({n_e});
  for (int i = 0; i < n_e; ++i) {
    // Zeroing the table here is what settings::density_effect switches off,
    // and it is the only place that needs to know: everything downstream --
    // the recoil model's screened slice, the grouped channel's screened
    // moments and the Berger-Seltzer stopping power below -- reads the
    // correction from this table or from the accessor over it.
    density_effect_(i) =
      settings::density_effect
        ? density_effect(osc.f, osc.e_b_sq, osc.e_p_sq, osc.n_conduction,
            osc.rho, data::brems_e_grid(i), 1.0e-6, 100)
        : 0.0;
  }

  // Tabulate the collision stopping power the material must reproduce, on the
  // same grid. 2 pi r_e^2 m c^2 is written from the constants already in hand
  // rather than added as another one, exactly as electron.cpp does.
  constexpr double bohr_radius_cm =
    PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
  constexpr double r_e = bohr_radius_cm / (FINE_STRUCTURE * FINE_STRUCTURE);
  constexpr double collision_const =
    2.0 * PI * 1.0e24 * r_e * r_e * MASS_ELECTRON_EV;
  log_I_ = osc.log_I;
  double log_I_over_mc2 = osc.log_I - std::log(MASS_ELECTRON_EV);
  for (int q = 0; q < 2; ++q) {
    collision_stopping_[q] = tensor::Tensor<double>({n_e});
    for (int i = 0; i < n_e; ++i) {
      double E = data::brems_e_grid(i);
      double tau = E / MASS_ELECTRON_EV;
      double gamma = tau + 1.0;
      double beta_sq = 1.0 - 1.0 / (gamma * gamma);
      if (!(tau > 0.0) || !(beta_sq > 0.0)) {
        collision_stopping_[q](i) = 0.0;
        continue;
      }
      double d_max = (q == 0) ? 0.5 * tau : tau;
      double bracket = std::log(tau * tau * (tau + 2.0) / 2.0) -
                       2.0 * log_I_over_mc2 - density_effect_(i) +
                       berger_seltzer_spin_term(tau, d_max, q == 1);
      // Per electron: the material's electron density belongs to the
      // transport, which already assembles it from the atom densities the
      // particle sees, density multiplier and all.
      collision_stopping_[q](i) =
        std::max(0.0, collision_const / beta_sq * bracket);
    }
  }

  // Sum the atom density of each distinct element over its nuclides: the
  // oscillator strength of a subshell is the share it holds of all the
  // electrons in the material, so every isotope of the element contributes.
  std::unordered_map<int, double> atom_density;
  for (int i = 0; i < element_.size(); ++i) {
    double awr = data::nuclides[nuclide_[i]]->awr_;
    atom_density[element_[i]] +=
      (atom_density_[0] > 0.0) ? atom_density_[i] : -atom_density_[i] / awr;
  }

  // Build an oscillator for every electroionization subshell. The table the
  // density effect is solved on has a shell list of its own, which lumps the
  // outermost electrons into a conduction term carrying no binding energy. A
  // collision has to be classified for the subshell that was actually ionized,
  // so the resonance energies are rebuilt here on the ENDF list.
  //
  // The Sternheimer factor solved above belongs to the FIRST list, which lumps
  // the outer electrons into a conduction term with no binding energy. The
  // rebuilt list gives those electrons a real photoatomic binding energy, so it
  // does not inherit that factor's meaning. Rather than rescale every resonance
  // by one number -- which would move the deep shells as far as the outer ones,
  // the opposite of what the model wants -- the adjustment is solved again
  // below on the rebuilt list itself.
  oscillator_element_.clear();
  oscillator_nuclide_.clear();
  oscillator_offset_.clear();
  oscillator_.clear();
  oscillator_block_.clear();
  vector<double> strength;
  vector<double> binding_sq;
  // Ordered, not in whatever order the hash map happens to hold them. What is
  // built below is indexed by position, so the order decides the layout of
  // every table derived from it -- and a layout that depends on the standard
  // library's bucketing is one that can differ between builds.
  vector<std::pair<int, double>> ordered(
    atom_density.begin(), atom_density.end());
  std::sort(ordered.begin(), ordered.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& kv : ordered) {
    const auto& elm = *data::elements[kv.first];
    oscillator_block_[kv.first] = oscillator_element_.size();
    oscillator_element_.push_back(kv.first);
    // Which nuclide a collision with this block is attributed to. An element
    // can stand for several nuclides of the material and the collision is
    // with the element, not with any one of them, so the first is taken --
    // the same one the element loop would have reached first.
    int i_nuc = -1;
    for (int i = 0; i < element_.size(); ++i) {
      if (element_[i] == kv.first) {
        i_nuc = nuclide_[i];
        break;
      }
    }
    oscillator_nuclide_.push_back(i_nuc);
    oscillator_offset_.push_back(oscillator_.size());
    for (int j = 0; j < elm.electron_shell_map_.size(); ++j) {
      const auto& shell = elm.shells_[elm.electron_shell_map_[j]];
      double f_i = shell.num_electrons * kv.second / osc.electron_density;
      double u = shell.binding_energy;
      oscillator_.push_back({f_i, shell.num_electrons, u,
        std::sqrt(osc.rho * osc.rho * u * u + 2.0 / 3.0 * f_i * osc.e_p_sq)});
      strength.push_back(f_i);
      binding_sq.push_back(u * u);
    }
  }
  oscillator_offset_.push_back(oscillator_.size());

  // Re-solve the adjustment on this list, so that sum_i f_i ln(W_i) = ln(I)
  // holds for the oscillators actually used. This is one more Newton solve per
  // material at setup, and it scales each resonance by what the Sternheimer
  // model asks of it rather than by a common factor: a deeply bound shell,
  // where rho^2 U^2 dominates, moves almost in proportion, while an outer one
  // held up by the plasma term barely moves at all.
  if (!binding_sq.empty()) {
    double rho = sternheimer_adjustment(
      strength, binding_sq, osc.e_p_sq, 0.0, osc.log_I, 1.0e-6, 100);
    for (int i = 0; i < oscillator_.size(); ++i) {
      oscillator_[i].w_r = std::sqrt(
        rho * rho * binding_sq[i] + 2.0 / 3.0 * strength[i] * osc.e_p_sq);
    }
  }

  this->init_oscillator_renorm();
  this->init_gos_tables();
  this->check_electron_tables();
}
//! Renormalise each oscillator, and compensate so the total is ICRU 37
//
// PENELOPE's scheme, and both halves of it are needed. The shells bound above
// the transport cutoffs have their rate taken from the evaluated subshell
// cross sections, because that is the one thing the delta-oscillator model is
// not good enough at and it is what characteristic x-ray yields rest on. The
// remaining oscillators are then scaled by one common factor so the collision
// stopping power comes back to the ICRU 37 value exactly.
//
// Skipping the compensation is not an option. Renormalising every shell
// without it moves the stopping power by up to 80 per cent in carbon and 24
// per cent the other way in lead, because an oscillator's total cross section
// counts distant collisions that excite without ionising while the evaluated
// one counts only ionisations. Restricted to the inner shells and left
// uncompensated it is still 8 per cent out in lead. Compensated, the outer
// oscillators move by at most 11 per cent, which is well inside the
// uncertainty of resonance energies that are Sternheimer constructs rather
// than measured quantities.
//! Evaluated ionisation cross section of the subshell one oscillator stands
//! for, in [b]
//
// The oscillators are laid out in blocks, one per distinct element, in the
// same order as that element's electroionization subshells, so an index into
// the flat list resolves to an element and a shell within it.
//! Which element and subshell an oscillator stands for
//
// The oscillators are laid out in blocks, one per distinct element, in the
// same order as that element's electroionization subshells, so an index into
// the flat list resolves to both.
bool Material::resolve_oscillator(
  int k, int& i_element, int& i_shell, int& i_nuclide) const
{
  for (int b = 0; b + 1 < oscillator_offset_.size(); ++b) {
    if (k >= oscillator_offset_[b] && k < oscillator_offset_[b + 1]) {
      i_element = oscillator_element_[b];
      i_shell = k - oscillator_offset_[b];
      i_nuclide = oscillator_nuclide_[b];
      return true;
    }
  }
  return false;
}

double Material::evaluated_subshell_xs(int k, double E) const
{
  int i_element, i_shell, i_nuclide;
  if (!this->resolve_oscillator(k, i_element, i_shell, i_nuclide))
    return 0.0;
  return data::elements[i_element]->subshell_ionization_xs(i_shell, E);
}

void Material::init_oscillator_renorm()
{
  auto n_grid = data::brems_e_grid.size();
  auto n_osc = oscillator_.size();
  if (n_osc == 0 || n_grid < 2)
    return;

  // A shell is inner if a vacancy in it produces something that would be
  // transported. Below that the evaluated cross section buys nothing the
  // model does not already have.
  double u_min =
    std::max(settings::energy_cutoff[ParticleType::photon().transport_index()],
      settings::energy_cutoff[ParticleType::electron().transport_index()]);

  bool warned = false;
  for (int q = 0; q < 2; ++q) {
    bool positron = q == 1;
    oscillator_renorm_[q] =
      tensor::Tensor<double>({n_osc, static_cast<size_t>(n_grid)});
    for (int i = 0; i < n_grid; ++i) {
      double E = data::brems_e_grid(i);
      double delta = density_effect_(i);
      double target = collision_stopping_[q](i);

      double s_inner = 0.0;
      double s_outer = 0.0;
      double s_raw = 0.0;
      int n_outer = 0;
      vector<double> r(n_osc, 1.0);
      for (int k = 0; k < n_osc; ++k) {
        const auto& o = oscillator_[k];
        auto g = gos_oscillator(E, o.u_b, o.w_r, delta, 0.0, positron);
        double s_k = o.f * (g.s_soft + g.s_hard);
        s_raw += s_k;
        bool inner = o.u_b > u_min;
        if (inner) {
          // Renormalised to the free-atom cross section: the ratio is taken
          // against the unscreened model, and the screening then applies on
          // top of it, exactly as PENELOPE's DFERMI does
          auto g0 = gos_oscillator(E, o.u_b, o.w_r, 0.0, 0.0, positron);
          // gos_oscillator() works per electron and the evaluated cross
          // section is per atom, so the shell's occupancy is what puts the
          // two on the same footing
          double model = o.n_e * (g0.xs_soft + g0.xs_hard);
          double ev = this->evaluated_subshell_xs(k, E);
          r[k] = (model > 0.0 && ev > 0.0) ? ev / model : 1.0;
          s_inner += r[k] * s_k;
        } else {
          s_outer += s_k;
          ++n_outer;
        }
      }

      double fnorm =
        (s_outer > 0.0 && target > 0.0) ? (target - s_inner) / s_outer : 1.0;
      if (!(fnorm > 0.0)) {
        // The inner shells alone already carry more than the material's whole
        // collision stopping power, so there is nothing left for the outer
        // ones to be scaled to. Rather than clamp and leave the total wrong,
        // fall back to one factor over every oscillator: the rates lose their
        // grip on the evaluated cross sections, but the stopping power is
        // what the transport is built on and it stays right. The factor is
        // measured against the UNrenormalised total, since it replaces the
        // inner-shell scaling rather than compounding with it.
        double common = (s_raw > 0.0) ? target / s_raw : 1.0;
        if (!warned) {
          warned = true;
          warning(fmt::format(
            "In material {} at {:.4g} eV the renormalised inner shells carry "
            "{:.4g} of a collision stopping power of {:.4g} b eV per "
            "electron, leaving {:.4g} for the {} outer oscillators. The rates "
            "are scaled together there rather than shell by shell, so "
            "inner-shell ionisation follows the model rather than the "
            "evaluated data. Reported once per material.",
            id_, E, s_inner, target, s_outer, n_outer));
        }
        for (int k = 0; k < n_osc; ++k)
          oscillator_renorm_[q](k, i) = common;
        continue;
      }
      for (int k = 0; k < n_osc; ++k) {
        oscillator_renorm_[q](k, i) =
          (oscillator_[k].u_b > u_min) ? r[k] : fnorm;
      }
    }
  }
}

//! Tabulate what the oscillator model contributes to the transport
//
// Everything here is per electron of the material, on the same grid the
// density effect lives on. The transport multiplies by the electron density
// the particle actually sees, density multiplier and all.
void Material::init_gos_tables()
{
  auto n_grid = data::brems_e_grid.size();
  auto n_osc = oscillator_.size();
  if (n_osc == 0 || n_grid < 2)
    return;

  vector<double> energy(n_grid);
  for (int i = 0; i < n_grid; ++i)
    energy[i] = data::brems_e_grid(i);

  // Sum the model over the oscillators at one cutoff. At zero cutoff there is
  // no soft channel, so the soft moments come back zero of their own accord
  // and every field is still filled -- the tables are read unconditionally.
  auto tabulate = [&](GosTables& t, int q, bool grouped) {
    ParticleType projectile =
      (q == 0) ? ParticleType::electron() : ParticleType::positron();
    bool positron = q == 1;
    t.hard = tensor::zeros<double>({n_grid});
    t.stopping = tensor::zeros<double>({n_grid});
    t.straggling = tensor::zeros<double>({n_grid});
    t.xs1 = tensor::zeros<double>({n_grid});
    t.xs2 = tensor::zeros<double>({n_grid});
    t.cumulative = tensor::zeros<double>({n_osc, static_cast<size_t>(n_grid)});

    vector<double> hard(n_grid), lowest(n_grid);
    for (int i = 0; i < n_grid; ++i) {
      double E = energy[i];
      double w_cc = grouped ? soft_collision_cutoff(projectile, E) : 0.0;
      double delta = density_effect_(i);

      double running = 0.0;
      for (int k = 0; k < n_osc; ++k) {
        const auto& o = oscillator_[k];
        auto g = gos_oscillator(E, o.u_b, o.w_r, delta, w_cc, positron);
        double r = o.f * oscillator_renorm_[q](k, i);
        running += r * g.xs_hard;
        t.cumulative(k, i) = running;
        t.stopping(i) += r * g.s_soft;
        t.straggling(i) += r * g.w2_soft;
        t.xs1(i) += r * g.xs1_soft;
        t.xs2(i) += r * g.xs2_soft;
      }
      t.hard(i) = running;
      // Normalised, so a draw against it needs no total
      if (running > 0.0) {
        for (int k = 0; k < n_osc; ++k)
          t.cumulative(k, i) /= running;
      }

      hard[i] = running;
      lowest[i] = E - MAX_SOFT_LOSS_OVERSHOOT * soft_loss_budget(projectile, E);
    }

    // A bound on the rate over every energy a step begun here can reach
    vector<double> majorant = step_majorant(energy, hard, lowest);
    t.majorant = tensor::zeros<double>({n_grid});
    for (int i = 0; i < n_grid; ++i)
      t.majorant(i) = majorant[i];
  };

  // The whole cross section, which single-event transport runs on and which a
  // step that declines to group falls back to, is always built. The soft/hard
  // split beside it is built only when something is actually grouped.
  bool grouped = settings::deflection_cutoff > 0.0;
  for (int q = 0; q < 2; ++q) {
    tabulate(gos_full_[q], q, false);
    if (grouped) {
      tabulate(gos_[q], q, true);
    } else {
      gos_[q] = gos_full_[q];
    }
  }
}

bool Material::sample_inelastic(Particle& p, bool hard) const
{
  int q = p.type().is_positron() ? 1 : 0;
  // The table has to be the one the rate was counted from. A collision
  // ending a grouped step comes from the discrete channel; one on a flight
  // that declined to group comes from the whole cross section, and drawing it
  // from the discrete table would sample a spectrum the rate never described.
  const auto& t = hard ? gos_[q] : gos_full_[q];
  auto n_grid = data::brems_e_grid.size();
  auto n_osc = oscillator_.size();
  if (n_osc == 0 || t.cumulative.size() != n_osc * n_grid)
    return false;

  double E = p.E();
  const auto& grid = data::brems_e_grid;
  int i;
  if (E <= grid(0)) {
    i = 0;
  } else if (E >= grid(n_grid - 1)) {
    i = n_grid - 1;
  } else {
    // One of the two bracketing points, chosen with the interpolation
    // fraction as its probability, rather than a blend of two cumulatives --
    // which is not a cumulative. PENELOPE does the same.
    int j = lower_bound_index(grid.cbegin(), grid.cend(), E);
    double f = std::log(E / grid(j)) / std::log(grid(j + 1) / grid(j));
    i = (prn(p.current_seed()) < f) ? j + 1 : j;
  }

  // Which oscillator. The cumulative is normalised, so this is a plain
  // binary search on a uniform draw.
  double xi = prn(p.current_seed());
  int lo = 0;
  int hi = n_osc - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (xi > t.cumulative(mid, i)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int k = lo;

  const auto& o = oscillator_[k];
  double w_cc = hard ? soft_collision_cutoff(p.type(), E) : 0.0;
  auto c = sample_gos_collision(E, o.u_b, o.w_r,
    this->density_effect_correction(E), w_cc, q == 1, p.current_seed());
  if (!(c.w > 0.0) || c.w >= E)
    return false;

  int i_element = -1;
  int i_shell = -1;
  int i_nuclide = -1;
  bool resolved = this->resolve_oscillator(k, i_element, i_shell, i_nuclide);
  if (resolved)
    p.event_nuclide() = i_nuclide;

  // The projectile is deflected through the momentum transfer and the
  // knock-on leaves along it, so the two are coplanar with azimuths differing
  // by pi. Both are measured from the direction the projectile came in on,
  // which is why that is captured before either is applied.
  Direction u_inc = p.u();
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());

  // The projectile loses the whole transfer; the ejected electron carries
  // what is left of it once the shell has been paid for, and the vacancy
  // carries the rest away through the relaxation below.
  if (c.e_knock > 0.0) {
    Direction u_knock = rotate_angle(u_inc, c.mu_knock, &phi, p.current_seed());
    p.create_secondary(p.wgt(), u_knock, c.e_knock, ParticleType::electron());
  }
  p.mu() = c.mu;
  phi += PI;
  p.u() = rotate_angle(u_inc, c.mu, &phi, p.current_seed());
  p.E() = E - c.w;

  p.event() = TallyEvent::SCATTER;
  if (resolved && c.ionised) {
    // There is no ENDF MT for total electroionization; 534 upwards name the
    // individual subshells, which is what the data resolves anyway
    const auto& elm = *data::elements[i_element];
    p.event_mt() =
      533 + elm.shells_[elm.electron_shell_map_[i_shell]].index_subshell;
    if (settings::atomic_relaxation && elm.has_atomic_relaxation_)
      elm.atomic_relaxation(elm.electron_shell_map_[i_shell], p);
  } else {
    // The transfer stayed under the binding energy, so the atom was excited
    // as a whole and no vacancy was left behind.
    p.event_mt() = ELECTROEXCITATION;
  }
  return true;
}

Material::CollisionMoments Material::gos_collision_moments(
  int q_index, double E, double w_cc) const
{
  CollisionMoments m;
  if (q_index < 0 || q_index > 1 || oscillator_.empty())
    return m;

  bool positron = q_index == 1;
  double delta = this->density_effect_correction(E);
  // The renormalisation lives on the same grid the density effect does
  const auto& grid = data::brems_e_grid;
  const auto& renorm = oscillator_renorm_[q_index];
  auto n_grid = grid.size();
  bool have_renorm = n_grid > 1 && renorm.size() == oscillator_.size() * n_grid;

  int i = 0;
  double f_grid = 0.0;
  if (have_renorm) {
    if (E <= grid(0)) {
      i = 0;
    } else if (E >= grid(n_grid - 1)) {
      i = n_grid - 2;
      f_grid = 1.0;
    } else {
      i = lower_bound_index(grid.cbegin(), grid.cend(), E);
      f_grid = std::log(E / grid(i)) / std::log(grid(i + 1) / grid(i));
    }
  }

  for (int k = 0; k < oscillator_.size(); ++k) {
    const auto& o = oscillator_[k];
    auto g = gos_oscillator(E, o.u_b, o.w_r, delta, w_cc, positron);
    double r = o.f;
    if (have_renorm) {
      r *= renorm(k, i) + f_grid * (renorm(k, i + 1) - renorm(k, i));
    }
    m.s_soft += r * g.s_soft;
    m.w2_soft += r * g.w2_soft;
    m.xs_hard += r * g.xs_hard;
    m.s_hard += r * g.s_hard;
    m.xs1_soft += r * g.xs1_soft;
    m.xs2_soft += r * g.xs2_soft;
  }
  return m;
}

Material::CollisionMoments Material::collision_moments(
  int q_index, double E, double w_cc) const
{
  CollisionMoments m;
  if (q_index < 0 || q_index > 1 || !(E > 0.0))
    return m;

  constexpr double bohr_radius_cm =
    PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
  constexpr double r_e = bohr_radius_cm / (FINE_STRUCTURE * FINE_STRUCTURE);
  constexpr double collision_const =
    2.0 * PI * 1.0e24 * r_e * r_e * MASS_ELECTRON_EV;

  bool positron = q_index == 1;
  double tau = E / MASS_ELECTRON_EV;
  double gamma = tau + 1.0;
  double beta_sq = 1.0 - 1.0 / (gamma * gamma);
  if (!(beta_sq > 0.0))
    return m;
  double k = collision_const / beta_sq;

  double d_max = positron ? tau : 0.5 * tau;
  double w_max = d_max * MASS_ELECTRON_EV;
  double cut = std::max(0.0, std::min(w_cc, w_max));

  // Soft: Berger-Seltzer restricted to transfers under the cutoff. The
  // leading logarithm, the mean excitation energy and the density effect are
  // the same whatever the cutoff, so only the spin term moves.
  double leading = std::log(tau * tau * (tau + 2.0) / 2.0) -
                   2.0 * (log_I_ - std::log(MASS_ELECTRON_EV)) -
                   this->density_effect_correction(E);
  if (cut > 0.0) {
    m.s_soft =
      std::max(0.0, k * (leading + berger_seltzer_spin_term(
                                     tau, cut / MASS_ELECTRON_EV, positron)));
  }

  // Hard: the free binary cross section over what is left. Its stopping power
  // is what the restricted form above leaves out, identically, which is the
  // whole point of splitting it this way.
  if (w_max > cut) {
    double lo = std::max(cut, 1.0e-9 * w_max);
    m.xs_hard = k * (positron ? detail::bhabha_moment(E, lo, w_max, 0)
                              : detail::moller_moment(E, lo, w_max, 0));
    m.s_hard = k * (positron ? detail::bhabha_moment(E, lo, w_max, 1)
                             : detail::moller_moment(E, lo, w_max, 1));
  }

  // Second moment of the soft loss. Berger and Seltzer give the first moment
  // and not this one, so it comes from the free cross section, which is the
  // right shape where the second moment lives: W^2 dsigma/dW tends to a
  // constant as W falls, so the moment is carried by the top of the soft
  // range rather than by the bound transfers at the bottom of it. The
  // integral from zero converges and is started just above it, the moment
  // functions dividing by their lower limit.
  if (cut > 0.0) {
    double lo = 1.0e-9 * cut;
    m.w2_soft = k * (positron ? detail::bhabha_moment(E, lo, cut, 2)
                              : detail::moller_moment(E, lo, cut, 2));
  }
  return m;
}

double Material::collision_stopping_power(int q_index, double E) const
{
  if (q_index < 0 || q_index > 1)
    return 0.0;
  const auto& v = collision_stopping_[q_index];
  auto n = v.size();
  if (n == 0)
    return 0.0;

  const auto& grid = data::brems_e_grid;
  if (E <= grid(0))
    return v(0);
  if (E >= grid(n - 1))
    return v(n - 1);

  int i = lower_bound_index(grid.cbegin(), grid.cend(), E);
  double f = std::log(E / grid(i)) / std::log(grid(i + 1) / grid(i));
  return std::max(0.0, v(i) + f * (v(i + 1) - v(i)));
}

double Material::density_effect_correction(double E) const
{
  auto n = density_effect_.size();
  if (n == 0)
    return 0.0;

  const auto& grid = data::brems_e_grid;
  if (E <= grid(0))
    return density_effect_(0);
  if (E >= grid(n - 1))
    return density_effect_(n - 1);

  int i = lower_bound_index(grid.cbegin(), grid.cend(), E);
  double f = std::log(E / grid(i)) / std::log(grid(i + 1) / grid(i));
  return density_effect_(i) + f * (density_effect_(i + 1) - density_effect_(i));
}

void Material::init_bremsstrahlung()
{
  // Create new object
  ttb_ = make_unique<Bremsstrahlung>();

  // Get the size of the energy grids
  auto n_k = data::ttb_k_grid.size();
  auto n_e = data::ttb_e_grid.size();

  // Determine number of elements
  int n = element_.size();

  for (int particle = 0; particle < 2; ++particle) {
    // Loop over logic twice, once for electron, once for positron
    BremsstrahlungData* ttb =
      (particle == 0) ? &ttb_->electron : &ttb_->positron;
    bool positron = (particle == 1);

    // Allocate arrays for TTB data
    ttb->pdf = tensor::zeros<double>({n_e, n_e});
    ttb->cdf = tensor::zeros<double>({n_e, n_e});
    ttb->yield = tensor::zeros<double>({n_e});

    // Allocate temporary arrays
    auto stopping_power_collision = tensor::zeros<double>({n_e});
    auto stopping_power_radiative = tensor::zeros<double>({n_e});
    auto dcs = tensor::zeros<double>({n_e, n_k});

    double Z_eq_sq = 0.0;
    double sum_density = 0.0;

    // Get the collision stopping power of the material
    this->collision_stopping_power(stopping_power_collision.data(), positron);

    // Calculate the molecular DCS and the molecular radiative stopping power
    // using Bragg's additivity rule.
    for (int i = 0; i < n; ++i) {
      // Get pointer to current element
      const auto& elm = *data::elements[element_[i]];
      double awr = data::nuclides[nuclide_[i]]->awr_;

      // Get atomic density and mass density of nuclide given atom/weight
      // percent
      double atom_density =
        (atom_density_[0] > 0.0) ? atom_density_[i] : -atom_density_[i] / awr;

      // Calculate the "equivalent" atomic number Zeq of the material
      Z_eq_sq += atom_density * elm.Z_ * elm.Z_;
      sum_density += atom_density;

      // Accumulate material DCS
      dcs += (atom_density * elm.Z_ * elm.Z_) * elm.dcs_;

      // Accumulate material radiative stopping power
      stopping_power_radiative += atom_density * elm.stopping_power_radiative_;
    }
    Z_eq_sq /= sum_density;

    // Calculate the positron DCS and radiative stopping power. These are
    // obtained by multiplying the electron DCS and radiative stopping powers by
    // a factor r, which is a numerical approximation of the ratio of the
    // radiative stopping powers for positrons and electrons. Source: F. Salvat,
    // J. M. Fernández-Varea, and J. Sempau, "PENELOPE-2011: A Code System for
    // Monte Carlo Simulation of Electron and Photon Transport," OECD-NEA,
    // Issy-les-Moulineaux, France (2011).
    if (positron) {
      for (int i = 0; i < n_e; ++i) {
        double r = salvat_factor(Z_eq_sq, data::ttb_e_grid(i));
        stopping_power_radiative(i) *= r;
        tensor::View<double> dcs_i = dcs.slice(i);
        dcs_i *= r;
      }
    }

    // Total material stopping power
    tensor::Tensor<double> stopping_power =
      stopping_power_collision + stopping_power_radiative;

    // Loop over photon energies
    auto f = tensor::zeros<double>({n_e});
    auto z = tensor::zeros<double>({n_e});
    for (int i = 0; i < n_e - 1; ++i) {
      double w = data::ttb_e_grid(i);

      // Loop over incident particle energies
      for (int j = i; j < n_e; ++j) {
        double e = data::ttb_e_grid(j);

        // Reduced photon energy
        double k = w / e;

        // Find the lower bounding index of the reduced photon energy
        int i_k = lower_bound_index(
          data::ttb_k_grid.cbegin(), data::ttb_k_grid.cend(), k);

        // Get the interpolation bounds
        double k_l = data::ttb_k_grid(i_k);
        double k_r = data::ttb_k_grid(i_k + 1);
        double x_l = dcs(j, i_k);
        double x_r = dcs(j, i_k + 1);

        // Find the value of the DCS using linear interpolation in reduced
        // photon energy k
        double x = x_l + (k - k_l) * (x_r - x_l) / (k_r - k_l);

        // Square of the ratio of the speed of light to the velocity of the
        // charged particle
        double beta_sq = e * (e + 2.0 * MASS_ELECTRON_EV) /
                         ((e + MASS_ELECTRON_EV) * (e + MASS_ELECTRON_EV));

        // Compute the integrand of the PDF
        f(j) = x / (beta_sq * stopping_power(j) * w);
      }

      // Number of points to integrate
      int n = n_e - i;

      // Integrate the PDF using cubic spline integration over the incident
      // particle energy
      if (n > 2) {
        spline(n, &data::ttb_e_grid(i), &f(i), &z(i));

        double c = 0.0;
        for (int j = i; j < n_e - 1; ++j) {
          c += spline_integrate(n, &data::ttb_e_grid(i), &f(i), &z(i),
            data::ttb_e_grid(j), data::ttb_e_grid(j + 1));

          ttb->pdf(j + 1, i) = c;
        }

        // Integrate the last two points using trapezoidal rule in log-log space
      } else {
        double e_l = std::log(data::ttb_e_grid(i));
        double e_r = std::log(data::ttb_e_grid(i + 1));
        double x_l = std::log(f(i));
        double x_r = std::log(f(i + 1));

        ttb->pdf(i + 1, i) =
          0.5 * (e_r - e_l) * (std::exp(e_l + x_l) + std::exp(e_r + x_r));
      }
    }

    // Loop over incident particle energies
    for (int j = 1; j < n_e; ++j) {
      // Set last element of PDF to small non-zero value to enable log-log
      // interpolation
      ttb->pdf(j, j) = std::exp(-500.0);

      // Loop over photon energies
      double c = 0.0;
      for (int i = 0; i < j; ++i) {
        // Integrate the CDF from the PDF using the fact that the PDF is linear
        // in log-log space
        double w_l = std::log(data::ttb_e_grid(i));
        double w_r = std::log(data::ttb_e_grid(i + 1));
        double x_l = std::log(ttb->pdf(j, i));
        double x_r = std::log(ttb->pdf(j, i + 1));
        double beta = (x_r - x_l) / (w_r - w_l);
        double a = beta + 1.0;
        c += std::exp(w_l + x_l) / a * std::expm1(a * (w_r - w_l));
        ttb->cdf(j, i + 1) = c;
      }

      // Set photon number yield
      ttb->yield(j) = c;
    }

    // Use logarithm of number yield since it is log-log interpolated
    ttb->yield =
      tensor::where(ttb->yield > 0.0, tensor::log(ttb->yield), -500.0);
  }
}

void Material::init_nuclide_index()
{
  int n = settings::run_CE ? data::nuclides.size() : data::mg.nuclides_.size();
  mat_nuclide_index_.resize(n);
  std::fill(mat_nuclide_index_.begin(), mat_nuclide_index_.end(), C_NONE);
  for (int i = 0; i < nuclide_.size(); ++i) {
    mat_nuclide_index_[nuclide_[i]] = i;
  }
}

void Material::calculate_xs(Particle& p) const
{
  // Set all material macroscopic cross sections to zero
  p.macro_xs().total = 0.0;
  p.macro_xs().absorption = 0.0;
  p.macro_xs().fission = 0.0;
  p.macro_xs().nu_fission = 0.0;

  if (p.type().is_neutron()) {
    this->calculate_neutron_xs(p);
  } else if (p.type().is_photon()) {
    this->calculate_photon_xs(p);
  } else if (p.type().is_electron() || p.type().is_positron()) {
    // The step description is zeroed with the charged particle that reads it
    // rather than with every particle, this being on the path of every cross
    // section lookup in the run
    p.macro_xs().step = StepXS {};
    if (settings::electron_transport)
      this->calculate_electron_xs(p);
  }
}

void Material::calculate_neutron_xs(Particle& p) const
{
  // Find energy index on energy grid
  int neutron = ParticleType::neutron().transport_index();
  int i_grid =
    std::log(p.E() / data::energy_min[neutron]) / simulation::log_spacing;

  // Determine if this material has S(a,b) tables
  bool check_sab = (thermal_tables_.size() > 0);

  // Initialize position in i_sab_nuclides
  int j = 0;

  // Calculate NCrystal cross section
  double ncrystal_xs = -1.0;
  if (ncrystal_mat_ && p.E() < NCRYSTAL_MAX_ENERGY) {
    ncrystal_xs = ncrystal_mat_.xs(p);
  }

  // Add contribution from each nuclide in material
  for (int i = 0; i < nuclide_.size(); ++i) {
    // ======================================================================
    // CHECK FOR S(A,B) TABLE

    int i_sab = C_NONE;
    double sab_frac = 0.0;

    // Check if this nuclide matches one of the S(a,b) tables specified.
    // This relies on thermal_tables_ being sorted by .index_nuclide
    if (check_sab) {
      const auto& sab {thermal_tables_[j]};
      if (i == sab.index_nuclide) {
        // Get index in sab_tables
        i_sab = sab.index_table;
        sab_frac = sab.fraction;

        // If particle energy is greater than the highest energy for the
        // S(a,b) table, then don't use the S(a,b) table
        if (p.E() > data::thermal_scatt[i_sab]->energy_max_)
          i_sab = C_NONE;

        // Increment position in thermal_tables_
        ++j;

        // Don't check for S(a,b) tables if there are no more left
        if (j == thermal_tables_.size())
          check_sab = false;
      }
    }

    // ======================================================================
    // CALCULATE MICROSCOPIC CROSS SECTION

    // Get nuclide index
    int i_nuclide = nuclide_[i];

    // Update microscopic cross section for this nuclide
    p.update_neutron_xs(i_nuclide, i_grid, i_sab, sab_frac, ncrystal_xs);
    auto& micro = p.neutron_xs(i_nuclide);

    // ======================================================================
    // ADD TO MACROSCOPIC CROSS SECTION

    // Copy atom density of nuclide in material
    double atom_density = this->atom_density(i, p.density_mult());

    // Add contributions to cross sections
    p.macro_xs().total += atom_density * micro.total;
    p.macro_xs().absorption += atom_density * micro.absorption;
    p.macro_xs().fission += atom_density * micro.fission;
    p.macro_xs().nu_fission += atom_density * micro.nu_fission;
  }
}

void Material::calculate_photon_xs(Particle& p) const
{
  p.macro_xs().coherent = 0.0;
  p.macro_xs().incoherent = 0.0;
  p.macro_xs().photoelectric = 0.0;
  p.macro_xs().pair_production = 0.0;
  p.macro_xs().photonuclear = 0.0;
  p.macro_xs().neutron_prod = 0.0;

  // Add contribution from each nuclide in material
  for (int i = 0; i < nuclide_.size(); ++i) {
    // ========================================================================
    // CALCULATE MICROSCOPIC CROSS SECTION

    // Determine microscopic cross sections for this nuclide
    int i_element = element_[i];

    // Calculate microscopic cross section for this nuclide
    const auto& micro {p.photon_xs(i_element)};
    if (p.E() != micro.last_E) {
      data::elements[i_element]->calculate_xs(p);
    }

    // ========================================================================
    // ADD TO MACROSCOPIC CROSS SECTION

    // Copy atom density of nuclide in material
    double atom_density = this->atom_density(i, p.density_mult());

    // Add contributions to material macroscopic cross sections
    p.macro_xs().total += atom_density * micro.total;
    p.macro_xs().coherent += atom_density * micro.coherent;
    p.macro_xs().incoherent += atom_density * micro.incoherent;
    p.macro_xs().photoelectric += atom_density * micro.photoelectric;
    p.macro_xs().pair_production += atom_density * micro.pair_production;
  }
  if (settings::photonuclear_physics &&
      (p.E() >= data::photonuclear_energy_min)) {
    for (int i = 0; i < nuclide_.size(); ++i) {
      // Get nuclide name
      auto& name = data::nuclides[nuclide_[i]]->name_;

      // Skip nuclides without photonuclear data
      // Reuse the iterator from find() rather than calling the non-const
      // operator[] on a map shared between threads, and to avoid a second
      // string hash per nuclide per collision
      auto it = data::photonuclear_map.find(name);
      if (it == data::photonuclear_map.end())
        continue;

      int i_nuclide = it->second;

      // Calculate microscopic cross section for this nuclide
      const auto& micro {p.photonuclear_xs(i_nuclide)};
      if (p.E() != micro.last_E) {
        data::photonuclears[i_nuclide]->calculate_xs(p);
      }
      // Same per-cell density override every other macroscopic loop applies.
      // Without it macro_xs().total mixes a scaled photo-atomic part with an
      // unscaled photonuclear one, and the cutoff in sample_photon_element()
      // can go negative.
      double atom_density = this->atom_density(i, p.density_mult());

      // Add contributions to material photonuclear macroscopic cross section
      p.macro_xs().photonuclear += atom_density * micro.total;
      p.macro_xs().neutron_prod += atom_density * micro.neutron_prod;
    }
    // Update total macroscopic cross section according to macroscopic
    // photonuclear cross section
    p.macro_xs().total += p.macro_xs().photonuclear;
  }
}
//! Refuse to transport charged particles on tables that were not built
//
// Every accessor below returns zero for a table it cannot find, which is what
// the single-event mode wants -- it builds no grouped tables and reads none.
// The same zero returned because a table that should exist is missing or
// short is a different thing entirely: it is not a missing density effect or
// a missing stopping power but a wrong one, silently, in a direction nothing
// downstream can notice. So what must exist is checked once, here, where the
// material can be named.
void Material::check_electron_tables() const
{
  auto n_grid = data::brems_e_grid.size();
  auto require = [&](bool ok, const std::string& what) {
    if (!ok) {
      fatal_error("Charged-particle transport in material " +
                  std::to_string(id_) + " has no " + what +
                  ". This is a bug in the setup rather than a property of the "
                  "data; transporting on it would give a wrong answer without "
                  "saying so.");
    }
  };

  require(density_effect_.size() == n_grid, "density-effect table");
  for (int q = 0; q < 2; ++q) {
    require(
      collision_stopping_[q].size() == n_grid, "collision stopping power");
    require(gos_[q].hard.size() == n_grid, "oscillator cross section table");
    require(gos_[q].majorant.size() == n_grid, "oscillator majorant");
    require(gos_[q].cumulative.size() == n_grid * oscillator_.size(),
      "oscillator cumulative");
    require(gos_full_[q].hard.size() == n_grid, "ungrouped oscillator table");
    require(gos_full_[q].cumulative.size() == n_grid * oscillator_.size(),
      "ungrouped oscillator cumulative");
    // The whole rate cannot be under the discrete part of itself. They are
    // equal when nothing is grouped and the first is larger otherwise, so a
    // violation means the two tables were built from different models.
    for (int i = 0; i < n_grid; ++i) {
      require(gos_full_[q].hard(i) >= gos_[q].hard(i) * (1.0 - 1.0e-9),
        "a whole inelastic rate at least as large as its discrete part");
    }
  }

  // The grouped and discrete halves of the collision loss have to add up to
  // the stopping power the material has. They do so identically rather than
  // approximately -- the transfers the restricted Berger-Seltzer form leaves
  // out are exactly the ones the free binary cross section describes -- so
  // this is not a tolerance to be tuned but a statement that the two halves
  // are still the two halves of one thing. It is checked here, once per
  // material, because if it ever stops holding the transport quietly runs on
  // a stopping power that is neither.
  //
  // Only where there is a soft channel at all. The free binary cross section
  // has no lower limit of its own -- its stopping integral runs away
  // logarithmically as the transfer goes to zero -- so it can only ever
  // describe the part of the loss above a cutoff. What makes the total finite
  // is the binding, which enters through the mean excitation energy in the
  // Berger-Seltzer form and nowhere else. A vanishing cutoff therefore does
  // not mean "sample everything discretely"; it means this pair of
  // descriptions has nothing to say, which is why EGSnrc's own threshold is
  // never zero.
  for (int q = 0; q < 2; ++q) {
    ParticleType projectile =
      (q == 0) ? ParticleType::electron() : ParticleType::positron();
    for (int i = 0; i < n_grid; ++i) {
      double E = data::brems_e_grid(i);
      double total = collision_stopping_[q](i);
      if (!(total > 0.0))
        continue;

      // The oscillator model is what the transport runs on, and what has to
      // land on ICRU 37. It does so by construction -- the sum rules fix it,
      // and the compensation in init_oscillator_renorm() repairs what the
      // inner-shell renormalisation costs -- so a discrepancy here means one
      // of those has stopped holding rather than a tolerance wanting widened.
      //
      // Checked at the cutoff the transport will actually use, including zero:
      // every oscillator has a threshold of its own, so the model stays finite
      // there and single-event transport is held to the same total as a
      // grouped one.
      double w_cc = settings::deflection_cutoff > 0.0
                      ? soft_collision_cutoff(projectile, E)
                      : 0.0;
      auto gos = this->gos_collision_moments(q, E, w_cc);
      double gos_total = gos.s_soft + gos.s_hard;
      if (gos_total > 0.0 && std::abs(gos_total - total) > 1.0e-6 * total) {
        fatal_error(fmt::format(
          "The oscillator model of material {} gives a collision stopping "
          "power of {:.8g} b eV per electron at {:.4g} eV against the ICRU 37 "
          "value of {:.8g}. The model reproduces that total by construction, "
          "so this is a broken oscillator table rather than an approximation "
          "that wants loosening.",
          id_, gos_total, E, total));
      }

      if (!(w_cc > 0.0))
        continue;
      auto m = this->collision_moments(q, E, w_cc);

      // The split has a second limit, and it is the medium's rather than the
      // step's. The restricted stopping power counts the loss to transfers
      // under the cutoff, and once the cutoff falls below the mean excitation
      // energy there are no such transfers to count: the expression goes
      // negative, which is the formula saying it has left its domain rather
      // than a small number. It happens only within a factor of a few of the
      // transport cutoff, where the step's budget is a hundredth of a kinetic
      // energy that is itself close to the cutoff -- in carbon, below about
      // 1.1 keV. There is nothing to add up there, and a step that short
      // holds far too few collisions to be described by two moments anyway.
      if (!(m.s_soft > 0.0))
        continue;
      double sum = m.s_soft + m.s_hard;
      if (std::abs(sum - total) > 1.0e-9 * total) {
        fatal_error(fmt::format(
          "The collision loss of material {} does not add up at {:.4g} eV: "
          "{:.8g} soft plus {:.8g} hard against a total of {:.8g} b eV per "
          "electron. The two are meant to be complementary parts of one "
          "stopping power, so transporting on them would be wrong in a way "
          "nothing downstream could notice.",
          id_, E, m.s_soft, m.s_hard, total));
      }
    }
  }

  // The oscillator list the recoil model reads, one block per element with one
  // entry per subshell of it
  int n_block = oscillator_offset_.empty() ? 0 : oscillator_offset_.size() - 1;
  require(n_block == oscillator_element_.size(), "oscillator blocks");
  for (int b = 0; b < n_block; ++b) {
    const auto& elm = *data::elements[oscillator_element_[b]];
    require(oscillator_offset_[b + 1] - oscillator_offset_[b] ==
              elm.electron_shell_map_.size(),
      "a complete oscillator block for " + elm.name_);
  }
  for (int i = 0; i < nuclide_.size(); ++i) {
    require(oscillator_block_.count(element_[i]) > 0,
      "an oscillator block for " + data::elements[element_[i]]->name_);
  }
}

void Material::calculate_electron_xs(Particle& p) const
{
  double electron_density = 0.0;
  int q = p.type().is_positron() ? 1 : 0;

  // Add contribution from each nuclide in material
  for (int i = 0; i < nuclide_.size(); ++i) {
    // ========================================================================
    // CALCULATE MICROSCOPIC CROSS SECTION

    // Determine microscopic cross sections for this nuclide
    int i_element = element_[i];

    // Calculate microscopic cross section for this nuclide
    const auto& micro {p.electron_xs(i_element)};
    if (p.E() != micro.last_E || q != micro.last_q) {
      data::elements[i_element]->calculate_electron_xs(p);
    }

    // ========================================================================
    // ADD TO MACROSCOPIC CROSS SECTION

    // Copy atom density of nuclide in material
    double atom_density = this->atom_density(i, p.density_mult());

    // The atom's own channels: elastic, bremsstrahlung and annihilation
    p.macro_xs().total += atom_density * micro.total;
    p.macro_xs().step.hard += atom_density * micro.hard_total;
    p.macro_xs().step.hard_majorant += atom_density * micro.hard_majorant;
    p.macro_xs().step.soft_rate += atom_density * micro.soft_rate;
    p.macro_xs().step.stopping += atom_density * micro.soft_stopping;
    p.macro_xs().step.straggling += atom_density * micro.soft_straggling;
    p.macro_xs().step.xs1_soft += atom_density * micro.soft_xs1;
    p.macro_xs().step.xs2_soft += atom_density * micro.soft_xs2;

    electron_density += atom_density * data::elements[i_element]->Z_;
  }

  // The inelastic collisions, which belong to the medium rather than to any
  // atom in it: the oscillator strengths are shares of all its electrons and
  // the resonance energies are fixed by its mean excitation energy. Summing
  // the model over the oscillators is far too slow to do here, so what is
  // read is the tabulation init_gos_tables() left behind.
  //
  // The stopping power this adds, with the discrete collisions above the
  // cutoff, is the ICRU 37 collision stopping power exactly -- by the sum
  // rules the oscillators satisfy and the compensation that repairs what the
  // inner-shell renormalisation costs. check_electron_tables() refuses to run
  // if it is not.
  const auto& t = gos_[q];
  const auto& tf = gos_full_[q];
  auto n_grid = data::brems_e_grid.size();
  if (electron_density > 0.0 && t.hard.size() == n_grid &&
      tf.hard.size() == n_grid) {
    const auto& grid = data::brems_e_grid;
    double E = p.E();
    int i;
    double f;
    if (E <= grid(0)) {
      i = 0;
      f = 0.0;
    } else if (E >= grid(n_grid - 1)) {
      i = n_grid - 2;
      f = 1.0;
    } else {
      i = lower_bound_index(grid.cbegin(), grid.cend(), E);
      f = std::log(E / grid(i)) / std::log(grid(i + 1) / grid(i));
    }
    auto on_grid = [i, f](const tensor::Tensor<double>& v) {
      return v(i) + f * (v(i + 1) - v(i));
    };

    // Two rates, because a condensed-history run uses both. The discrete one
    // is what may end a grouped step; the whole one is what a flight that
    // declined to group -- and every flight of a single-event run -- is
    // transported on. They are equal when nothing is grouped.
    double hard = electron_density * on_grid(t.hard);
    double full = electron_density * on_grid(tf.hard);
    p.macro_xs().step.hard += hard;
    p.macro_xs().step.inelastic = hard;
    p.macro_xs().step.inelastic_full = full;
    // The larger of the two bracketing bounds rather than a blend, for the
    // same reason the element's channels take it that way: a blend of two
    // bounds is not a bound where the rate is concave between them.
    p.macro_xs().step.hard_majorant +=
      electron_density *
      std::max(on_grid(t.hard), std::max(t.majorant(i), t.majorant(i + 1)));
    p.macro_xs().step.stopping += electron_density * on_grid(t.stopping);
    p.macro_xs().step.straggling += electron_density * on_grid(t.straggling);
    p.macro_xs().step.xs1_soft += electron_density * on_grid(t.xs1);
    p.macro_xs().step.xs2_soft += electron_density * on_grid(t.xs2);

    // The total a flight is drawn from carries the whole inelastic channel.
    // Carrying the discrete rate instead would drop every transfer under the
    // cutoff from any flight that declined to group, with no grouped channel
    // left to supply them.
    p.macro_xs().total += full;

    // The rate of the collisions actually being grouped, which decides
    // whether grouping is worth it. Every oscillator has a threshold of its
    // own, so this is finite and is the difference of the two rates rather
    // than the proxy the discrete rate used to stand in as.
    p.macro_xs().step.soft_rate += std::max(0.0, full - hard);
  }
}

void Material::set_id(int32_t id)
{
  assert(id >= 0 || id == C_NONE);

  // Clear entry in material map if an ID was already assigned before
  if (id_ != C_NONE) {
    model::material_map.erase(id_);
    id_ = C_NONE;
  }

  // Make sure no other material has same ID
  if (model::material_map.find(id) != model::material_map.end()) {
    throw std::runtime_error {
      "Two materials have the same ID: " + std::to_string(id)};
  }

  // If no ID specified, auto-assign next ID in sequence
  if (id == C_NONE) {
    id = 0;
    for (const auto& m : model::materials) {
      id = std::max(id, m->id_);
    }
    ++id;
  }

  // Update ID and entry in material map
  id_ = id;
  model::material_map[id] = index_;
}

void Material::set_density(double density, const std::string& units)
{
  assert(density >= 0.0);

  if (nuclide_.empty()) {
    throw std::runtime_error {"No nuclides exist in material yet."};
  }

  if (units == "atom/b-cm") {
    // Set total density based on value provided
    density_ = density;

    // Determine normalized atom percents
    double sum_percent = atom_density_.sum();
    atom_density_ /= sum_percent;

    // Recalculate nuclide atom densities based on given density
    atom_density_ *= density;

    // Calculate density in g/cm^3 and charge density in [e/b-cm]
    density_gpcc_ = 0.0;
    charge_density_ = 0.0;
    for (int i = 0; i < nuclide_.size(); ++i) {
      int i_nuc = nuclide_[i];
      double awr = data::nuclides[i_nuc]->awr_;
      int z = settings::run_CE ? data::nuclides[i_nuc]->Z_ : 0.0;
      density_gpcc_ += atom_density_(i) * awr * MASS_NEUTRON / N_AVOGADRO;
      charge_density_ += atom_density_(i) * z;
    }
  } else if (units == "g/cm3" || units == "g/cc") {
    // Determine factor by which to change densities
    double previous_density_gpcc = density_gpcc_;
    double f = density / previous_density_gpcc;

    // Update densities
    density_gpcc_ = density;
    density_ *= f;
    atom_density_ *= f;
    charge_density_ *= f;
  } else {
    throw std::invalid_argument {
      "Invalid units '" + std::string(units.data()) + "' specified."};
  }
}

void Material::set_densities(
  const vector<std::string>& name, const vector<double>& density)
{
  auto n = name.size();
  assert(n > 0);
  assert(n == density.size());

  if (n != nuclide_.size()) {
    nuclide_.resize(n);
    atom_density_ = tensor::zeros<double>({n});
    if (settings::photon_transport)
      element_.resize(n);
  }

  double sum_density = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const auto& nuc {name[i]};
    if (data::nuclide_map.find(nuc) == data::nuclide_map.end()) {
      int err = openmc_load_nuclide(nuc.c_str(), nullptr, 0);
      if (err < 0)
        throw std::runtime_error {get_errmsg()};
    }

    nuclide_[i] = data::nuclide_map.at(nuc);
    assert(density[i] > 0.0);
    atom_density_(i) = density[i];
    sum_density += density[i];

    if (settings::photon_transport) {
      auto element_name = to_element(nuc);
      element_[i] = data::element_map.at(element_name);
    }
  }

  // Set total density to the sum of the vector
  this->set_density(sum_density, "atom/b-cm");

  // Generate material bremsstrahlung data for electrons and positrons
  if (settings::photon_transport &&
      settings::electron_treatment == ElectronTreatment::TTB) {
    this->init_bremsstrahlung();
  }

  // Build the oscillator model used by the inelastic angular partition, and
  // the transport cross sections derived from it. The oscillator energies are
  // built from the electron density and the plasma energy, so new densities
  // move them -- and anything tabulated from them has to move with them, or it
  // describes the material this one used to be.
  if (settings::electron_transport) {
    this->init_electron_oscillators();
  }

  // Assign S(a,b) tables
  this->init_thermal();
}

double Material::volume() const
{
  if (volume_ < 0.0) {
    throw std::runtime_error {
      "Volume for material with ID=" + std::to_string(id_) + " not set."};
  }
  return volume_;
}

double Material::temperature() const
{
  // If material doesn't have an assigned temperature, use global default
  return temperature_ >= 0 ? temperature_ : settings::temperature_default;
}

void Material::to_hdf5(hid_t group) const
{
  hid_t material_group = create_group(group, "material " + std::to_string(id_));

  write_attribute(material_group, "depletable", static_cast<int>(depletable()));
  if (volume_ > 0.0) {
    write_attribute(material_group, "volume", volume_);
  }
  if (temperature_ > 0.0) {
    write_attribute(material_group, "temperature", temperature_);
  }
  write_dataset(material_group, "name", name_);
  write_dataset(material_group, "atom_density", density_);

  // Copy nuclide/macro name for each nuclide to vector
  vector<std::string> nuc_names;
  vector<std::string> macro_names;
  vector<double> nuc_densities;
  if (settings::run_CE) {
    for (int i = 0; i < nuclide_.size(); ++i) {
      int i_nuc = nuclide_[i];
      nuc_names.push_back(data::nuclides[i_nuc]->name_);
      nuc_densities.push_back(atom_density_(i));
    }
  } else {
    for (int i = 0; i < nuclide_.size(); ++i) {
      int i_nuc = nuclide_[i];
      if (data::mg.nuclides_[i_nuc].awr != MACROSCOPIC_AWR) {
        nuc_names.push_back(data::mg.nuclides_[i_nuc].name);
        nuc_densities.push_back(atom_density_(i));
      } else {
        macro_names.push_back(data::mg.nuclides_[i_nuc].name);
      }
    }
  }

  // Write vector to 'nuclides'
  if (!nuc_names.empty()) {
    write_dataset(material_group, "nuclides", nuc_names);
    write_dataset(material_group, "nuclide_densities", nuc_densities);
  }

  // Write vector to 'macroscopics'
  if (!macro_names.empty()) {
    write_dataset(material_group, "macroscopics", macro_names);
  }

  if (!thermal_tables_.empty()) {
    vector<std::string> sab_names;
    for (const auto& table : thermal_tables_) {
      sab_names.push_back(data::thermal_scatt[table.index_table]->name_);
    }
    write_dataset(material_group, "sab_names", sab_names);
  }

  close_group(material_group);
}

void Material::export_properties_hdf5(hid_t group) const
{
  hid_t material_group = create_group(group, "material " + std::to_string(id_));
  write_attribute(material_group, "atom_density", density_);
  write_attribute(material_group, "mass_density", density_gpcc_);
  close_group(material_group);
}

void Material::import_properties_hdf5(hid_t group)
{
  hid_t material_group = open_group(group, "material " + std::to_string(id_));
  double density;
  read_attribute(material_group, "atom_density", density);
  this->set_density(density, "atom/b-cm");
  close_group(material_group);
}

void Material::add_nuclide(const std::string& name, double density)
{
  // Check if nuclide is already in material
  for (int i = 0; i < nuclide_.size(); ++i) {
    int i_nuc = nuclide_[i];
    if (data::nuclides[i_nuc]->name_ == name) {
      double awr = data::nuclides[i_nuc]->awr_;
      density_ += density - atom_density_(i);
      density_gpcc_ +=
        (density - atom_density_(i)) * awr * MASS_NEUTRON / N_AVOGADRO;
      atom_density_(i) = density;
      return;
    }
  }

  // If nuclide wasn't found, extend nuclide/density arrays
  int err = openmc_load_nuclide(name.c_str(), nullptr, 0);
  if (err < 0)
    throw std::runtime_error {get_errmsg()};

  // Append new nuclide/density
  int i_nuc = data::nuclide_map[name];
  nuclide_.push_back(i_nuc);

  // Append new element if photon transport is on
  if (settings::photon_transport) {
    int i_elem = data::element_map[to_element(name)];
    element_.push_back(i_elem);
  }

  auto n = nuclide_.size();

  // Create copy of atom_density_ array with one extra entry
  tensor::Tensor<double> atom_density = tensor::zeros<double>({n});
  atom_density.slice(tensor::range(0, n - 1)) = atom_density_;
  atom_density(n - 1) = density;
  atom_density_ = atom_density;

  density_ += density;
  density_gpcc_ +=
    density * data::nuclides[i_nuc]->awr_ * MASS_NEUTRON / N_AVOGADRO;
}

//==============================================================================
// Non-method functions
//==============================================================================

double sternheimer_adjustment(const vector<double>& f,
  const vector<double>& e_b_sq, double e_p_sq, double n_conduction,
  double log_I, double tol, int max_iter)
{
  // Get the total number of oscillators
  int n = f.size();

  // Calculate the Sternheimer adjustment factor using Newton's method
  double rho = 2.0;
  int iter;
  for (iter = 0; iter < max_iter; ++iter) {
    double rho_0 = rho;

    // Function to find the root of and its derivative
    double g = 0.0;
    double gp = 0.0;

    for (int i = 0; i < n; ++i) {
      // Square of resonance energy of a bound-shell oscillator
      double e_r_sq = e_b_sq[i] * rho * rho + 2.0 / 3.0 * f[i] * e_p_sq;
      g += f[i] * std::log(e_r_sq);
      gp += e_b_sq[i] * f[i] * rho / e_r_sq;
    }
    // Include conduction electrons
    if (n_conduction > 0.0) {
      g += n_conduction * std::log(n_conduction * e_p_sq);
    }

    // Set the next guess: rho_n+1 = rho_n - g(rho_n)/g'(rho_n)
    rho -= (g - 2.0 * log_I) / (2.0 * gp);

    // If the initial guess is too large, rho can be negative
    if (rho < 0.0)
      rho = rho_0 / 2.0;

    // Check for convergence
    if (std::abs(rho - rho_0) / rho_0 < tol)
      break;
  }
  // Did not converge
  if (iter >= max_iter) {
    warning("Maximum Newton-Raphson iterations exceeded.");
    rho = 1.0e-6;
  }
  return rho;
}

double berger_seltzer_spin_term(double tau, double delta_cut, bool positron)
{
  double gamma = tau + 1.0;
  double gamma_sq = gamma * gamma;
  double beta_sq = tau * (tau + 2.0) / gamma_sq;

  // A positron is distinguishable from the electron it strikes and may give
  // it everything; an electron is not, and the faster of the two leaving is
  // the one called the primary, so half is the most it can transfer
  double d_max = positron ? tau : 0.5 * tau;
  double d = std::min(delta_cut, d_max);
  if (!(d > 0.0))
    return 0.0;

  // Berger and Seltzer write the restricted term with the whole transfer
  // logarithm inside it, which is what makes ln((tau - d) d) appear; the
  // unrestricted forms this code has always used fold ln(tau^2/4) of that
  // into the leading logarithm instead. Both charges differ by exactly that,
  // so taking it back out here leaves a term that drops into the existing
  // stopping-power expression without touching anything else, and that equals
  // the old closed forms at the kinematic limit. The unit tests check both of
  // those rather than trusting them.
  double offset = std::log(0.25 * tau * tau);

  if (positron) {
    double t2 = tau + 2.0;
    double d2 = d * d;
    double d3 = d2 * d;
    double d4 = d3 * d;
    return std::log(tau * d) -
           (beta_sq / tau) *
             (tau + 2.0 * d - 1.5 * d2 / t2 - (d - d3 / 3.0) / (t2 * t2) -
               (0.5 * d2 - tau * d3 / 3.0 + 0.25 * d4) / (t2 * t2 * t2)) -
           offset;
  }

  return -1.0 - beta_sq + std::log((tau - d) * d) + tau / (tau - d) +
         (0.5 * d * d + (2.0 * tau + 1.0) * std::log1p(-d / tau)) / gamma_sq -
         offset;
}

double density_effect(const vector<double>& f, const vector<double>& e_b_sq,
  double e_p_sq, double n_conduction, double rho, double E, double tol,
  int max_iter)
{
  // Get the total number of oscillators
  int n = f.size();

  // Square of the ratio of the speed of light to the velocity of the charged
  // particle
  double beta_sq = E * (E + 2.0 * MASS_ELECTRON_EV) /
                   ((E + MASS_ELECTRON_EV) * (E + MASS_ELECTRON_EV));

  // For nonmetals, delta = 0 for beta < beta_0, where beta_0 is obtained by
  // setting the frequency w = 0.
  double beta_0_sq = 0.0;
  if (n_conduction == 0.0) {
    for (int i = 0; i < n; ++i) {
      beta_0_sq += f[i] * e_p_sq / (e_b_sq[i] * rho * rho);
    }
    beta_0_sq = 1.0 / (1.0 + beta_0_sq);
  }
  double delta = 0.0;
  if (beta_sq < beta_0_sq)
    return delta;

  // Compute the square of the frequency w^2 using Newton's method, with the
  // initial guess of w^2 equal to beta^2 * gamma^2
  double w_sq = E / MASS_ELECTRON_EV * (E / MASS_ELECTRON_EV + 2);
  int iter;
  for (iter = 0; iter < max_iter; ++iter) {
    double w_sq_0 = w_sq;

    // Function to find the root of and its derivative
    double g = 0.0;
    double gp = 0.0;

    for (int i = 0; i < n; ++i) {
      double c = e_b_sq[i] * rho * rho / e_p_sq + w_sq;
      g += f[i] / c;
      gp -= f[i] / (c * c);
    }
    // Include conduction electrons
    g += n_conduction / w_sq;
    gp -= n_conduction / (w_sq * w_sq);

    // Set the next guess: w_n+1 = w_n - g(w_n)/g'(w_n)
    w_sq -= (g + 1.0 - 1.0 / beta_sq) / gp;

    // If the initial guess is too large, w can be negative
    if (w_sq < 0.0)
      w_sq = w_sq_0 / 2.0;

    // Check for convergence
    if (std::abs(w_sq - w_sq_0) / w_sq_0 < tol)
      break;
  }
  // Did not converge
  if (iter >= max_iter) {
    warning("Maximum Newton-Raphson iterations exceeded: setting density "
            "effect correction to zero.");
    return delta;
  }

  // Solve for the density effect correction
  for (int i = 0; i < n; ++i) {
    double l_sq = e_b_sq[i] * rho * rho / e_p_sq + 2.0 / 3.0 * f[i];
    delta += f[i] * std::log((l_sq + w_sq) / l_sq);
  }
  // Include conduction electrons
  if (n_conduction > 0.0) {
    delta += n_conduction * std::log((n_conduction + w_sq) / n_conduction);
  }

  return delta - w_sq * (1.0 - beta_sq);
}

void read_materials_xml()
{
  write_message("Reading materials XML file...", 5);

  pugi::xml_document doc;

  // Check if materials.xml exists
  std::string filename = settings::path_input + "materials.xml";
  if (!file_exists(filename)) {
    fatal_error("Material XML file '" + filename + "' does not exist!");
  }

  // Parse materials.xml file and get root element
  doc.load_file(filename.c_str());

  // Loop over XML material elements and populate the array.
  pugi::xml_node root = doc.document_element();

  read_materials_xml(root);
}

void read_materials_xml(pugi::xml_node root)
{
  for (pugi::xml_node material_node : root.children("material")) {
    model::materials.push_back(make_unique<Material>(material_node));
  }
  model::materials.shrink_to_fit();
}

void free_memory_material()
{
  model::materials.clear();
  model::material_map.clear();
}

//==============================================================================
// C API
//==============================================================================

extern "C" int openmc_get_material_index(int32_t id, int32_t* index)
{
  auto it = model::material_map.find(id);
  if (it == model::material_map.end()) {
    set_errmsg("No material exists with ID=" + std::to_string(id) + ".");
    return OPENMC_E_INVALID_ID;
  } else {
    *index = it->second;
    return 0;
  }
}

extern "C" int openmc_material_add_nuclide(
  int32_t index, const char* name, double density)
{
  int err = 0;
  if (index >= 0 && index < model::materials.size()) {
    try {
      model::materials[index]->add_nuclide(name, density);
    } catch (const std::runtime_error& e) {
      return OPENMC_E_DATA;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return err;
}

extern "C" int openmc_material_get_densities(
  int32_t index, const int** nuclides, const double** densities, int* n)
{
  if (index >= 0 && index < model::materials.size()) {
    auto& mat = model::materials[index];
    if (!mat->nuclides().empty()) {
      *nuclides = mat->nuclides().data();
      *densities = mat->densities().data();
      *n = mat->nuclides().size();
      return 0;
    } else {
      set_errmsg("Material atom density array has not been allocated.");
      return OPENMC_E_ALLOCATE;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_get_density(int32_t index, double* density)
{
  if (index >= 0 && index < model::materials.size()) {
    auto& mat = model::materials[index];
    *density = mat->density_gpcc();
    return 0;
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_get_fissionable(int32_t index, bool* fissionable)
{
  if (index >= 0 && index < model::materials.size()) {
    *fissionable = model::materials[index]->fissionable();
    return 0;
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_get_id(int32_t index, int32_t* id)
{
  if (index >= 0 && index < model::materials.size()) {
    *id = model::materials[index]->id();
    return 0;
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_get_temperature(
  int32_t index, double* temperature)
{
  if (index < 0 || index >= model::materials.size()) {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  *temperature = model::materials[index]->temperature();
  return 0;
}

extern "C" int openmc_material_get_volume(int32_t index, double* volume)
{
  if (index >= 0 && index < model::materials.size()) {
    try {
      *volume = model::materials[index]->volume();
    } catch (const std::exception& e) {
      set_errmsg(e.what());
      return OPENMC_E_UNASSIGNED;
    }
    return 0;
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_set_density(
  int32_t index, double density, const char* units)
{
  if (index >= 0 && index < model::materials.size()) {
    try {
      model::materials[index]->set_density(density, units);
    } catch (const std::exception& e) {
      set_errmsg(e.what());
      return OPENMC_E_UNASSIGNED;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return 0;
}

extern "C" int openmc_material_set_densities(
  int32_t index, int n, const char** name, const double* density)
{
  if (index >= 0 && index < model::materials.size()) {
    try {
      model::materials[index]->set_densities(
        {name, name + n}, {density, density + n});
    } catch (const std::exception& e) {
      set_errmsg(e.what());
      return OPENMC_E_UNASSIGNED;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return 0;
}

extern "C" int openmc_material_set_id(int32_t index, int32_t id)
{
  if (index >= 0 && index < model::materials.size()) {
    try {
      model::materials.at(index)->set_id(id);
    } catch (const std::exception& e) {
      set_errmsg(e.what());
      return OPENMC_E_UNASSIGNED;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
  return 0;
}

extern "C" int openmc_material_get_name(int32_t index, const char** name)
{
  if (index < 0 || index >= model::materials.size()) {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *name = model::materials[index]->name().data();

  return 0;
}

extern "C" int openmc_material_set_name(int32_t index, const char* name)
{
  if (index < 0 || index >= model::materials.size()) {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  model::materials[index]->set_name(name);

  return 0;
}

extern "C" int openmc_material_set_volume(int32_t index, double volume)
{
  if (index >= 0 && index < model::materials.size()) {
    auto& m {model::materials[index]};
    if (volume >= 0.0) {
      m->volume_ = volume;
      return 0;
    } else {
      set_errmsg("Volume must be non-negative");
      return OPENMC_E_INVALID_ARGUMENT;
    }
  } else {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }
}

extern "C" int openmc_material_get_depletable(int32_t index, bool* depletable)
{
  if (index < 0 || index >= model::materials.size()) {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  *depletable = model::materials[index]->depletable();

  return 0;
}

extern "C" int openmc_material_set_depletable(int32_t index, bool depletable)
{
  if (index < 0 || index >= model::materials.size()) {
    set_errmsg("Index in materials array is out of bounds.");
    return OPENMC_E_OUT_OF_BOUNDS;
  }

  model::materials[index]->depletable() = depletable;

  return 0;
}

extern "C" int openmc_extend_materials(
  int32_t n, int32_t* index_start, int32_t* index_end)
{
  if (index_start)
    *index_start = model::materials.size();
  if (index_end)
    *index_end = model::materials.size() + n - 1;
  for (int32_t i = 0; i < n; i++) {
    model::materials.push_back(make_unique<Material>());
  }
  return 0;
}

extern "C" size_t n_materials()
{
  return model::materials.size();
}

} // namespace openmc
