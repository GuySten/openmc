#ifndef OPENMC_MATERIAL_H
#define OPENMC_MATERIAL_H

#include <string>
#include <unordered_map>

#include "openmc/span.h"
#include "openmc/tensor.h"
#include "pugixml.hpp"
#include <hdf5.h>

#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/memory.h" // for unique_ptr
#include "openmc/ncrystal_interface.h"
#include "openmc/particle.h"
#include "openmc/settings.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

class Material;

namespace model {

extern std::unordered_map<int32_t, int32_t> material_map;
extern vector<unique_ptr<Material>> materials;

} // namespace model

//==============================================================================
//! A substance with constituent nuclides and thermal scattering data
//==============================================================================

class Material {
public:
  //----------------------------------------------------------------------------
  // Types
  struct ThermalTable {
    int index_table;   //!< Index of table in data::thermal_scatt
    int index_nuclide; //!< Index in nuclide_
    double fraction;   //!< How often to use table
  };

  //----------------------------------------------------------------------------
  // Constructors, destructors, factory functions
  Material() {};
  explicit Material(pugi::xml_node material_node);
  ~Material();

  //----------------------------------------------------------------------------
  // Methods

  void calculate_xs(Particle& p) const;

  //! Assign thermal scattering tables to specific nuclides within the material
  //! so the code knows when to apply bound thermal scattering data
  void init_thermal();

  //! Set up mapping between global nuclides vector and indices in nuclide_
  void init_nuclide_index();

  //! Finalize the material, assigning tables, normalize density, etc.
  void finalize();

  //! Write material data to HDF5
  void to_hdf5(hid_t group) const;

  //! Export physical properties to HDF5
  //! \param[in] group  HDF5 group to write to
  void export_properties_hdf5(hid_t group) const;

  //! Import physical properties from HDF5
  //! \param[in] group  HDF5 group to read from
  void import_properties_hdf5(hid_t group);

  //! Add nuclide to the material
  //
  //! \param[in] nuclide Name of the nuclide
  //! \param[in] density Density of the nuclide in [atom/b-cm]
  void add_nuclide(const std::string& nuclide, double density);

  //! Set atom densities for the material
  //
  //! \param[in] name Name of each nuclide
  //! \param[in] density Density of each nuclide in [atom/b-cm]
  void set_densities(
    const vector<std::string>& name, const vector<double>& density);

  //! Clone the material by deep-copying all members, except for the ID,
  //  which will get auto-assigned to the next available ID. After creating
  //  the new material, it is added to openmc::model::materials.
  //! \return reference to the cloned material
  Material& clone();

  //----------------------------------------------------------------------------
  // Accessors

  //! Get the atom density in [atom/b-cm]
  //! \return Density in [atom/b-cm]
  double atom_density(int32_t i, double rho_multiplier = 1.0) const
  {
    return atom_density_(i) * rho_multiplier;
  }

  //! Get density in [atom/b-cm]
  //! \return Density in [atom/b-cm]
  double density() const { return density_; }

  //! Get density in [g/cm^3].
  //! \return Density in [g/cm^3]
  double density_gpcc() const
  {
    return settings::run_CE ? density_gpcc_ : density();
  }

  //! Get charge density in [e/b-cm]
  //! \return Charge density in [e/b-cm]
  double charge_density() const { return charge_density_; };

  //! Get name
  //! \return Material name
  const std::string& name() const { return name_; }

  //! Set name
  void set_name(const std::string& name) { name_ = name; }

  //! Set total density of the material
  //
  //! \param[in] density Density value
  //! \param[in] units Units of density
  void set_density(double density, const std::string& units);

  //! Set temperature of the material
  void set_temperature(double temperature) { temperature_ = temperature; };

  //! Get nuclides in material
  //! \return Indices into the global nuclides vector
  span<const int> nuclides() const
  {
    return {nuclide_.data(), nuclide_.size()};
  }

  //! Get densities of each nuclide in material
  //! \return Densities in [atom/b-cm]
  span<const double> densities() const
  {
    return {atom_density_.data(), atom_density_.size()};
  }

  //! Get ID of material
  //! \return ID of material
  int32_t id() const { return id_; }

  //! Assign a unique ID to the material
  //! \param[in] Unique ID to assign. A value of -1 indicates that an ID
  //!   should be automatically assigned.
  void set_id(int32_t id);

  //! Get whether material is fissionable
  //! \return Whether material is fissionable
  bool fissionable() const { return fissionable_; }
  bool& fissionable() { return fissionable_; }

  //! Get volume of material
  //! \return Volume in [cm^3]
  double volume() const;

  //! Get temperature of material
  //! \return Temperature in [K]
  double temperature() const;

  //! Whether or not the material is depletable
  bool depletable() const { return depletable_; }
  bool& depletable() { return depletable_; }

  //! Get pointer to NCrystal material object
  //! \return Pointer to NCrystal material object
  const NCrystalMat& ncrystal_mat() const { return ncrystal_mat_; };

  //! Density-effect correction at kinetic energy \p E in [eV], interpolated
  //! on data::brems_e_grid, which is the grid electron transport populates.
  //! Zero unless init_electron_oscillators() has run.
  double density_effect_correction(double E) const;

  //! Collision stopping power this material must reproduce, per electron
  //!
  //! The ICRU 37 (Berger-Seltzer) value, built from the mean excitation energy
  //! the oscillator model was solved against and the density-effect correction
  //! tabulated beside it. The evaluated electroionization spectra do not
  //! integrate to it -- in copper at 16 MeV they fall 8.8 per cent short of
  //! the free atom, which the Sternheimer screening then more than accounts
  //! for -- so the grouped channel is held to this instead. It is the same
  //! quantity EGSnrc restricts and ESTAR tabulates, which is what makes the
  //! three comparable at all.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \return Stopping power in [b eV], to be multiplied by the material's
  //!   electron density in [1/(b cm)]
  double collision_stopping_power(int q_index, double E) const;

  //! What a collision costs, split at the soft cutoff, per electron
  //!
  //! The two halves are not independent. Below the cutoff the loss is the
  //! Berger-Seltzer stopping power restricted to those transfers; above it,
  //! it is the free binary cross section over the rest of the range; and the
  //! transfers the first leaves out are exactly the ones the second
  //! describes, so they sum to the unrestricted ICRU 37 total at every energy
  //! and every cutoff. That is what lets a mixed scheme group the first and
  //! sample the second without the total drifting away from the stopping
  //! power the material has. EGSnrc is built the same way, and PENELOPE
  //! reaches the same place from a single oscillator model.
  //!
  //! Everything here is evaluated rather than interpolated, apart from the
  //! density effect, which needs a Newton solve and is tabulated. Two
  //! interpolations of the two halves on different grids would leave the sum
  //! right only where the grids fall.
  struct CollisionMoments {
    double s_soft {0.0};  //!< restricted stopping power in [b eV]
    double w2_soft {0.0}; //!< second moment of the restricted loss in [b eV^2]
    double xs_hard {0.0}; //!< hard binary cross section in [b]
    double s_hard {0.0};  //!< stopping power the hard channel carries [b eV]
  };

  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \param[in] w_cc Soft cutoff in [eV]; zero groups nothing
  CollisionMoments collision_moments(int q_index, double E, double w_cc) const;

  //! Density-effect correction actually applied to collisions, per charge
  //!
  //! The Sternheimer correction less the amount by which the evaluated
  //! spectra already fall short of the free atom's stopping power, floored at
  //! zero. Taking the whole correction off data that never reached the free
  //! atom's value removes the same strength twice. See init_inelastic_transport
  //! for what this buys and what it assumes.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  double screening_correction(int q_index, double E) const;

  //! Resonance energy in [eV] of the Sternheimer-Liljequist oscillator
  //! standing for electroionization subshell \p i_shell of the element with
  //! global index \p i_element. Returns zero when this material carries no
  //! oscillator data, which closes the distant channel and leaves every
  //! collision a close one.
  double oscillator_energy(int i_element, int i_shell) const;

  //----------------------------------------------------------------------------
  // Data
  int32_t id_ {C_NONE};                 //!< Unique ID
  std::string name_;                    //!< Name of material
  vector<int> nuclide_;                 //!< Indices in nuclides vector
  vector<int> element_;                 //!< Indices in elements vector
  NCrystalMat ncrystal_mat_;            //!< NCrystal material object
  tensor::Tensor<double> atom_density_; //!< Nuclide atom density in [atom/b-cm]
  double density_;                      //!< Total atom density in [atom/b-cm]
  double density_gpcc_;                 //!< Total atom density in [g/cm^3]
  double charge_density_;               //!< Total charge density in [e/b-cm]
  double volume_ {-1.0};                //!< Volume in [cm^3]
  vector<bool> p0_; //!< Indicate which nuclides are to be treated with
                    //!< iso-in-lab scattering

  // To improve performance of tallying, we store an array (direct address
  // table) that indicates for each nuclide in data::nuclides the index of the
  // corresponding nuclide in the nuclide_ vector. If it is not present in the
  // material, the entry is set to -1.
  vector<int> mat_nuclide_index_;

  // Thermal scattering tables
  vector<ThermalTable> thermal_tables_;

  unique_ptr<Bremsstrahlung> ttb_;

  //----------------------------------------------------------------------------
  // Sternheimer-Liljequist oscillator data
  //
  // Empty unless electron transport is enabled; see
  // init_electron_oscillators(). The oscillators are what PENELOPE's
  // generalized oscillator strength model is built on, and they are used here
  // to split an inelastic collision into a distant and a close one.

  //! Density-effect correction tabulated on data::ttb_e_grid
  tensor::Tensor<double> density_effect_;
  //! Berger-Seltzer collision stopping power of this material per electron,
  //! in [b eV], on data::brems_e_grid, one table per projectile charge. See
  //! collision_stopping_power().
  array<tensor::Tensor<double>, 2> collision_stopping_;
  //! Natural log of the mean excitation energy in [eV], from the oscillator
  //! table. collision_moments() needs it at every cross section lookup.
  double log_I_ {0.0};
  //! The share of the density effect the evaluated data can absorb, per
  //! projectile charge, on data::brems_e_grid. See screening_correction().
  array<tensor::Tensor<double>, 2> screening_;

  //! Oscillator resonance energies in [eV], one per electroionization subshell,
  //! concatenated over the distinct elements of this material
  vector<double> oscillator_energy_;
  //! Global element index of each block of oscillator_energy_, and the
  //! reverse lookup the transport uses -- an inelastic collision asks for a
  //! resonance energy by element index, and a linear scan would be paid for on
  //! every one of them
  vector<int> oscillator_element_;
  std::unordered_map<int, int> oscillator_block_;
  //! Oscillator block of each global element index, or -1. Same content as
  //! the map above, laid out flat: sample_recoil() asks for a resonance
  //! energy by element index once per inelastic collision, which is too often
  //! to hash for it.
  vector<int> element_block_;
  //! Oscillator block of each entry in nuclide_, or -1. The map above is what
  //! builds this; the transport reads this, because a hash lookup per element
  //! per cross section evaluation is not free.
  vector<int> nuclide_block_;
  //! Start of each block in oscillator_energy_, with a trailing end marker
  vector<int> oscillator_offset_;

  //! First transport cross section of the grouped inelastic collisions, in
  //! [b], one table per block of oscillator_energy_ and per projectile charge,
  //! on that element's electron energy grid. It lives here rather than in
  //! Element because the recoil model that sets the deflection cuts between
  //! close and distant collisions at an oscillator energy of the material's,
  //! and because the density effect enters the same cut. Empty unless a run
  //! asked for condensed history.
  array<vector<tensor::Tensor<double>>, 2> inelastic_xs1_;
  //! Stopping power and straggling the density effect screens out of the
  //! grouped inelastic channel, per oscillator block and projectile charge.
  //! Tabulated per material because the screening is, the Sternheimer
  //! correction belonging to the medium and not to the atom.
  array<vector<tensor::Tensor<double>>, 2> inelastic_soft_screened_s_;
  array<vector<tensor::Tensor<double>>, 2> inelastic_soft_screened_w2_;
  //! Collision stopping power the evaluated data delivers after screening,
  //! per oscillator block and charge, in [b eV] on the element's grid
  array<vector<tensor::Tensor<double>>, 2> inelastic_total_s_;

  //! First transport cross section of the grouped inelastic collisions of one
  //! element of this material, in [b]
  //!
  //! \param[in] i_element Global element index
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! First transport cross section of the grouped inelastic deflections
  //!
  //! Takes the grid position the caller already computed rather than the
  //! energy: this runs for every element on every charged-particle cross
  //! section lookup, and Element::calculate_electron_xs has just searched that
  //! same grid for the same energy.
  //!
  //! \param[in] i_nuclide Index into this material's nuclide list
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] i_grid Index on the element's electron energy grid
  //! \param[in] f Interpolation factor on that interval
  //! Grouped stopping power and straggling the density effect screens away
  //!
  //! Both are per atom of the nuclide, as the element's own restricted moments
  //! are, so the caller weights them by the same atom density.
  //!
  //! \param[in] i_nuclide Index into this material's nuclide list
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] i_grid Index on the element's electron energy grid
  //! \param[in] f Interpolation factor on that grid
  //! \param[out] s Screened stopping power in [b eV]
  //! \param[out] w2 Screened second moment in [b eV^2]
  void inelastic_soft_screened(int i_nuclide, int q_index, int i_grid, double f,
    double& s, double& w2) const;

  //! Collision stopping power the evaluated data delivers, per atom, in [b eV]
  double inelastic_total_s(
    int i_nuclide, int q_index, int i_grid, double f) const;

  double inelastic_transport_xs(
    int i_nuclide, int q_index, int i_grid, double f) const;

private:
  //----------------------------------------------------------------------------
  // Private methods

  //! Parameters of the Sternheimer-Liljequist oscillator model of this
  //! material, shared by the density-effect correction and by the distant /
  //! close partition of inelastic collisions
  struct OscillatorTable {
    vector<double> f;        //!< strengths, normalized to one electron
    vector<double> e_b_sq;   //!< squared binding energies in [eV^2]
    double e_p_sq;           //!< squared plasma energy in [eV^2]
    double n_conduction;     //!< conduction electrons per electron
    double log_I;            //!< log of the mean excitation energy
    double electron_density; //!< in [electron/b-cm]
    double rho;              //!< Sternheimer adjustment factor
  };

  //! Build the oscillator table of this material
  OscillatorTable oscillator_table() const;

  //! Calculate the collision stopping power
  void collision_stopping_power(double* s_col, bool positron);

  //! Tabulate the density-effect correction and build an oscillator for each
  //! electroionization subshell
  void init_electron_oscillators();

  //! Tabulate the transport cross section of the grouped inelastic collisions
  void init_inelastic_transport();

  //! Refuse to transport charged particles on tables that were not built
  //
  //! Called once at the end of init_inelastic_transport(). An accessor that
  //! cannot find a table returns zero, which is what single-event transport
  //! wants; the same zero from a table that should exist would be a wrong
  //! density effect or a wrong stopping power rather than a missing one, and
  //! nothing downstream could notice.
  void check_electron_tables() const;

  //! Initialize bremsstrahlung data
  void init_bremsstrahlung();

  //! Normalize density
  void normalize_density();

  void calculate_neutron_xs(Particle& p) const;
  void calculate_photon_xs(Particle& p) const;
  void calculate_electron_xs(Particle& p) const;

  //----------------------------------------------------------------------------
  // Private data members
  int64_t index_;

  bool depletable_ {false}; //!< Is the material depletable?
  bool fissionable_ {
    false}; //!< Does this material contain fissionable nuclides
  //! \brief Default temperature for cells containing this material.
  //!
  //! A negative value indicates no default temperature was specified.
  double temperature_ {-1};
};

//==============================================================================
// Non-member functions
//==============================================================================

//! Calculate Sternheimer adjustment factor
double sternheimer_adjustment(const vector<double>& f,
  const vector<double>& e_b_sq, double e_p_sq, double n_conduction,
  double log_I, double tol, int max_iter);

//! Calculate density effect correction
double density_effect(const vector<double>& f, const vector<double>& e_b_sq,
  double e_p_sq, double n_conduction, double rho, double E, double tol,
  int max_iter);

//! Spin term of the Berger-Seltzer collision stopping power, restricted to
//! energy transfers below a cutoff
//!
//! This is the \f$F^\pm\f$ of ICRU 37, in the form Berger and Seltzer give it
//! for a restricted stopping power: the collision loss counting only transfers
//! under \f$\Delta\f$. The stopping power itself is
//!
//! \f[ S = \frac{2\pi r_e^2 m c^2}{\beta^2} n_e \left[
//!     \ln\frac{\tau^2(\tau+2)}{2(I/mc^2)^2} + F^\pm(\tau, \Delta)
//!     - \delta \right]. \f]
//!
//! What makes it worth having in this form is an identity rather than a
//! convenience. The transfers it leaves out are exactly those the free binary
//! cross section describes, so
//!
//! \f[ F^\pm(\tau, \Delta_{max}) - F^\pm(\tau, \Delta)
//!     = \int_{\Delta}^{\Delta_{max}} \varepsilon
//!       \frac{d\sigma_{M,B}}{d\varepsilon} d\varepsilon \f]
//!
//! holds identically -- verified to machine precision in the unit tests
//! against moller_moment() and bhabha_moment(). A scheme that takes its soft
//! collision loss from this and its hard collisions from the free cross
//! section above the same cutoff therefore reproduces the ICRU 37 total
//! exactly, at every energy and every cutoff, with nothing to calibrate. This
//! is how EGSnrc is built (PEGS4's SPIONB), and the two forms agree term by
//! term.
//!
//! \param[in] tau Kinetic energy in units of the electron rest mass
//! \param[in] delta_cut Largest transfer counted, in the same units. The
//!   unrestricted stopping power is \f$\tau/2\f$ for an electron and
//!   \f$\tau\f$ for a positron; larger values are clamped to those.
//! \param[in] positron Whether the projectile is a positron
//! \return \f$F^\pm(\tau, \Delta)\f$, dimensionless
double berger_seltzer_spin_term(double tau, double delta_cut, bool positron);

//! Read material data from materials.xml
void read_materials_xml();

//! Read material data XML node
//! \param[in] root node of materials XML element
void read_materials_xml(pugi::xml_node root);

void free_memory_material();

} // namespace openmc
#endif // OPENMC_MATERIAL_H
