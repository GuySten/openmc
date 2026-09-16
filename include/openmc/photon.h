#ifndef OPENMC_PHOTON_H
#define OPENMC_PHOTON_H

#include "openmc/array.h"
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
//! Interaction data for a single element
//!
//! Holds the photoatomic cross sections and atomic relaxation data, the
//! Compton profiles, the stopping powers and scaled bremsstrahlung cross
//! sections used by the thick-target approximation, and -- when electron
//! transport is enabled -- the electron interaction data as well.
//==============================================================================

class ElectronSubshell {
public:
  struct Transition {
    int primary_subshell;   //!< Index in shells_ of originating subshell
    int secondary_subshell; //!< Index in shells_ of Auger electron subshell
    double energy;          //!< Energy of transition
    double probability;     //!< Probability of transition between subshells
  };

  // Constructors
  ElectronSubshell() {};

  int index_subshell; //!< index in SUBSHELLS
  int threshold;
  double binding_energy;
  double num_electrons {0.0}; //!< occupancy, needed by Bhabha scattering
  vector<Transition> transitions;
};

class Element {
public:
  // Constructors/destructor
  Element(hid_t group);
  ~Element();

  // Methods
  void calculate_xs(Particle& p) const;

  void compton_scatter(double alpha, bool doppler, double* alpha_out,
    double* mu, int* i_shell, uint64_t* seed) const;

  double rayleigh_scatter(double alpha, uint64_t* seed) const;

  void pair_production(double alpha, double* E_electron, double* E_positron,
    double* mu_electron, double* mu_positron, uint64_t* seed) const;

  void atomic_relaxation(int i_shell, Particle& p) const;

  //! Read the electron interaction data for this element.
  //
  //! Called only when electron transport is enabled, from a group in the
  //! electron library rather than the photoatomic one. Everything it fills is
  //! left empty otherwise, which costs a few hundred bytes per element and no
  //! heap at all.
  void read_electron_data(hid_t group);

  void calculate_electron_xs(Particle& p) const;

  //! Sample an elastic deflection. \param q_index 0 for an electron, 1 for a
  //! positron; the two differ little in rate and a great deal in first moment
  double elastic_scatter(int q_index, double E, uint64_t* seed) const;

  double excitation(double E) const;

  //! Electroionization: Moller scattering for an electron, Bhabha for a
  //! positron. Returns false when a positron's sampled transfer is rejected,
  //! which leaves the particle untouched -- see compute_moller_majorant().
  bool ionization(Particle& p, int i_shell) const;

  int sample_ionization_shell(Particle& p) const;

  void bremsstrahlung(Particle& p) const;

  //! Bhabha scattering above the Moller kinematic limit
  //
  //! A Moller collision cannot transfer more than half of what is left after
  //! the binding energy is paid, so the evaluated knock-on spectra stop there
  //! and a positron's larger transfers are simply absent from them. This is
  //! that missing range, which lies far above every binding energy and is
  //! therefore free-electron territory.
  void bhabha(Particle& p, int i_shell) const;

  //! Sample the subshell in which such a collision occurs
  int sample_bhabha_shell(Particle& p) const;

  //! Emit the knock-on electron and deflect the projectile, for a transfer of
  //! W out of which the atom keeps the binding energy e_b, the collision
  //! having handed the atom a recoil energy Q
  //
  //! Both polar angles follow from Q alone: the projectile is deflected
  //! through the momentum transfer and the knock-on leaves along it. Setting
  //! Q = W recovers the free binary collision, in which the two are the
  //! familiar Moller pair.
  void emit_knock_on(Particle& p, double W, double e_b, double Q) const;

  //! Sample the recoil energy of an inelastic collision transferring W out of
  //! subshell i_shell
  //
  //! A free electron takes up the whole transfer, Q = W. A bound one does not:
  //! while the momentum transfer stays below the scale of the subshell's
  //! oscillator the atom is excited as a whole, through a dipole-like
  //! interaction whose recoil is far smaller than the binary value and which
  //! PENELOPE splits into a longitudinal part, distributed as 1/(Q(Q+2mc^2)),
  //! and a transverse part that carries no momentum at all. Above that scale
  //! the subshell responds as a free electron and Q = W again.
  double sample_recoil(Particle& p, int i_shell, double W) const;

  //! Two-photon annihilation of a positron in flight
  //
  //! The positron annihilates with a bound electron, which is taken to be free
  //! and at rest, and is replaced by two photons sharing its kinetic energy
  //! plus both rest masses. Unlike annihilation at the transport cutoff, the
  //! photons are not 511 keV: they carry up to T + m_e c^2 each.
  void annihilation(Particle& p) const;

  //! Cross section for in-flight annihilation, per atom, in [b]
  //
  //! Heitler's two-photon result for a free electron at rest, multiplied by
  //! the Z electrons of the atom. It is a closed form in the incident energy
  //! alone, so it is evaluated exactly at the energy wanted rather than
  //! tabulated on the grid and interpolated.
  double annihilation_xs(double E) const;

  //! Sample the energy of a bremsstrahlung photon from the scaled cross
  //! sections of the photon library, above the threshold the electron
  //! library's cross section was integrated from. Returns zero when the
  //! incident energy leaves no room above that threshold.
  double sample_bremsstrahlung_energy(double E, uint64_t* seed) const;

  // Data members
  std::string name_; //!< Name of element, e.g. "Zr"
  int Z_;            //!< Atomic number
  int64_t index_;    //!< Index in global elements vector

  // Microscopic cross sections
  tensor::Tensor<double> energy_;
  tensor::Tensor<double> coherent_;
  tensor::Tensor<double> incoherent_;
  tensor::Tensor<double> photoelectric_total_;
  tensor::Tensor<double> pair_production_total_;
  tensor::Tensor<double> pair_production_electron_;
  tensor::Tensor<double> pair_production_nuclear_;
  tensor::Tensor<double> heating_;

  // Form factors
  Tabulated1D incoherent_form_factor_;
  Tabulated1D coherent_int_form_factor_;
  Tabulated1D coherent_anomalous_real_;
  Tabulated1D coherent_anomalous_imag_;

  // Photoionization and atomic relaxation data. Subshell cross sections are
  // stored separately to improve memory access pattern when calculating the
  // total cross section
  vector<ElectronSubshell> shells_;
  tensor::Tensor<double> cross_sections_;

  // Compton profile data
  tensor::Tensor<double> profile_pdf_;
  tensor::Tensor<double> profile_cdf_;
  tensor::Tensor<double> profile_tail_slope_;
  tensor::Tensor<double> profile_negative_mass_; //!< Mass from -1/alpha to 0
  tensor::Tensor<double> binding_energy_;
  tensor::Tensor<double> electron_pdf_;

  // Map subshells from Compton profile data obtained from Biggs et al,
  // "Hartree-Fock Compton profiles for the elements" to ENDF/B atomic
  // relaxation data
  tensor::Tensor<int> subshell_map_;

  // Stopping power data
  double I_ {0.0}; // mean excitation energy
  tensor::Tensor<int> n_electrons_;
  tensor::Tensor<double> ionization_energy_;
  tensor::Tensor<double> stopping_power_radiative_;

  // Bremsstrahlung scaled DCS
  tensor::Tensor<double> dcs_;

  // Whether atomic relaxation data is present
  bool has_atomic_relaxation_ {false};

  //============================================================================
  // Electron interaction data
  //
  // Empty unless electron transport is enabled; see read_electron_data(). The
  // energy grid is the electron library's own and is not the photon grid
  // above, so it is named separately.

  //! For each electroionization subshell, the index of the matching subshell in
  //! shells_. The two lists are not guaranteed to have the same length or
  //! ordering, so they are matched by ENDF designator rather than by position.
  vector<int> electron_shell_map_;

  tensor::Tensor<double> electron_energy_;
  //! Elastic cross sections and angular distributions, indexed by projectile
  //! charge: 0 for an electron, 1 for a positron
  array<tensor::Tensor<double>, 2> elastic_;
  array<AngleDistribution, 2> elastic_angle_;
  tensor::Tensor<double> electroionization_;
  vector<unique_ptr<ContinuousTabular>> ionization_dist_;
  //! Bhabha cross section above the Moller limit, per subshell, on
  //! electron_energy_. Filled by compute_bhabha_xs(), used only for positrons.
  tensor::Tensor<double> bhabha_;

  //! Largest value the Bhabha-to-Moller ratio takes at each grid energy.
  //! Filled by compute_moller_majorant(), used only for positrons.
  tensor::Tensor<double> moller_majorant_;

  tensor::Tensor<double> excitation_;
  Tabulated1D excitation_energy_loss_;
  tensor::Tensor<double> electron_bremsstrahlung_;

  //! Lowest emitted photon energy the bremsstrahlung cross section above was
  //! integrated from; the spectrum has to be sampled above the same one
  double bremsstrahlung_photon_cutoff_ {0.0};

  // Constant data
  static constexpr int MAX_STACK_SIZE =
    7; //!< maximum possible size of atomic relaxation stack
private:
  //! Integrate the free Bhabha differential cross section over the transfers
  //! the evaluated knock-on spectra cannot reach, for every subshell and every
  //! point of the electron energy grid
  void compute_bhabha_xs();

  //! Tabulate the largest Bhabha-to-Moller ratio at each grid energy, which is
  //! the factor by which a positron's electroionization cross section is
  //! raised to make it a majorant of the true one
  void compute_moller_majorant();

  struct ShellKinematics {
    double pz_max;       //!< Upper bound in Kaltiaisenaho Eq. (3.73)
    double c_limit;      //!< Half-profile integral K_i(|pz_max|), Eq. (3.117)
    double profile_mass; //!< Accessible profile mass, Eq. (3.118)
  };

  void compton_doppler(
    double alpha, double mu, double* E_out, int* i_shell, uint64_t* seed) const;

  //! Determine pz_max and the accessible profile mass (Eqs. 3.73, 3.118)
  ShellKinematics compton_shell_kinematics(
    double alpha, double mu, double E, int i_shell) const;

  //! Sample pz and E' for a selected shell (Eqs. 3.120-3.127)
  bool sample_compton_momentum(double alpha, double mu, double E, int i_shell,
    const ShellKinematics& kinematics, double* E_out, uint64_t* seed) const;

  //! Sample from the shell PMF in Kaltiaisenaho Eq. (3.116)
  bool compton_doppler_conditional(double alpha, double mu, double E,
    double* E_out, int* i_shell, uint64_t* seed) const;

  //! Evaluate K_i(pz), the normalized half-profile integral (Eq. 3.117)
  double compton_profile_cdf(int i_shell, double pz) const;

  //! Invert K_i using the inverse transforms of Eqs. (3.123) and (3.126)
  double invert_compton_profile_cdf(int i_shell, double c) const;

  //! Calculate the maximum size of the vacancy stack in atomic relaxation
  //
  //! These helper functions use the subshell transition data to calculate the
  //! maximum size the stack of unprocessed subshell vacancies can grow to for
  //! the given element while simulating the cascade of photons and electrons
  //! in atomic relaxation.
  int calc_max_stack_size() const;
  int calc_helper(std::unordered_map<int, int>& visited, int i_shell) const;
};

//==============================================================================
// Non-member functions
//==============================================================================

namespace detail {

//! Integrate an exponentially extrapolated Compton-profile tail
double compton_profile_tail_integral(
  double pz, double pz_last, double profile_last, double slope);

//! Invert an exponentially extrapolated Compton-profile tail integral
double invert_compton_profile_tail(
  double integral, double pz_last, double profile_last, double slope);

//! Calculate the outgoing-to-incident energy ratio for signed electron momentum
double compton_energy_ratio(double alpha, double mu, double pz);

} // namespace detail

std::pair<double, double> klein_nishina(double alpha, uint64_t* seed);

void free_memory_photon();

//==============================================================================
// Global variables
//==============================================================================

namespace data {

extern tensor::Tensor<double>
  compton_profile_pz; //! Compton profile momentum grid

//! Grids of the Seltzer-Berger scaled bremsstrahlung cross sections, read when
//! electron transport is enabled: incident electron kinetic energies in [eV]
//! and reduced photon energies kappa = k/T
extern tensor::Tensor<double> brems_e_grid;
extern tensor::Tensor<double> brems_k_grid;

//! Interaction data for each element
extern std::unordered_map<std::string, int> element_map;
extern vector<unique_ptr<Element>> elements;

} // namespace data

} // namespace openmc

#endif // OPENMC_PHOTON_H
