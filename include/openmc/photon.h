#ifndef OPENMC_PHOTON_H
#define OPENMC_PHOTON_H

#include "openmc/array.h"
#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"
#include "openmc/electroionization.h"
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

//==============================================================================
//! Soft/hard split of elastic scattering at one energy
//!
//! A mixed (class II) condensed-history step is bounded by a hard elastic
//! collision and carries the soft ones as a single artificial deflection. The
//! first transport cross section of the soft part sets the size of that
//! deflection, the second its shape.
//==============================================================================
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

  //! Sample a hard elastic deflection, the part a step did not group
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \param[inout] seed pseudorandom number seed pointer
  double elastic_scatter_hard(int q_index, double E, double xs_elastic,
    double xs_hard, uint64_t* seed) const;

  //! Elastic cross section in [b] at one energy
  double elastic_xs(int q_index, double E) const;

  //! Collision stopping power the hard inelastic channel carries, per atom
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \return Stopping power in [b eV]
  double inelastic_hard_stopping(int q_index, double E) const;

  //! Collision stopping power of the evaluated data, unscreened, per atom
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \return Stopping power in [b eV]
  double inelastic_unscreened_stopping(int q_index, double E) const;

  //! First transport cross section of the grouped inelastic collisions, in [b]
  //!
  //! Grouping a collision takes its deflection away with its energy loss, and
  //! that deflection is not small: in carbon it is a quarter of what elastic
  //! scattering contributes, against a per cent or two in tungsten, where
  //! \f$Z^2\f$ puts nuclear elastic scattering far ahead.
  //!
  //! The angle comes from the recoil the collision leaves, and the model that
  //! decides how much recoil that is cuts between close and distant collisions
  //! at an oscillator energy belonging to the material rather than to the
  //! atom. So this is tabulated per material, which is why the oscillator
  //! energies and the density-effect correction are passed in.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] w_r Oscillator energy of each electroionization subshell in
  //!   this material, in [eV]
  //! \param[in] delta Density-effect correction on the electron energy grid
  //! \param[out] xs1 First transport cross section in [b], on that grid
  //! \param[out] s_screened Stopping power the density effect screens out of
  //!   the grouped channel, in [b eV] on that grid, to be subtracted from the
  //!   element's own restricted stopping power
  //! \param[out] w2_screened The same for the second moment, in [b eV^2]
  //! \param[out] s_total Collision stopping power the evaluated data delivers
  //!   after that screening, over the whole spectrum, in [b eV] on that grid
  void compute_inelastic_transport(int q_index, const vector<double>& w_r,
    const vector<double>& delta, tensor::Tensor<double>& xs1,
    tensor::Tensor<double>& s_screened, tensor::Tensor<double>& w2_screened,
    tensor::Tensor<double>& s_total) const;

  //! Electron energy grid this element's cross sections are tabulated on
  const tensor::Tensor<double>& electron_energy() const
  {
    return electron_energy_;
  }

  double ionization_hard_fraction(int q_index, int i_shell, double E) const;
  double bhabha_hard_fraction(int i_shell, double E) const;

  double excitation(double E) const;

  //! Electroionization: Moller scattering for an electron, Bhabha for a
  //! positron. Returns false when a positron's sampled transfer is rejected,
  //! which leaves the particle untouched -- see compute_moller_majorant().
  bool ionization(Particle& p, int i_shell, bool hard = false) const;

  //! Sample an energy transfer from the free binary cross section
  //!
  //! Moller's for an electron and Bhabha's for a positron, over a window the
  //! caller has already checked the hard channel was built from. The 1/W^2
  //! the shape is built around is inverted exactly and the rest is carried by
  //! rejection.
  //!
  //! \param[inout] p Projectile, whose energy sets the shape
  //! \param[in] W_lo Lower limit on the transfer in [eV]
  //! \param[in] W_hi Upper limit in [eV], below the projectile's energy
  //! \return Energy transfer in [eV], zero if the window is empty
  double sample_free_transfer(Particle& p, double W_lo, double W_hi) const;

  //! \param[in] hard Restrict the choice to the hard part of each subshell's
  //!   cross section, for a collision that ends a condensed-history step
  int sample_ionization_shell(Particle& p, bool hard = false) const;

  void bremsstrahlung(Particle& p, double k_min = 0.0) const;

  //! Bhabha scattering above the Moller kinematic limit
  //
  //! A Moller collision cannot transfer more than half of what is left after
  //! the binding energy is paid, so the evaluated knock-on spectra stop there
  //! and a positron's larger transfers are simply absent from them. This is
  //! that missing range, which lies far above every binding energy and is
  //! therefore free-electron territory.
  void bhabha(Particle& p, int i_shell, double w_min = 0.0) const;

  //! Sample the subshell in which such a collision occurs
  int sample_bhabha_shell(Particle& p, bool hard = false) const;

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
  //! in a distant collision the atom is excited as a whole, through a
  //! dipole-like interaction whose recoil is far smaller than the binary value
  //! and which PENELOPE splits into a longitudinal part, distributed as
  //! 1/(Q(Q+2mc^2)) up to the subshell's oscillator resonance, and a transverse
  //! part that carries no momentum at all.
  //
  //! Which of the two occurred is decided by how much of the evaluated cross
  //! section at this transfer the free binary collision can account for: that
  //! ratio is the probability the collision was close. \p density is the
  //! evaluated spectrum's density at W, without which the decision falls back
  //! to comparing W with the resonance energy.
  //!
  //! \param[out] declined Whether the density effect screened this collision
  //!   away. A distant transverse collision an isolated atom would have made
  //!   is suppressed in a medium, and what is suppressed does not happen: this
  //!   is how the Sternheimer correction reaches the stopping power rather
  //!   than merely the recoil.
  double sample_recoil(
    Particle& p, int i_shell, double W, double density, bool& declined) const;

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
  double sample_bremsstrahlung_energy(
    double E, uint64_t* seed, double k_min = 0.0) const;

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
  //! Transport moments <1-mu> and <(3/2)(1-mu^2)> of the elastic distribution,
  //! on the electron energy grid, indexed by projectile charge
  array<tensor::Tensor<double>, 2> elastic_mu1_;
  array<tensor::Tensor<double>, 2> elastic_mu2_;
  //! Soft/hard split of the elastic distribution at
  //! settings::deflection_cutoff, on the electron energy grid,
  //! indexed by projectile charge. The cutoff is held as the deflection 1-mu
  //! rather than as the cosine: at C1 = 0.001 and 100 MeV it is 1.1e-4 in
  //! tungsten, so four digits of the cosine carry no information, and both the
  //! interpolation between grid points and the deflection the sampler works in
  //! would inherit the loss.
  array<tensor::Tensor<double>, 2> elastic_dcut_;
  array<tensor::Tensor<double>, 2> elastic_p_hard_;
  array<tensor::Tensor<double>, 2> elastic_mu1_soft_;
  array<tensor::Tensor<double>, 2> elastic_mu2_soft_;
  //! Soft inelastic stopping power and straggling in [b eV] and [b eV^2] per
  //! atom, on the electron energy grid, indexed by projectile charge. Every
  //! channel a mixed step groups is summed into these, the bremsstrahlung one
  //! with the positron's radiative yield factor already applied.
  array<tensor::Tensor<double>, 2> inelastic_soft_s_;
  array<tensor::Tensor<double>, 2> inelastic_soft_w2_;
  //! Fraction of each inelastic channel that stays a discrete collision, on
  //! the electron energy grid and indexed by projectile charge, since the two
  //! projectiles have different cutoffs and so different thresholds. The
  //! electroionization one carries a subshell index as well. Bhabha scattering
  //! exists only for a positron, so it takes no charge index.
  array<tensor::Tensor<double>, 2> excitation_p_hard_;
  array<tensor::Tensor<double>, 2> ionization_p_hard_;
  //! The two halves of the hard electroionization channel, whose differential
  //! cross section is max(evaluated, free binary). The first is the evaluated
  //! spectrum's own tail above the soft cutoff (carrying the positron's
  //! majorant factor, since that half is sampled by reweighting rejection);
  //! the second is the amount by which the free cross section exceeds it,
  //! with the bound its rejection is drawn against. All in [b] on the electron
  //! energy grid. See compute_soft_inelastic().
  array<tensor::Tensor<double>, 2> ionization_hard_eval_;
  array<tensor::Tensor<double>, 2> ionization_deficit_xs_;
  array<tensor::Tensor<double>, 2> ionization_deficit_bound_;
  array<tensor::Tensor<double>, 2> ionization_deficit_xi_;
  //! Collision stopping power the hard inelastic channel carries, in [b eV] on
  //! the electron energy grid. An element quantity, so the material's pinning
  //! reads it rather than rebuilding it per material.
  array<tensor::Tensor<double>, 2> inelastic_hard_s_;
  //! Collision stopping power the evaluated data delivers unscreened and
  //! unrestricted, in [b eV]. The medium's screening is measured against it.
  array<tensor::Tensor<double>, 2> inelastic_unscreened_s_;
  array<tensor::Tensor<double>, 2> brems_p_hard_;
  tensor::Tensor<double> bhabha_p_hard_;
  //! The same two summed over subshells, in [b], which is all a cross section
  //! lookup wants. Summing them there instead meant walking every subshell and
  //! searching the energy grid once per subshell, on every lookup of every
  //! flight -- forty-four searches per lookup in tungsten, for a number that
  //! does not depend on the particle.
  array<tensor::Tensor<double>, 2> ionization_hard_xs_;
  tensor::Tensor<double> bhabha_hard_xs_;
  //! Hard cross section in [b], and an upper bound on it over the energies one
  //! step can cover. The flight to the next hard interaction is drawn from the
  //! bound, which does not change along the step, and the excess is taken back
  //! by declining that fraction of the interactions -- a delta interaction, in
  //! PENELOPE's terms. Sampling from the cross section at the energy the step
  //! began with would be drawing a flight from the wrong distribution, the
  //! projectile having slowed down in the meantime.
  array<tensor::Tensor<double>, 2> hard_total_;
  array<tensor::Tensor<double>, 2> hard_majorant_;
  //! Range the partial-wave data actually covers. Outside it the elastic cross
  //! sections are clamped to the endpoints, which is tolerable for the total --
  //! nearly flat at high energy -- but not for the first transport cross
  //! section, which is still falling as 1/E^2.
  double elastic_energy_min_ {0.0};
  double elastic_energy_max_ {INFTY};
  tensor::Tensor<double> electroionization_;
  //! Electroionization summed over subshells, and Bhabha likewise, on the
  //! electron energy grid. The transport wants only the total at an energy,
  //! and summing a slice for it on every cross section lookup meant walking
  //! every subshell and heap-allocating the slice's shape and strides to do
  //! it. Summed once here instead.
  tensor::Tensor<double> ionization_sum_;
  tensor::Tensor<double> bhabha_sum_;
  vector<unique_ptr<ElectroionizationSpectrum>> ionization_dist_;
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

  //! Sum the per-subshell inelastic spectra over subshells, once at load
  void compute_inelastic_sums();

  //! Tabulate the largest Bhabha-to-Moller ratio at each grid energy, which is
  //! the factor by which a positron's electroionization cross section is
  //! raised to make it a majorant of the true one
  void compute_moller_majorant();

  //! Tabulate the soft/hard split of the inelastic channels
  //!
  //! The thresholds are not free parameters: they follow from the transport
  //! cutoffs, since a collision whose every product would be killed on
  //! creation is one that nothing is lost by grouping. See
  //! soft_collision_cutoff() and soft_radiative_cutoff().
  void compute_soft_inelastic();

  //! Tabulate the collision stopping power the evaluated spectra deliver,
  //! unscreened and unrestricted. Built for every run, not only a grouped one.
  void compute_unscreened_stopping();

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

//! Integral of \f$W^{order}\f$ times the free Bhabha cross section over a
//! range of energy transfers, with the leading constant dropped
//!
//! Order 0 is the cross section a positron's transfers above the Moller limit
//! contribute, 1 the stopping power and 2 the straggling.
//!
//! \param[in] E Incident kinetic energy in [eV]
//! \param[in] W_lo, W_hi Range of energy transfer in [eV]
//! \param[in] order 0, 1 or 2
double bhabha_moment(double E, double W_lo, double W_hi, int order);

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

//==============================================================================
// Non-member functions
//==============================================================================

} // namespace openmc

#endif // OPENMC_PHOTON_H
