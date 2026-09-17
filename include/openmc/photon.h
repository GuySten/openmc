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

struct ElasticSplit {
  double mu_cut {1.0};   //!< cosine below which a deflection is hard
  double xs_hard {0.0};  //!< hard elastic cross section in [b]
  double xs1_soft {0.0}; //!< first transport cross section of the soft part
  double xs2_soft {0.0}; //!< second transport cross section of the soft part
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

  //! Transport cross sections of elastic scattering
  //!
  //! \f$\sigma_\ell = \sigma_{el} \langle 1 - P_\ell(\mu) \rangle\f$, the
  //! moments a condensed-history scheme groups soft collisions by. The first
  //! sets the transport mean free path \f$1/(n\sigma_1)\f$, over which the
  //! mean deflection relaxes by 1/e; the second enters the width of the
  //! grouped angular distribution.
  //!
  //! These are per atom and in the same units as the elastic cross section
  //! itself. They are tabulated at load time, so this is a grid lookup.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \param[in] order 1 or 2
  //! \return Transport cross section in [b]
  double elastic_transport_xs(int q_index, double E, int order) const;

  //! Soft/hard split of elastic scattering at one energy
  //!
  //! All three cross sections are per atom and in the same units as the
  //! elastic cross section itself. The split is tabulated at load time from
  //! settings::electron_c1, so this is a grid lookup.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \return The cutoff and the cross sections it implies
  ElasticSplit elastic_split(int q_index, double E) const;

  //! Soft inelastic energy loss, per atom and per unit path
  //!
  //! The stopping power restricted to collisions the mixed scheme groups, and
  //! the second moment of the same energy loss, which is the straggling that
  //! grouping would otherwise throw away.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \param[out] s Restricted stopping power in [b eV]
  //! \param[out] w2 Second moment of the restricted loss in [b eV^2]
  void inelastic_soft(int q_index, double E, double& s, double& w2) const;

  //! Fraction of a channel that stays a discrete collision
  //!
  //! Multiply the channel's cross section by this to get the rate of hard
  //! collisions. For electroionization the fraction is a quantile of the
  //! knock-on spectrum, which is what lets a positron's collisions keep using
  //! rejection: rejecting within the hard part leaves that part's quantile
  //! range alone. The two projectiles still differ, because their thresholds
  //! do. Bhabha scattering exists only for a positron and takes no charge.
  //!
  //! \param[in] q_index 0 for an electron, 1 for a positron
  //! \param[in] E Kinetic energy in [eV]
  //! \param[in] i_shell Index into the electroionization subshell list
  double excitation_hard_fraction(int q_index, double E) const;
  double ionization_hard_fraction(int q_index, int i_shell, double E) const;
  double bhabha_hard_fraction(int i_shell, double E) const;
  double bremsstrahlung_hard_fraction(int q_index, double E) const;

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
  double sample_recoil(
    Particle& p, int i_shell, double W, double density) const;

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
  //! Transport moments <1-mu> and <(3/2)(1-mu^2)> of the elastic distribution,
  //! on the electron energy grid, indexed by projectile charge
  array<tensor::Tensor<double>, 2> elastic_mu1_;
  array<tensor::Tensor<double>, 2> elastic_mu2_;
  //! Soft/hard split of the elastic distribution at settings::electron_c1, on
  //! the electron energy grid, indexed by projectile charge. The cutoff is
  //! held as the deflection 1-mu rather than as the cosine: at C1 = 0.001 and
  //! 100 MeV it is 1.1e-4 in tungsten, so four digits of the cosine carry no
  //! information, and both the interpolation between grid points and the
  //! deflection the sampler works in would inherit the loss.
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
  array<tensor::Tensor<double>, 2> brems_p_hard_;
  tensor::Tensor<double> bhabha_p_hard_;
  //! Range the partial-wave data actually covers. Outside it the elastic cross
  //! sections are clamped to the endpoints, which is tolerable for the total --
  //! nearly flat at high energy -- but not for the first transport cross
  //! section, which is still falling as 1/E^2.
  double elastic_energy_min_ {0.0};
  double elastic_energy_max_ {INFTY};
  tensor::Tensor<double> electroionization_;
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

//! Energy a grouped event may take from the projectile itself
//!
//! A soft collision must not be able to carry the projectile across the energy
//! at which it stops being transported. Below its own cutoff the projectile
//! would have been killed where it was -- and for a positron, killed means
//! annihilated at rest, which makes two 511 keV photons where an annihilation
//! in flight would have made one of up to \f$T + 1.5 m_e c^2\f$. A grouped
//! event that stepped over that energy would swap one outcome for the other,
//! so the transfer is bounded by how far the projectile is above it.
//!
//! \param[in] q_index 0 for an electron, 1 for a positron
//! \param[in] E Kinetic energy in [eV]
//! \return Headroom in [eV]
double soft_projectile_headroom(int q_index, double E);

//! Largest energy transfer a collision may make and still be grouped
//!
//! Two things have to hold. Nothing the collision produces may be lost: a
//! collision transferring \f$W\f$ puts on the stack a knock-on electron of
//! \f$W - B\f$ and, from the vacancy it leaves, fluorescence photons and Auger
//! electrons of at most \f$B\f$, every one of them below \f$W\f$ itself, so a
//! transfer under both the electron and the photon cutoff produces nothing
//! that would have been transported. And the projectile has to survive it,
//! which is soft_projectile_headroom().
//!
//! All three cutoffs therefore bear on the threshold: the photon and electron
//! ones through what the collision emits, the positron one through what the
//! projectile becomes.
//!
//! \param[in] q_index 0 for an electron, 1 for a positron
//! \param[in] E Kinetic energy in [eV]
//! \return Cutoff in [eV]; zero means nothing may be grouped
double soft_collision_cutoff(int q_index, double E);

//! Largest bremsstrahlung photon energy that may be grouped
//!
//! A bremsstrahlung collision produces one photon and leaves the projectile in
//! flight, so of what it emits only the photon cutoff bears on it. The
//! projectile bound matters more here than anywhere else: a single photon may
//! carry off nearly the whole kinetic energy, and without that bound a grouped
//! emission could take an electron from just above its cutoff to nearly at
//! rest and then smear the loss along the step.
//!
//! \param[in] q_index 0 for an electron, 1 for a positron
//! \param[in] E Kinetic energy in [eV]
//! \return Cutoff in [eV]
double soft_radiative_cutoff(int q_index, double E);

} // namespace openmc

#endif // OPENMC_PHOTON_H
