#include "openmc/photon.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/photon.h"
#include "openmc/physics.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"

#include "openmc/tensor.h"

#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <tuple> // for tie

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

//==============================================================================
// Electron interaction data, read into PhotonInteraction
//==============================================================================

void PhotonInteraction::read_electron_data(hid_t group)
{
  // name_, Z_ and index_ are already set from the photoatomic data; only the
  // electron grid and reactions are read here.
  read_dataset(group, "energy", electron_energy_);

  // Read elastic scattering
  hid_t rgroup = open_group(group, "elastic");
  read_dataset(rgroup, "xs", elastic_);
  // A partial-wave differential cross section covers the whole angular range,
  // so there is no large-angle/forward-peak split to reconcile. An evaluated
  // library does have one, and this build has none of the machinery that used
  // to bridge it, so refuse the data rather than transport it wrongly.
  if (object_exists(rgroup, "xs_total")) {
    tensor::Tensor<double> total;
    read_dataset(rgroup, "xs_total", total);
    for (int i = 0; i < total.size(); ++i) {
      if (total(i) > elastic_(i) * (1.0 + 1e-9)) {
        fatal_error(fmt::format(
          "Electron elastic data for {} splits the forward peak out of the "
          "angular distribution (xs_total exceeds xs). This build samples a "
          "partial-wave cross section over the whole angular range and cannot "
          "use such a library.",
          name_));
      }
    }
  }
  hid_t dist_group = open_group(rgroup, "distribution");
  // Interpolate between the tabulated distributions log-log in energy rather
  // than with the lin_lin rule the data carries. EEDL does specify INT=2 for
  // this TAB2, but that rule assumes the tables are close enough together for
  // a linear blend of them to mean something, and here they are not.
  //
  // Elastic angular data is tabulated on a sparse, geometric energy grid --
  // for aluminium there is no table between 256 keV and 10 MeV, an interval
  // across which 1-<mu> falls by a factor of 35. The default linear stochastic
  // interpolation picks the low-energy, wide-angle table 92% of the time right
  // across that gap, over-scattering by a factor of about 3.5. That leaves the
  // stopping power and CSDA range correct, so a range check passes, but stops
  // electrons penetrating and drives depth-deposition profiles far too shallow.
  elastic_angle_ = AngleDistribution {dist_group, Interpolation::log_log};
  close_group(dist_group);
  close_group(rgroup);

  // Read excitation
  rgroup = open_group(group, "excitation");
  read_dataset(rgroup, "xs", excitation_);
  hid_t dset = open_dataset(rgroup, "energy_loss");
  excitation_energy_loss_ = Tabulated1D {dset};
  close_dataset(dset);
  close_group(rgroup);

  // Read ionization
  rgroup = open_group(group, "ionization");
  read_dataset(rgroup, "xs", electroionization_);
  vector<std::string> designators;
  read_attribute(rgroup, "designators", designators);
  for (auto designator : designators) {
    hid_t shell_group = open_group(rgroup, designator.c_str());
    hid_t egroup = open_group(shell_group, "energy");
    // Knock-on spectra are anchored at the subshell binding energy; they must
    // not be remapped onto the interpolated endpoint range. See the note on
    // the ContinuousTabular constructor.
    ionization_dist_.push_back(make_unique<ContinuousTabular>(egroup, false));
    close_group(egroup);
    close_group(shell_group);
  }
  close_group(rgroup);

  // Map each electroionization subshell onto the corresponding entry of
  // shells_, which holds the binding energies and relaxation transitions.
  // Matching is by ENDF designator: the electroionization list (NXS(7)
  // subshells) and shells_ need not agree in length or order, and in
  // particular neither corresponds to the Compton Doppler broadening shell
  // list (NXS(5) shells).
  electron_shell_map_.resize(designators.size(), -1);
  for (int i = 0; i < designators.size(); ++i) {
    int endf_index = 0;
    int j = 1;
    for (const auto& subshell : SUBSHELLS) {
      if (designators[i] == subshell) {
        endf_index = j;
        break;
      }
      ++j;
    }

    for (int k = 0; k < shells_.size(); ++k) {
      if (shells_[k].index_subshell == endf_index) {
        electron_shell_map_[i] = k;
        break;
      }
    }

    if (electron_shell_map_[i] < 0) {
      fatal_error(fmt::format(
        "Electroionization subshell {} of element {} has no counterpart in the "
        "photoatomic data, so its binding energy and relaxation transitions "
        "are unavailable. The electron and photon libraries are inconsistent.",
        designators[i], name_));
    }
  }

  // Read bremsstrahlung
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", electron_bremsstrahlung_);
  dist_group = open_group(rgroup, "distribution");
  hid_t egroup = open_group(dist_group, "energy");
  bremsstrahlung_dist_ = make_unique<ContinuousTabular>(egroup);
  close_group(egroup);
  close_group(dist_group);
  close_group(rgroup);
}

void PhotonInteraction::calculate_electron_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = electron_energy_.size();
  double E = p.E();
  int i_grid;
  if (E <= electron_energy_[0]) {
    i_grid = 0;
  } else if (E > electron_energy_(n_grid - 1)) {
    i_grid = n_grid - 2;
  } else {
    // We use upper_bound_index here because sometimes photons are created with
    // energies that exactly match a grid point
    i_grid =
      upper_bound_index(electron_energy_.cbegin(), electron_energy_.cend(), E);
  }

  // check for case where two energy points are the same
  if (electron_energy_(i_grid) == electron_energy_(i_grid + 1))
    ++i_grid;

  // calculate interpolation factor
  double f = (E - electron_energy_(i_grid)) /
             (electron_energy_(i_grid + 1) - electron_energy_(i_grid));

  auto& xs {p.electron_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic elastic cross section
  xs.elastic = elastic_(i_grid) + f * (elastic_(i_grid + 1) - elastic_(i_grid));

  // Calculate microscopic excitation cross section
  xs.excitation =
    excitation_(i_grid) + f * (excitation_(i_grid + 1) - excitation_(i_grid));

  // Calculate microscopic ionization cross section
  const auto ion_i = electroionization_.slice(tensor::all, i_grid).sum();
  const auto ion_ip1 = electroionization_.slice(tensor::all, i_grid + 1).sum();
  xs.ionization = ion_i + f * (ion_ip1 - ion_i);

  // Calculate microscopic bremsstrahlung cross section
  xs.bremsstrahlung = electron_bremsstrahlung_(i_grid) +
                      f * (electron_bremsstrahlung_(i_grid + 1) -
                            electron_bremsstrahlung_(i_grid));

  // Calculate microscopic total cross section
  xs.total = xs.elastic + xs.excitation + xs.ionization + xs.bremsstrahlung;
  xs.last_E = p.E();
}

double PhotonInteraction::elastic_scatter(double E, uint64_t* seed) const
{
  return elastic_angle_.sample(E, seed);
}

double PhotonInteraction::excitation(double E) const
{
  return E - excitation_energy_loss_(E);
}

void PhotonInteraction::ionization(Particle& p, int i_shell) const
{
  double E_knock = ionization_dist_[i_shell]->sample(p.E(), p.current_seed());
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());
  // Binding energies live on shells_. Note this is NOT binding_energy_, which
  // belongs to the shorter Compton Doppler broadening shell list and would be
  // indexed out of bounds here.
  double e_b = shells_[electron_shell_map_[i_shell]].binding_energy;

  // The scattered primary must be left with positive energy. A sampled
  // knock-on energy that violates this would give a negative energy electron
  // and a negative argument in the scattering cosine below.
  if (E_knock + e_b >= p.E()) {
    p.write_restart();
    fatal_error(fmt::format(
      "Electroionization of {} shell {} at {} eV sampled a knock-on energy of "
      "{} eV which, with a binding energy of {} eV, exceeds the energy of the "
      "incident electron.",
      name_, i_shell, p.E(), E_knock, e_b));
  }

  // The knock-on is deflected according to the energy the primary actually
  // transferred, not the kinetic energy it is left with. The two differ by the
  // binding energy, which the atom absorbs: the momentum transfer that set the
  // recoil direction corresponds to E_knock + e_b, so using E_knock alone
  // ejects the electron too far sideways -- by 22% of the incident momentum
  // for a tantalum K shell at 100 keV. This is the PENELOPE convention, and it
  // makes the two polar angles the consistent free binary-collision pair for a
  // transfer of E_knock + e_b.
  double E_transfer = E_knock + e_b;
  double mu_knock = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                              (1.0 + 2.0 * MASS_ELECTRON_EV / E_transfer));
  Direction u_knock = rotate_angle(p.u(), mu_knock, &phi, p.current_seed());
  p.create_secondary(p.wgt(), u_knock, E_knock, ParticleType::electron());

  p.mu() = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                     (1.0 + 2.0 * MASS_ELECTRON_EV / (p.E() - E_knock - e_b)));
  phi += PI;
  p.u() = rotate_angle(p.u(), p.mu(), &phi, p.current_seed());
  p.E() = p.E() - E_knock - e_b;
}

int PhotonInteraction::sample_ionization_shell(Particle& p) const
{
  auto& xs {p.electron_xs(index_)};

  // Sample cumulative distribution function
  double cutoff = prn(p.current_seed()) * xs.ionization;
  int n_shell = electroionization_.shape(0);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;

  int i_shell;
  double prob = 0.0;
  for (i_shell = 0; i_shell < n_shell; ++i_shell) {
    double sigma = electroionization_(i_shell, i_grid) +
                   f * (electroionization_(i_shell, i_grid + 1) -
                         electroionization_(i_shell, i_grid));
    // Increment probability to compare to cutoff
    prob += sigma;
    if (prob > cutoff)
      return i_shell;
  }

  // If we made it here, no shell was sampled
  p.write_restart();
  fatal_error("Did not sample any electron shell during electro-ionization.");
}

namespace {

//! Polar angle of a bremsstrahlung photon, from Koch and Motz formula 2BS
//
//! EPICS carries no angular information for this channel -- the evaluation
//! states outright that the direction is left to the transport code -- so an
//! analytic form is the only option. This is the screened Schiff/Bethe-Heitler
//! shape, sampled by rejection in the reduced angle y = gamma*theta following
//! the formulation in EGSnrc, which uses it as its default for all energies.
//!
//! The emitting electron's direction is deliberately left alone. PENELOPE,
//! EGSnrc and MCNP all do the same, on the grounds that elastic scattering
//! governs the electron's deflection and the radiative recoil is negligible
//! beside it.
//!
//! \param Z Atomic number of the target
//! \param E Kinetic energy of the electron before emission [eV]
//! \param E_photon Energy of the emitted photon [eV]
//! \param seed Pseudorandom number seed pointer
//! \return Cosine of the angle between the photon and the electron
double bremsstrahlung_cos_theta(
  int Z, double E, double E_photon, uint64_t* seed)
{
  // (Zeff/111)^2 with Zeff^2 = Z(Z+1), the (Z+1) carrying the
  // electron-electron contribution. 1/111^2 = 8.116224e-5.
  double z_screen = 8.116224e-5 * std::cbrt(static_cast<double>(Z) * (Z + 1));
  double log_z_screen = -std::log(z_screen);

  double e_total = E + MASS_ELECTRON_EV;
  double e_final = e_total - E_photon;
  if (E_photon <= 0.0 || e_final <= MASS_ELECTRON_EV)
    return 1.0;

  double gamma = e_total / MASS_ELECTRON_EV;
  double beta = std::sqrt((gamma - 1.0) * (gamma + 1.0)) / gamma;
  double y2_max = 2.0 * beta * (1.0 + beta) * gamma * gamma;
  if (y2_max <= 0.0)
    return 1.0;
  double y2_max_inv = 1.0 / y2_max;
  double z2_max_sqrt = std::sqrt(y2_max + 1.0);

  double ratio = (e_final / MASS_ELECTRON_EV) / gamma;
  double arg1 = 1.0 + ratio * ratio;
  double arg2 = arg1 + 2.0 * ratio;

  // (2*E_final*gamma/k)^2, the argument of the screening logarithm
  double aux = 2.0 * e_final * gamma / E_photon;
  aux *= aux;
  double aux1 = aux * z_screen;
  double arg3 = (aux1 > 10.0) ? log_z_screen + (1.0 - aux1) / (aux1 * aux1)
                              : std::log(aux / (1.0 + aux1));

  double rej_max = arg1 * arg3 - arg2;
  if (!(rej_max > 0.0))
    return 1.0;

  // Rejection loop. The trial variable is drawn from the leading term and
  // rejected against the screened shape; it converges in a few iterations, but
  // cap it so that a pathological energy cannot hang a history.
  double y2 = 0.0;
  for (int iter = 0; iter < 1000; ++iter) {
    double xi = prn(seed);
    double test = prn(seed);
    double aux3 = z2_max_sqrt / (xi + (1.0 - xi) * z2_max_sqrt);
    test *= aux3 * rej_max;
    y2 = aux3 * aux3 - 1.0;
    double aux3_4 = aux3 * aux3 * aux3 * aux3;
    double y2s = ratio * y2 / aux3_4;
    double aux4 = 16.0 * y2s - arg2;
    double aux5 = arg1 - 4.0 * y2s;
    if (test < aux4 + aux5 * arg3)
      break;
    double aux2 = std::log(aux / (1.0 + aux1 / aux3_4));
    if (test < aux4 + aux5 * aux2)
      break;
  }

  return std::max(-1.0, std::min(1.0, 1.0 - 2.0 * y2 * y2_max_inv));
}

} // namespace

void PhotonInteraction::bremsstrahlung(Particle& p) const
{
  double E_photon = bremsstrahlung_dist_->sample(p.E(), p.current_seed());
  double mu = bremsstrahlung_cos_theta(Z_, p.E(), E_photon, p.current_seed());
  Direction u = rotate_angle(p.u(), mu, nullptr, p.current_seed());
  p.E() -= E_photon;
  p.create_secondary(p.wgt(), u, E_photon, ParticleType::photon());
}

} // namespace openmc
