#include "openmc/electron.h"

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

namespace data {

std::unordered_map<std::string, int> electron_map;
vector<unique_ptr<ElectronInteraction>> electroatomic;

} // namespace data

//==============================================================================
// ElectronInteraction implementation
//==============================================================================

ElectronInteraction::ElectronInteraction(hid_t group)
{
  // Set index of element in global vector
  index_ = data::electroatomic.size();

  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);
  data::electron_map[name_] = index_;

  // Resolve the index of this element in data::photoatomic, which is needed
  // for subshell binding energies and atomic relaxation. The photoatomic data
  // for an element is always loaded immediately before its electron data, so
  // the entry exists by now. Do not assume the two indices coincide.
  auto it = data::element_map.find(name_);
  if (it == data::element_map.end()) {
    fatal_error(fmt::format("Photoatomic data for element {} must be loaded "
                            "before its electron data.",
      name_));
  }
  i_photoatomic_ = it->second;

  // Get atomic number
  read_attribute(group, "Z", Z_);

  // Determine number of energies and read energy grid
  read_dataset(group, "energy", energy_);

  // Read elastic scattering
  hid_t rgroup = open_group(group, "elastic");
  read_dataset(rgroup, "xs", elastic_);
  read_dataset(rgroup, "xs_transport", elastic_transport_);
  read_dataset(rgroup, "xs_total", elastic_total_);
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
  elastic_angle_ =
    AngleDistribution {dist_group, Interpolation::log_log};
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
  read_dataset(rgroup, "xs", ionization_);
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

  // Map each electroionization subshell onto the corresponding subshell of the
  // photoatomic data, which holds the binding energies and relaxation
  // transitions. Matching is by ENDF designator: the electroionization list
  // (NXS(7) subshells) and the photoatomic list need not agree in length or
  // order, and in particular neither corresponds to the Compton Doppler
  // broadening shell list (NXS(5) shells).
  const auto& photoatomic {*data::photoatomic[i_photoatomic_]};
  shell_map_.resize(designators.size(), -1);
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

    for (int k = 0; k < photoatomic.shells_.size(); ++k) {
      if (photoatomic.shells_[k].index_subshell == endf_index) {
        shell_map_[i] = k;
        break;
      }
    }

    if (shell_map_[i] < 0) {
      fatal_error(fmt::format(
        "Electroionization subshell {} of element {} has no counterpart in the "
        "photoatomic data, so its binding energy and relaxation transitions "
        "are unavailable. The electron and photon libraries are inconsistent.",
        designators[i], name_));
    }
  }

  // Read bremsstrahlung
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", bremsstrahlung_);
  dist_group = open_group(rgroup, "distribution");
  hid_t egroup = open_group(dist_group, "energy");
  bremsstrahlung_dist_ = make_unique<ContinuousTabular>(egroup);
  close_group(egroup);
  close_group(dist_group);
  close_group(rgroup);
}

void ElectronInteraction::calculate_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = energy_.size();
  double E = p.E();
  int i_grid;
  if (E <= energy_[0]) {
    i_grid = 0;
  } else if (E > energy_(n_grid - 1)) {
    i_grid = n_grid - 2;
  } else {
    // We use upper_bound_index here because sometimes photons are created with
    // energies that exactly match a grid point
    i_grid = upper_bound_index(energy_.cbegin(), energy_.cend(), E);
  }

  // check for case where two energy points are the same
  if (energy_(i_grid) == energy_(i_grid + 1))
    ++i_grid;

  // calculate interpolation factor
  double f = (E - energy_(i_grid)) / (energy_(i_grid + 1) - energy_(i_grid));

  auto& xs {p.electron_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic elastic cross section
  xs.elastic = elastic_(i_grid) + f * (elastic_(i_grid + 1) - elastic_(i_grid));

  // Calculate microscopic excitation cross section
  xs.excitation =
    excitation_(i_grid) + f * (excitation_(i_grid + 1) - excitation_(i_grid));

  // Calculate microscopic ionization cross section
  const auto ion_i = ionization_.slice(tensor::all, i_grid).sum();
  const auto ion_ip1 = ionization_.slice(tensor::all, i_grid + 1).sum();
  xs.ionization = ion_i + f * (ion_ip1 - ion_i);

  // Calculate microscopic bremsstrahlung cross section
  xs.bremsstrahlung =
    bremsstrahlung_(i_grid) +
    f * (bremsstrahlung_(i_grid + 1) - bremsstrahlung_(i_grid));

  // Calculate microscopic total cross section
  xs.total = xs.elastic + xs.excitation + xs.ionization + xs.bremsstrahlung;
  xs.last_E = p.E();
}

double ElectronInteraction::elastic_scatter(double E, uint64_t* seed) const
{
  double mu = elastic_angle_.sample(E, seed);

  // The angular tables are spaced far too sparsely to interpolate between:
  // for aluminium there is none between 256 keV and 10 MeV, an interval across
  // which 1-<mu> falls by a factor of 35. Whatever is done with them, the mean
  // deflection in that gap is a guess.
  //
  // It does not have to be. EPRDATA14 tabulates the transport-corrected
  // elastic cross section on the same 373-point grid as the cross sections,
  // and it gives the first moment of the angular distribution directly. Use it
  // to set the mean deflection and let the tables supply only the shape, by
  // scaling the sampled deflection to the tabulated first moment.
  //
  // The first moment is what governs multiple scattering, so getting it right
  // matters far more than the detail of the shape between tables. Where a
  // table does exist the two agree to better than 2%, so this leaves the
  // sampling essentially untouched there and corrects it only in the gaps.
  double target = this->mean_deflection(E);
  if (target <= 0.0)
    return mu;
  double sampled = elastic_angle_.mean_deflection(E);
  if (sampled <= 0.0)
    return mu;

  double deflection = (1.0 - mu) * (target / sampled);
  // 1-mu cannot exceed 2; a rescaling that overshoots means exact backscatter
  return 1.0 - std::min(2.0, deflection);
}

namespace {

//! Mean deflection 1-<mu> of elastic scattering inside the forward peak
//
//! The evaluation tabulates the angular distribution only out to
//! mu = 1 - 1e-6, about 1.4 mrad from forward, and leaves the peak itself to
//! the screened-Rutherford form C/(2*eta + 1 - mu)^2 with Moliere's screening
//! angle eta. The first moment of that form over the peak depends only on eta
//! and the cutoff -- the normalization C cancels -- so it can be evaluated
//! without reference to the tables the peak is missing from.
double peak_mean_deflection(int Z, double E)
{
  // 1-mu at the upper end of the tabulated angular distributions. Verified
  // against every table of every element when the library is converted; see
  // openmc/data/electron.py.
  constexpr double X0 {1.0e-6};

  double e_total = E + MASS_ELECTRON_EV;
  double pc =
    std::sqrt(e_total * e_total - MASS_ELECTRON_EV * MASS_ELECTRON_EV);
  if (pc <= 0.0)
    return 0.5 * X0;
  double tau = E / MASS_ELECTRON_EV;
  double beta = pc / e_total;

  // Moliere's screening angle with the low-energy correction Seltzer
  // recommends, the same form ITS and MCNP use for this peak.
  double screen = FINE_STRUCTURE * MASS_ELECTRON_EV / (0.885 * pc);
  double coulomb = FINE_STRUCTURE * Z / beta;
  double eta = 0.25 * screen * screen * std::cbrt(static_cast<double>(Z) * Z) *
               (1.13 + 3.76 * coulomb * coulomb * std::sqrt(tau / (tau + 1.0)));

  // <1-mu> = a(1+u)g(u)/u, with a = 2*eta, u = X0/a and g = ln(1+u) - u/(1+u).
  //
  // g is log1pmx in disguise -- exactly -log1pmx(-w) for w = u/(1+u) -- and
  // shares its problem: log1p keeps the logarithm accurate, but g subtracts
  // two quantities that are both u to leading order and is itself only u^2/2,
  // so it sheds about u of its precision however the terms are computed.
  // Rewriting it does not help; log1p(u)/u, the substitution above and the
  // expm1 form all relocate the cancellation rather than remove it, and all
  // three degrade as ~2*eps/u. That is why no libm carries log1pmx and why
  // every implementation of it splits into a series near zero.
  //
  // The series below is carried far enough that the split stops being a tuned
  // parameter: at the handover its truncation error is 5e-13 and the direct
  // form's rounding error is 4.7e-13, so neither branch limits the other and
  // the worst error over the whole range, 4.7e-13, belongs to the direct form.
  // Both branches are used: u runs from 3.4e-10 (Am at 12 eV) to 1e9 (H at
  // 100 GeV).
  double a = 2.0 * eta;
  double u = X0 / a;
  double g = (u < 5.0e-4)
               ? 0.5 * u * u *
                   (1.0 + u * (-4.0 / 3.0 + u * (3.0 / 2.0 - 8.0 * u / 5.0)))
               : std::log1p(u) - u / (1.0 + u);
  return a * (1.0 + u) * g / u;
}

} // namespace

//! Mean deflection 1-<mu> of the tabulated large-angle distribution
//
//! The transport-corrected cross section is the first moment of the *total*
//! elastic cross section, peak included, while the angular tables describe
//! only the large-angle part. Take the peak's contribution back out before
//! forming the ratio, or the deflection is overstated wherever the peak
//! carries appreciable cross section: by 2% at 10 MeV in aluminium, rising to
//! 23% at 66 MeV. Below the energy at which the peak opens up the correction
//! is identically zero, the evaluation having sigma_total == sigma_elastic
//! there.
double ElectronInteraction::transport_ratio(int i) const
{
  if (elastic_(i) <= 0.0)
    return 0.0;
  double peak = elastic_total_(i) - elastic_(i);
  double moment = elastic_transport_(i);
  if (peak > 0.0)
    moment -= peak * peak_mean_deflection(Z_, energy_(i));
  return moment > 0.0 ? moment / elastic_(i) : 0.0;
}

double ElectronInteraction::mean_deflection(double E) const
{
  int n = energy_.size();
  if (E <= energy_[0])
    return this->transport_ratio(0);
  if (E >= energy_(n - 1))
    return this->transport_ratio(n - 1);

  int i = lower_bound_index(energy_.cbegin(), energy_.cend(), E);
  double e0 = energy_(i);
  double e1 = energy_(i + 1);
  if (e1 <= e0)
    return this->transport_ratio(i);

  double r0 = this->transport_ratio(i);
  double r1 = this->transport_ratio(i + 1);

  // 1-<mu> falls by orders of magnitude over this grid, so interpolate it
  // logarithmically; a linear interpolation would badly overshoot in between.
  if (r0 > 0.0 && r1 > 0.0) {
    double f = std::log(E / e0) / std::log(e1 / e0);
    return std::exp((1.0 - f) * std::log(r0) + f * std::log(r1));
  }
  double f = (E - e0) / (e1 - e0);
  return (1.0 - f) * r0 + f * r1;
}
double ElectronInteraction::excitation(double E) const
{
  return E - excitation_energy_loss_(E);
}

void ElectronInteraction::ionization(Particle& p, int i_shell) const
{
  double E_knock = ionization_dist_[i_shell]->sample(p.E(), p.current_seed());
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());
  // Binding energies live on the photoatomic subshell list. Note this is NOT
  // PhotonInteraction::binding_energy_, which belongs to the shorter Compton
  // Doppler broadening shell list and would be indexed out of bounds here.
  const auto& element {*data::photoatomic[i_photoatomic_]};
  double e_b = element.shells_[shell_map_[i_shell]].binding_energy;

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

  double mu_knock = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                              (1.0 + 2.0 * MASS_ELECTRON_EV / E_knock));
  Direction u_knock = rotate_angle(p.u(), mu_knock, &phi, p.current_seed());
  p.create_secondary(p.wgt(), u_knock, E_knock, ParticleType::electron());

  p.mu() = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                     (1.0 + 2.0 * MASS_ELECTRON_EV / (p.E() - E_knock - e_b)));
  phi += PI;
  p.u() = rotate_angle(p.u(), p.mu(), &phi, p.current_seed());
  p.E() = p.E() - E_knock - e_b;
}

int ElectronInteraction::sample_ionization_shell(Particle& p) const
{
  auto& xs {p.electron_xs(index_)};

  // Sample cumulative distribution function
  double cutoff = prn(p.current_seed()) * xs.ionization;
  int n_shell = ionization_.shape(0);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;

  int i_shell;
  double prob = 0.0;
  for (i_shell = 0; i_shell < n_shell; ++i_shell) {
    double sigma =
      ionization_(i_shell, i_grid) +
      f * (ionization_(i_shell, i_grid + 1) - ionization_(i_shell, i_grid));
    // Increment probability to compare to cutoff
    prob += sigma;
    if (prob > cutoff)
      return i_shell;
  }

  // If we made it here, no shell was sampled
  p.write_restart();
  fatal_error("Did not sample any electron shell during electro-ionization.");
}

void ElectronInteraction::bremsstrahlung(Particle& p) const
{
  double E_photon = bremsstrahlung_dist_->sample(p.E(), p.current_seed());
  p.E() -= E_photon;
  p.create_secondary(p.wgt(), p.u(), E_photon, ParticleType::photon());
}

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_electron()
{
  data::electroatomic.clear();
  data::electron_map.clear();
}

} // namespace openmc
