#include <cmath>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/charged_particle.h"
#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"

using namespace openmc;

// Relativistic two-body kinematics following ENDF-6 Appendix E
// All energies/masses in eV, using c=1 convention internally
struct TwoBodyResult {
  double T_ejectile_lab;  // Ejectile kinetic energy in lab [eV]
  double T_residual_lab;  // Residual kinetic energy in lab [eV]
  double mu_ejectile_lab; // Ejectile direction cosine in lab
  double mu_residual_lab; // Residual direction cosine in lab
};

TwoBodyResult relativistic_two_body(
  double T_i_lab, // Incident kinetic energy in lab [eV]
  double m_i,     // Incident particle rest mass [eV/c^2]
  double m_t,     // Target rest mass [eV/c^2]
  double m_e,     // Ejectile rest mass [eV/c^2]
  double Q,       // Q-value [eV]
  double mu_cm    // cos(theta) in CM frame
)
{
  // Eq. (E.21): Residual rest mass from mass-energy conservation
  double m_R = m_t + (m_i - m_e) - Q;

  // Eq. (E.6): Lorentz invariant S (square of total CM energy)
  double S = (m_i + m_t) * (m_i + m_t) + 2.0 * m_t * T_i_lab;

  // Eq. (E.5): Incident momentum squared in lab
  double p2_i_lab = 2.0 * m_i * T_i_lab + T_i_lab * T_i_lab;

  // Eq. (E.10): Incident momentum squared in CM
  double p2_i_cm = m_t * m_t * p2_i_lab / S;

  // Eq. (E.12): Boost parameter
  double sinh_chi = std::sqrt(p2_i_cm) / m_t;
  double cosh_chi = std::sqrt(1.0 + sinh_chi * sinh_chi);

  // Eq. (E.24): Sum of all masses
  double M = m_t + m_R + m_i + m_e;

  // Eq. (E.25): Ejectile momentum squared in CM
  double term1 = M * Q + 2.0 * m_t * T_i_lab;
  double term2 = term1 + 4.0 * m_R * m_e;
  double p2_e_cm = term1 * term2 / (4.0 * S);

  // Handle threshold (endothermic reactions)
  if (p2_e_cm < 0.0) {
    return {0.0, 0.0, 1.0, -1.0};
  }

  double p_e_cm = std::sqrt(p2_e_cm);

  // Eq. (E.27): Ejectile kinetic energy in CM
  double T_e_cm = p2_e_cm / (m_e + std::sqrt(m_e * m_e + p2_e_cm));
  double E_e_cm = m_e + T_e_cm; // Total energy in CM

  // Eq. (E.15): Transform ejectile to lab frame
  double sin_cm = std::sqrt(1.0 - mu_cm * mu_cm);
  double p_e1_cm = p_e_cm * mu_cm;  // parallel to incident direction
  double p_e2_cm = p_e_cm * sin_cm; // perpendicular

  double p_e1_lab = E_e_cm * sinh_chi + p_e1_cm * cosh_chi;
  double p_e2_lab = p_e2_cm;

  double p2_e_lab = p_e1_lab * p_e1_lab + p_e2_lab * p_e2_lab;

  // Eq. (E.4): Ejectile kinetic energy in lab
  double T_e_lab = p2_e_lab / (m_e + std::sqrt(m_e * m_e + p2_e_lab));

  // Eq. (E.16): Direction cosine in lab
  double mu_e_lab = (p2_e_lab > 0.0) ? p_e1_lab / std::sqrt(p2_e_lab) : 1.0;

  // Now do the same for the residual (opposite direction in CM)
  // Residual momentum in CM has same magnitude, opposite direction
  double p_R_cm = p_e_cm; // magnitude same by momentum conservation
  double T_R_cm = p2_e_cm / (m_R + std::sqrt(m_R * m_R + p2_e_cm));
  double E_R_cm = m_R + T_R_cm;

  // Residual goes opposite direction in CM
  double p_R1_cm = -p_e_cm * mu_cm;
  double p_R2_cm = -p_e_cm * sin_cm;

  double p_R1_lab = E_R_cm * sinh_chi + p_R1_cm * cosh_chi;
  double p_R2_lab = p_R2_cm;

  double p2_R_lab = p_R1_lab * p_R1_lab + p_R2_lab * p_R2_lab;
  double T_R_lab = p2_R_lab / (m_R + std::sqrt(m_R * m_R + p2_R_lab));
  double mu_R_lab = (p2_R_lab > 0.0) ? p_R1_lab / std::sqrt(p2_R_lab) : -1.0;

  return {T_e_lab, T_R_lab, mu_e_lab, mu_R_lab};
}

// Classical (Newtonian) two-body for comparison
TwoBodyResult classical_two_body(
  double T_i_lab, double m_i, double m_t, double m_e, double Q, double mu_cm)
{
  // Residual mass (classical: mass conserved, Q comes from binding)
  double m_R = m_t + m_i - m_e; // Wrong! Doesn't account for Q properly

  // CM energy (classical)
  double E_cm = T_i_lab * m_t / (m_i + m_t);
  double E_total_cm = E_cm + Q;

  if (E_total_cm <= 0.0) {
    return {0.0, 0.0, 1.0, -1.0};
  }

  // Ejectile energy in CM (classical partition)
  double T_e_cm = E_total_cm * m_R / (m_e + m_R);

  // Velocities (classical: v = sqrt(2T/m))
  double v_e_cm = std::sqrt(2.0 * T_e_cm / m_e);
  double v_cm_lab = std::sqrt(2.0 * T_i_lab / m_i) * m_i / (m_i + m_t);

  // Transform to lab
  double sin_cm = std::sqrt(1.0 - mu_cm * mu_cm);
  double v_para = v_e_cm * mu_cm + v_cm_lab;
  double v_perp = v_e_cm * sin_cm;
  double v_lab = std::sqrt(v_para * v_para + v_perp * v_perp);

  double T_e_lab = 0.5 * m_e * v_lab * v_lab;
  double mu_e_lab = (v_lab > 0.0) ? v_para / v_lab : 1.0;

  // Residual
  double T_R_cm = E_total_cm * m_e / (m_e + m_R);
  double v_R_cm = std::sqrt(2.0 * T_R_cm / m_R);

  double v_R_para = -v_R_cm * mu_cm + v_cm_lab;
  double v_R_perp = -v_R_cm * sin_cm;
  double v_R_lab = std::sqrt(v_R_para * v_R_para + v_R_perp * v_R_perp);

  double T_R_lab = 0.5 * m_R * v_R_lab * v_R_lab;
  double mu_R_lab = (v_R_lab > 0.0) ? v_R_para / v_R_lab : -1.0;

  return {T_e_lab, T_R_lab, mu_e_lab, mu_R_lab};
}

TEST_CASE("D-T Fusion Relativistic Kinematics")
{
  // D-T fusion: d + t -> n + alpha
  // Q = 17.59 MeV
  // Expected: neutron ~14.1 MeV, alpha ~3.5 MeV at low incident energy

  double m_d = MASS_DEUTERON_EV;
  double m_t = MASS_TRITON_EV;
  double m_n = MASS_NEUTRON_EV;
  double m_alpha = MASS_ALPHA_EV;
  double Q = 17.59e6; // eV

  std::cout << "\n=== D-T Fusion Relativistic Kinematics ===" << std::endl;
  std::cout << "Masses (MeV/c^2):" << std::endl;
  std::cout << "  Deuteron: " << m_d / 1e6 << std::endl;
  std::cout << "  Triton:   " << m_t / 1e6 << std::endl;
  std::cout << "  Neutron:  " << m_n / 1e6 << std::endl;
  std::cout << "  Alpha:    " << m_alpha / 1e6 << std::endl;
  std::cout << "  Q-value:  " << Q / 1e6 << " MeV" << std::endl;

  // Verify Eq. (E.21): m_R = m_t + (m_i - m_e) - Q should equal m_alpha
  double m_R_calc = m_t + (m_d - m_n) - Q;
  std::cout << "\nResidual mass check (Eq. E.21):" << std::endl;
  std::cout << "  Calculated m_R: " << m_R_calc / 1e6 << " MeV" << std::endl;
  std::cout << "  Actual m_alpha: " << m_alpha / 1e6 << " MeV" << std::endl;
  std::cout << "  Difference:     " << (m_R_calc - m_alpha) / 1e3 << " keV" << std::endl;

  SECTION("At 10 keV - comparison of relativistic vs classical")
  {
    double T_d = 10.0e3; // 10 keV

    auto rel = relativistic_two_body(T_d, m_d, m_t, m_n, Q, 1.0);
    auto cls = classical_two_body(T_d, m_d, m_t, m_n, Q, 1.0);

    std::cout << "\nAt T_d = 10 keV (forward direction, mu_cm = 1):" << std::endl;
    std::cout << "\n  RELATIVISTIC:" << std::endl;
    std::cout << "    Neutron: " << rel.T_ejectile_lab / 1e6 << " MeV, mu = "
              << rel.mu_ejectile_lab << std::endl;
    std::cout << "    Alpha:   " << rel.T_residual_lab / 1e6 << " MeV, mu = "
              << rel.mu_residual_lab << std::endl;
    std::cout << "    Total:   " << (rel.T_ejectile_lab + rel.T_residual_lab) / 1e6
              << " MeV" << std::endl;

    std::cout << "\n  CLASSICAL:" << std::endl;
    std::cout << "    Neutron: " << cls.T_ejectile_lab / 1e6 << " MeV, mu = "
              << cls.mu_ejectile_lab << std::endl;
    std::cout << "    Alpha:   " << cls.T_residual_lab / 1e6 << " MeV, mu = "
              << cls.mu_residual_lab << std::endl;
    std::cout << "    Total:   " << (cls.T_ejectile_lab + cls.T_residual_lab) / 1e6
              << " MeV" << std::endl;

    std::cout << "\n  Expected: T_d + Q = " << (T_d + Q) / 1e6 << " MeV" << std::endl;

    // The famous 14.1 MeV neutron
    CHECK_THAT(rel.T_ejectile_lab / 1e6, Catch::Matchers::WithinRel(14.1, 0.02));
  }

  SECTION("At 100 keV")
  {
    double T_d = 100.0e3;

    auto rel = relativistic_two_body(T_d, m_d, m_t, m_n, Q, 1.0);
    auto cls = classical_two_body(T_d, m_d, m_t, m_n, Q, 1.0);

    std::cout << "\nAt T_d = 100 keV (forward):" << std::endl;
    std::cout << "  Relativistic neutron: " << rel.T_ejectile_lab / 1e6 << " MeV" << std::endl;
    std::cout << "  Classical neutron:    " << cls.T_ejectile_lab / 1e6 << " MeV" << std::endl;
    std::cout << "  Difference:           "
              << (rel.T_ejectile_lab - cls.T_ejectile_lab) / 1e3 << " keV" << std::endl;
  }

  SECTION("Angular distribution - mu_cm = 0 (perpendicular)")
  {
    double T_d = 10.0e3;

    auto rel = relativistic_two_body(T_d, m_d, m_t, m_n, Q, 0.0);

    std::cout << "\nAt T_d = 10 keV, mu_cm = 0 (perpendicular in CM):" << std::endl;
    std::cout << "  Neutron: " << rel.T_ejectile_lab / 1e6 << " MeV, mu_lab = "
              << rel.mu_ejectile_lab << std::endl;
    std::cout << "  Alpha:   " << rel.T_residual_lab / 1e6 << " MeV, mu_lab = "
              << rel.mu_residual_lab << std::endl;
  }

  SECTION("Angular distribution - mu_cm = -1 (backward)")
  {
    double T_d = 10.0e3;

    auto rel = relativistic_two_body(T_d, m_d, m_t, m_n, Q, -1.0);

    std::cout << "\nAt T_d = 10 keV, mu_cm = -1 (backward in CM):" << std::endl;
    std::cout << "  Neutron: " << rel.T_ejectile_lab / 1e6 << " MeV, mu_lab = "
              << rel.mu_ejectile_lab << std::endl;
    std::cout << "  Alpha:   " << rel.T_residual_lab / 1e6 << " MeV, mu_lab = "
              << rel.mu_residual_lab << std::endl;
  }

  SECTION("Energy conservation check")
  {
    double T_d = 100.0e3;

    auto rel = relativistic_two_body(T_d, m_d, m_t, m_n, Q, 1.0);

    double T_total = rel.T_ejectile_lab + rel.T_residual_lab;
    double T_expected = T_d + Q;

    std::cout << "\nEnergy conservation (relativistic):" << std::endl;
    std::cout << "  T_d + Q     = " << T_expected / 1e6 << " MeV" << std::endl;
    std::cout << "  T_n + T_a   = " << T_total / 1e6 << " MeV" << std::endl;
    std::cout << "  Difference: " << (T_total - T_expected) / 1e3 << " keV" << std::endl;

    // Should conserve energy to high precision
    CHECK_THAT(T_total, Catch::Matchers::WithinRel(T_expected, 1e-6));
  }

  SECTION("Load and verify D-T data from HDF5")
  {
    std::string filename = "data/deuteron_H3.h5";
    read_charged_particle_data(filename);

    REQUIRE(data::charged_particles.size() >= 1);

    // Find deuteron_H3
    int idx = -1;
    for (size_t i = 0; i < data::charged_particles.size(); ++i) {
      if (data::charged_particles[i]->name_ == "deuteron_H3") {
        idx = i;
        break;
      }
    }
    REQUIRE(idx >= 0);

    auto& cp = *data::charged_particles[idx];

    std::cout << "\nLoaded D-T data:" << std::endl;
    for (int i = 0; i < cp.n_reactions_; ++i) {
      std::cout << "  MT=" << cp.MT_[i]
                << ", Q=" << cp.Q_value_[i] / 1e6 << " MeV"
                << ", ejectile=" << particle_type_to_string(cp.ejectile_[i])
                << std::endl;
    }

    // Find n+alpha reaction (MT=50)
    int i_rx = -1;
    for (int i = 0; i < cp.n_reactions_; ++i) {
      if (cp.MT_[i] == 50) {
        i_rx = i;
        break;
      }
    }
    REQUIRE(i_rx >= 0);

    CHECK(cp.ejectile_[i_rx] == ParticleType::neutron);
    CHECK_THAT(cp.Q_value_[i_rx] / 1e6, Catch::Matchers::WithinRel(17.59, 0.01));

    charged_particles_clear();
  }
}
