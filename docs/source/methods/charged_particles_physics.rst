.. _methods_charged_particle_physics:

========================
Charged Particle Physics
========================

Charged particles (electrons and positrons) can be treated in two ways. By
default OpenMC neglects their spatial transport, assuming they deposit all
their energy locally and produce bremsstrahlung photons at their birth
location. This approximation, called the thick-target bremsstrahlung (TTB)
approximation, is justified by the fact that charged particles have much
shorter stopping ranges compared to neutrons and photons, especially in
high-density materials.

When electron transport is enabled, electrons and positrons are instead
transported explicitly by an analog single-event scheme, described in
:ref:`electron_transport` below. Every elastic, excitation, electroionization
and bremsstrahlung interaction is sampled individually; there is no
condensed-history step and no multiple-scattering theory. This is far more
expensive -- a 1 MeV electron undergoes on the order of :math:`10^5`
interactions before it ranges out -- but it makes no assumption about the step
length and remains valid where condensed-history schemes break down, which is
generally below about 1 keV.

-----------------------------
Charged Particle Interactions
-----------------------------

Bremsstrahlung
--------------

When a charged particle is decelerated in the field of an atom, some of its
kinetic energy is converted into electromagnetic radiation known as
bremsstrahlung, or 'braking radiation'. In each event, an electron or positron
with kinetic energy :math:`T` generates a photon with an energy :math:`E`
between :math:`0` and :math:`T`. Bremsstrahlung is described by a cross section
that is differential in photon energy, in the direction of the emitted photon,
and in the final direction of the charged particle. However, in Monte Carlo
simulations it is typical to integrate over the angular variables to obtain a
single differential cross section with respect to photon energy, which is often
expressed in the form

.. math::
    :label: bremsstrahlung-dcs

    \frac{d\sigma_{\text{br}}}{dE} = \frac{Z^2}{\beta^2} \frac{1}{E}
    \chi(Z, T, \kappa),

where :math:`\kappa = E/T` is the reduced photon energy and :math:`\chi(Z, T,
\kappa)` is the scaled bremsstrahlung cross section, which is experimentally
measured.

Because electrons are attracted to atomic nuclei whereas positrons are
repulsed, the cross section for positrons is smaller, though it approaches that
of electrons in the high energy limit. To obtain the positron cross section, we
multiply :eq:`bremsstrahlung-dcs` by the :math:`\kappa`-independent factor used
in Salvat_,

.. math::
    :label: positron-factor

    \begin{aligned}
    F_{\text{p}}(Z,T) =
    & 1 - \text{exp}(-1.2359\times 10^{-1}t + 6.1274\times 10^{-2}t^2 - 3.1516\times 10^{-2}t^3 \\
    & + 7.7446\times 10^{-3}t^4 - 1.0595\times 10^{-3}t^5 + 7.0568\times 10^{-5}t^6 \\
    & - 1.8080\times 10^{-6}t^7),
    \end{aligned}

where

.. math::
    :label: positron-factor-t

    t = \ln\left(1 + \frac{10^6}{Z^2}\frac{T}{\text{m}_\text{e}c^2} \right).

:math:`F_{\text{p}}(Z,T)` is the ratio of the radiative stopping powers for
positrons and electrons. Stopping power describes the average energy loss per
unit path length of a charged particle as it passes through matter:

.. math::
    :label: stopping-power

    -\frac{dT}{ds} = n \int E \frac{d\sigma}{dE} dE \equiv S(T),

where :math:`n` is the number density of the material and :math:`d\sigma/dE` is
the cross section differential in energy loss. The total stopping power
:math:`S(T)` can be separated into two components: the radiative stopping
power :math:`S_{\text{rad}}(T)`, which refers to energy loss due to
bremsstrahlung, and the collision stopping power :math:`S_{\text{col}}(T)`,
which refers to the energy loss due to inelastic collisions with bound
electrons in the material that result in ionization and excitation. The
radiative stopping power for electrons is given by

.. math::
    :label: radiative-stopping-power

    S_{\text{rad}}(T) = n \frac{Z^2}{\beta^2} T \int_0^1 \chi(Z,T,\kappa)
    d\kappa.


To obtain the radiative stopping power for positrons,
:eq:`radiative-stopping-power`  is multiplied by :eq:`positron-factor`.

While the models for photon interactions with matter described above can safely
assume interactions occur with free atoms, sampling the target atom based on
the macroscopic cross sections, molecular effects cannot necessarily be
disregarded for charged particle treatment. For compounds and mixtures, the
bremsstrahlung cross section is calculated using Bragg's additivity rule as

.. math::
    :label: material-bremsstrahlung-dcs

    \frac{d\sigma_{\text{br}}}{dE} = \frac{1}{\beta^2 E} \sum_i \gamma_i Z^2_i
    \chi(Z_i, T, \kappa),

where the sum is over the constituent elements and :math:`\gamma_i` is the
atomic fraction of the :math:`i`-th element. Similarly, the radiative stopping
power is calculated using Bragg's additivity rule as

.. math::
    :label: material-radiative-stopping-power

    S_{\text{rad}}(T) = \sum_i w_i S_{\text{rad},i}(T),

where :math:`w_i` is the mass fraction of the :math:`i`-th element and
:math:`S_{\text{rad},i}(T)` is found for element :math:`i` using
:eq:`radiative-stopping-power`. The collision stopping power, however, is a
function of certain quantities such as the mean excitation energy :math:`I` and
the density effect correction :math:`\delta_F` that depend on molecular
properties. These quantities cannot simply be summed over constituent elements
in a compound, but should instead be calculated for the material. The Bethe
formula can be used to find the collision stopping power of the material:

.. math::
    :label: material-collision-stopping-power

    S_{\text{col}}(T) = \frac{2 \pi r_e^2 m_e c^2}{\beta^2} N_A \frac{Z}{A_M}
    [\ln(T^2/I^2) + \ln(1 + \tau/2) + F(\tau) - \delta_F(T)],

where :math:`N_A` is Avogadro's number, :math:`A_M` is the molar mass,
:math:`\tau = T/m_e`, and :math:`F(\tau)` depends on the particle type. For
electrons,

.. math::
    :label: F-electron

    F_{-}(\tau) = (1 - \beta^2)[1 + \tau^2/8 - (2\tau + 1) \ln2],

while for positrons

.. math::
    :label: F-positron

    F_{+}(\tau) = 2\ln2 - (\beta^2/12)[23 + 14/(\tau + 2) + 10/(\tau + 2)^2 +
    4/(\tau + 2)^3].

The density effect correction :math:`\delta_F` takes into account the reduction
of the collision stopping power due to the polarization of the material the
charged particle is passing through by the electric field of the particle.
It can be evaluated using the method described by Sternheimer_, where the
equation for :math:`\delta_F` is

.. math::
    :label: density-effect-correction

    \delta_F(\beta) = \sum_{i=1}^n f_i \ln[(l_i^2 + l^2)/l_i^2] -
    l^2(1-\beta^2).

Here, :math:`f_i` is the oscillator strength of the :math:`i`-th transition,
given by :math:`f_i = n_i/Z`, where :math:`n_i` is the number of electrons in
the :math:`i`-th subshell. The frequency :math:`l` is the solution of the
equation

.. math::
    :label: density-effect-l

    \frac{1}{\beta^2} - 1 = \sum_{i=1}^{n} \frac{f_i}{\bar{\nu}_i^2 + l^2},

where :math:`\bar{v}_i` is defined as

.. math::
    :label: density-effect-nubar

    \bar{\nu}_i = h\nu_i \rho / h\nu_p.

The plasma energy :math:`h\nu_p` of the medium is given by

.. math::
    :label: plasma-frequency

    h\nu_p = \sqrt{\frac{(hc)^2 r_e \rho_m N_A Z}{\pi A}},

where :math:`A` is the atomic weight and :math:`\rho_m` is the density of the
material. In :eq:`density-effect-nubar`, :math:`h\nu_i` is the oscillator
energy, and :math:`\rho` is an adjustment factor introduced to give agreement
between the experimental values of the oscillator energies and the mean
excitation energy. The :math:`l_i` in :eq:`density-effect-correction` are
defined as

.. math::
    :label: density-effect-li

    \begin{aligned}
    l_i &= (\bar{\nu}_i^2 + 2/3f_i)^{1/2} ~~~~&\text{for}~~ \bar{\nu}_i > 0 \\
    l_n &= f_n^{1/2} ~~~~&\text{for}~~ \bar{\nu}_n = 0,
    \end{aligned}

where the second case applies to conduction electrons. For a conductor,
:math:`f_n` is given by :math:`n_c/Z`, where :math:`n_c` is the effective
number of conduction electrons, and :math:`v_n = 0`. The adjustment factor
:math:`\rho` is determined using the equation for the mean excitation energy:

.. math::
    :label: mean-excitation-energy

    \ln I = \sum_{i=1}^{n-1} f_i \ln[(h\nu_i\rho)^2 + 2/3f_i(h\nu_p)^2]^{1/2} +
    f_n \ln (h\nu_pf_n^{1/2}).

.. _ttb:


Thick-Target Bremsstrahlung Approximation
+++++++++++++++++++++++++++++++++++++++++

Since charged particles lose their energy on a much shorter distance scale than
neutral particles, not much error should be introduced by neglecting to
transport electrons. However, the bremsstrahlung emitted from high energy
electrons and positrons can travel far from the interaction site. Thus, even
without a full electron transport mode it is necessary to model bremsstrahlung.
We use a thick-target bremsstrahlung (TTB) approximation based on the models in
Salvat_ and Kaltiaisenaho_ for generating bremsstrahlung photons, which assumes
the charged particle loses all its energy in a single homogeneous material
region.

To model bremsstrahlung using the TTB approximation, we need to know the number
of photons emitted by the charged particle and the energy distribution of the
photons. These quantities can be calculated using the continuous slowing down
approximation (CSDA). The CSDA assumes charged particles lose energy
continuously along their trajectory with a rate of energy loss equal to the
total stopping power, ignoring fluctuations in the energy loss. The
approximation is useful for expressing average quantities that describe how
charged particles slow down in matter. For example, the CSDA range approximates
the average path length a charged particle travels as it slows to rest:

.. math::
    :label: csda-range

    R(T) = \int^T_0 \frac{dT'}{S(T')}.

Actual path lengths will fluctuate around :math:`R(T)`. The average number of
photons emitted per unit path length is given by the inverse bremsstrahlung
mean free path:

.. math::
    :label: inverse-bremsstrahlung-mfp

    \lambda_{\text{br}}^{-1}(T,E_{\text{cut}})
    = n\int_{E_{\text{cut}}}^T\frac{d\sigma_{\text{br}}}{dE}dE
    = n\frac{Z^2}{\beta^2}\int_{\kappa_{\text{cut}}}^1\frac{1}{\kappa}
    \chi(Z,T,\kappa)d\kappa.

The lower limit of the integral in :eq:`inverse-bremsstrahlung-mfp` is non-zero
because the bremsstrahlung differential cross section diverges for small photon
energies but is finite for photon energies above some cutoff energy
:math:`E_{\text{cut}}`. The mean free path
:math:`\lambda_{\text{br}}^{-1}(T,E_{\text{cut}})` is used to calculate the
photon number yield, defined as the average number of photons emitted with
energy greater than :math:`E_{\text{cut}}` as the charged particle slows down
from energy :math:`T` to :math:`E_{\text{cut}}`. The photon number yield is
given by

.. math::
    :label: photon-number-yield

    Y(T,E_{\text{cut}}) = \int^{R(T)}_{R(E_{\text{cut}})}
    \lambda_{\text{br}}^{-1}(T',E_{\text{cut}})ds = \int_{E_{\text{cut}}}^T
    \frac{\lambda_{\text{br}}^{-1}(T',E_{\text{cut}})}{S(T')}dT'.

:math:`Y(T,E_{\text{cut}})` can be used to construct the energy spectrum of
bremsstrahlung photons: the number of photons created with energy between
:math:`E_1` and :math:`E_2` by a charged particle with initial kinetic energy
:math:`T` as it comes to rest is given by :math:`Y(T,E_1) - Y(T,E_2)`.

To simulate the emission of bremsstrahlung photons, the total stopping power
and bremsstrahlung differential cross section for positrons and electrons must
be calculated for a given material using :eq:`material-bremsstrahlung-dcs` and
:eq:`material-radiative-stopping-power`. These quantities are used to build the
tabulated bremsstrahlung energy PDF and CDF for that material for each incident
energy :math:`T_k` on the energy grid. The following algorithm is then applied
to sample the photon energies:

1. For an incident charged particle with energy :math:`T`, sample the number of
   emitted photons as

   .. math::

       N = \lfloor Y(T,E_{\text{cut}}) + \xi_1 \rfloor.

2. Rather than interpolate the PDF between indices :math:`k` and :math:`k+1`
   for which :math:`T_k < T < T_{k+1}`, which is computationally expensive, use
   the composition method and sample from the PDF at either :math:`k` or
   :math:`k+1`. Using linear interpolation on a logarithmic scale, the PDF can
   be expressed as

   .. math::

       p_{\text{br}}(T,E) = \pi_k p_{\text{br}}(T_k,E) + \pi_{k+1}
       p_{\text{br}}(T_{k+1},E),

   where the interpolation weights are

   .. math::

       \pi_k = \frac{\ln T_{k+1} - \ln T}{\ln T_{k+1} - \ln T_k},~~~
       \pi_{k+1} = \frac{\ln T - \ln T_k}{\ln T_{k+1} - \ln T_k}.

   Sample either the index :math:`i = k` or :math:`i = k+1` according to the
   point probabilities :math:`\pi_{k}` and :math:`\pi_{k+1}`.

3. Determine the maximum value of the CDF :math:`P_{\text{br,max}}`.

3. Sample the photon energies using the inverse transform method with the
   tabulated CDF :math:`P_{\text{br}}(T_i, E)` i.e.,

   .. math::

       E = E_j \left[ (1 + a_j) \frac{\xi_2 P_{\text{br,max}} -
       P_{\text{br}}(T_i, E_j)} {E_j p_{\text{br}}(T_i, E_j)} + 1
       \right]^{\frac{1}{1 + a_j}}

   where the interpolation factor :math:`a_j` is given by

   .. math::

       a_j = \frac{\ln p_{\text{br}}(T_i,E_{j+1}) - \ln p_{\text{br}}(T_i,E_j)}
       {\ln E_{j+1} - \ln E_j}

   and :math:`P_{\text{br}}(T_i, E_j) \le \xi_2 P_{\text{br,max}} \le
   P_{\text{br}}(T_i, E_{j+1})`.

We ignore the range of the electron or positron, i.e., the bremsstrahlung
photons are produced in the same location that the charged particle was
created. The direction of the photons is assumed to be the same as the
direction of the incident charged particle, which is a reasonable approximation
at higher energies when the bremsstrahlung radiation is emitted at small
angles. This is an approximation of the TTB model specifically; when electrons
are transported, the emission angle is sampled from the distribution given in
:ref:`bremsstrahlung_angle`.

Electron-Positron Annihilation
------------------------------

When a positron collides with an electron, both particles are annihilated and
generally two photons with equal energy are created. If the kinetic energy of
the positron is high enough, the two photons can have different energies, and
the higher-energy photon is emitted preferentially in the direction of flight
of the positron. It is also possible to produce a single photon if the
interaction occurs with a bound electron, and in some cases three (or, rarely,
even more) photons can be emitted. However, the annihilation cross section is
largest for low-energy positrons, and as the positron energy decreases, the
angular distribution of the emitted photons becomes isotropic.

In OpenMC, we assume the most likely case in which a low-energy positron (which
has already lost most of its energy to bremsstrahlung radiation) interacts with
an electron which is free and at rest. Two photons with energy equal to the
electron rest mass energy :math:`m_e c^2 = 0.511` MeV are emitted isotropically
in opposite directions.



.. _electron_transport:

------------------
Electron Transport
------------------

When electron transport is enabled, electrons and positrons are transported by
an analog single-event scheme. The distance to the next interaction is sampled
from the total electroatomic cross section, one of the four channels below is
selected in proportion to its cross section, and that interaction is sampled in
full. Nothing is condensed into a step: there is no multiple-scattering
distribution, no substep energy loss, and no path-length correction.

The interaction data are those of the EPICS evaluated libraries, in particular
the Evaluated Electron Data Library (EEDL), read from the eprdata ACE format.
For each element the library supplies the four cross sections on a common dense
energy grid, the elastic angular distributions, the average energy loss to
excitation, the electroionization spectra for each subshell, and the
bremsstrahlung photon spectra.

Elastic Scattering
------------------

Elastic scattering changes the direction of the electron without transferring
energy to the atom. It is by a wide margin the most frequent interaction, and
the accumulation of many small deflections is what limits how deeply electrons
penetrate.

The evaluation splits the angular distribution in two. A tabulated distribution
covers scattering away from the forward direction, out to a cutoff at
:math:`\mu_{\text{max}} = 1 - 10^{-6}`, about 1.4 mrad from forward; the
narrow peak beyond that cutoff is left to an analytic screened-Rutherford form

.. math::
    :label: elastic-peak

    f(\mu) \propto \frac{1}{(2\eta + 1 - \mu)^2},

with Molière's screening angle carrying the low-energy correction
recommended by Seltzer,

.. math::
    :label: elastic-screening

    \eta = \frac{1}{4}\left(\frac{\alpha m_e c}{0.885\,p}\right)^2
    Z^{2/3} \left[1.13 + 3.76 \left(\frac{\alpha Z}{\beta}\right)^2
    \sqrt{\frac{\tau}{\tau + 1}}\right],

where :math:`\tau = T/m_ec^2`. The cross section that accompanies the
tabulated distribution is the large-angle cross section
:math:`\sigma_{\text{el}}`, not the total; pairing the total with these tables
would count the peak twice.

The angular tables are given at only 14 to 16 energies per element, spanning
ten decades. For aluminium there is no table between 256 keV and 10 MeV, an
interval across which the mean deflection falls by a factor of 35, so the
tables cannot simply be interpolated. Two things follow. First, the tables are
interpolated logarithmically in energy, with both bracketing distributions
sampled at the same cumulative probability and the resulting deflections
combined geometrically; a linear rule places essentially all of the weight on
the lower table across a gap of that size. Second, the mean deflection is not
taken from the tables at all. The library tabulates a transport-corrected
elastic cross section :math:`\sigma_{\text{tr}}` on the same dense grid as the
cross sections, and its first moment gives the mean deflection directly:

.. math::
    :label: mean-deflection

    \langle 1 - \mu \rangle = \frac{\sigma_{\text{tr}} -
    \sigma_{\text{peak}} \langle 1 - \mu \rangle_{\text{peak}}}
    {\sigma_{\text{el}}},

where :math:`\sigma_{\text{peak}}` is the difference between the total and
large-angle cross sections. The subtraction is needed because
:math:`\sigma_{\text{tr}}` is the first moment of the *total* distribution,
peak included, while the tables describe only the large-angle part. Integrating
:eq:`elastic-peak` over the peak gives its contribution in closed form,

.. math::
    :label: peak-moment

    \langle 1 - \mu \rangle_{\text{peak}} =
    -\frac{a(a + x_0)}{x_0}\left[\ln(1 - w) + w\right], \quad
    a = 2\eta,\quad x_0 = 1 - \mu_{\text{max}},\quad
    w = \frac{x_0}{a + x_0}.

The bracketed term is :math:`\ln(1+x) - x` at :math:`x = -w`. Both of its
terms approach :math:`-w` while their difference is only :math:`w^2/2`, so it
is evaluated by a dedicated routine rather than by subtracting them directly.

Below the energy at which the peak opens up -- 1.75 MeV in aluminium, 3 MeV in
iron, 8 MeV in uranium -- the evaluation gives
:math:`\sigma_{\text{tot}} = \sigma_{\text{el}}` and the correction vanishes
identically.

Each sampled deflection is then scaled so that its mean matches
:eq:`mean-deflection`, leaving the tables to supply the shape and the dense
grid to supply the first moment.

Atomic Excitation
-----------------

An excitation event raises a bound electron to a higher state without ionizing
the atom. The evaluation tabulates only the average energy loss
:math:`\Delta(T)`, so the incident electron loses that average,

.. math::
    :label: excitation-loss

    T' = T - \Delta(T),

and continues undeflected with no secondary particle produced. There is no
straggling within an event, because the evaluation provides no distribution to
sample; fluctuation in the total excitation loss arises only from the number of
events. This matches the treatment in MCNP's single-event mode.

Electroionization
-----------------

Electroionization ejects a bound electron. The subshell is sampled in
proportion to the subshell ionization cross sections, and the kinetic energy
:math:`T_{\text{k}}` of the ejected knock-on electron is sampled from the
spectrum tabulated for that subshell. The incident electron loses the knock-on
energy together with the binding energy :math:`B` of the vacated subshell,

.. math::
    :label: ionization-loss

    T' = T - T_{\text{k}} - B.

Each spectrum ends exactly at the kinematic limit
:math:`(T - B)/2`, which is affine in :math:`T`, and the spectra are anchored
at the low end near the binding energy rather than scaling with :math:`T`.
They are therefore not self-similar and are sampled without unit-base scaling,
with the two bracketing tables combined geometrically at matched cumulative
probability. As with the angular tables, the incident-energy grids are sparse
-- aluminium's K shell jumps from 15.8 keV to 501 keV -- and are interpolated
logarithmically.

The polar deflections of both electrons follow from conservation of momentum
and are not sampled independently:

.. math::
    :label: ionization-angles

    \mu = \left[\frac{T'(T + 2m_ec^2)}{T(T' + 2m_ec^2)}\right]^{1/2},
    \quad
    \mu_{\text{k}} = \left[\frac{T_{\text{k}}(T + 2m_ec^2)}
    {T(T_{\text{k}} + 2m_ec^2)}\right]^{1/2}.

The two are emitted coplanar, with azimuthal angles differing by :math:`\pi`.
The vacancy is passed to the atomic relaxation model, which follows the full
cascade.

.. _bremsstrahlung_angle:

Bremsstrahlung Emission
-----------------------

The photon energy is sampled from the spectra tabulated in the library, whose
incident-energy grids are again sparse and are interpolated logarithmically,
with unit-base scaling because the spectrum is self-similar: its upper endpoint
is the incident energy itself.

The evaluation carries no angular information for this channel at all, so the
emission angle must come from a model. OpenMC samples it from formula 2BS of
Koch_ and Motz, the screened Schiff form of the Bethe-Heitler cross section,
using rejection in the reduced angle :math:`y = \gamma\theta` where
:math:`\gamma` is the Lorentz factor of the incident electron. The screening
enters through

.. math::
    :label: brems-screening

    Z_{\text{s}} = \left(\frac{Z_{\text{eff}}}{111}\right)^2, \quad
    Z_{\text{eff}}^2 = Z(Z+1),

in which the factor :math:`(Z+1)` accounts for electron-electron
bremsstrahlung. The emission is confined to a cone of order :math:`1/\gamma`
at relativistic energies and becomes broad as the electron energy falls.

The emitting electron is left undeflected. Its direction is governed by elastic
scattering, beside which the recoil from radiative emission is negligible; the
same choice is made in PENELOPE, EGSnrc and MCNP.

Energy Cutoff
-------------

Transport stops when the kinetic energy falls below the electron or positron
cutoff. The residual kinetic energy is deposited at that point rather than
discarded, so that energy is conserved exactly regardless of where the cutoff
is placed. A positron reaching the cutoff annihilates first, emitting two
photons of :math:`m_ec^2` isotropically in opposite directions, as described in
`Electron-Positron Annihilation`_.

A practical caution applies at the very bottom of the range. Elastic scattering
transfers no energy, electroionization stops once the incident energy falls
below the binding energy of the least-bound shell, and excitation can also
vanish at some energy above 10 eV depending on the element. Between those
thresholds an electron can take a very large number of elastic steps without
losing energy. Setting the cutoff no lower than about 12 eV avoids this.

.. _Koch: https://doi.org/10.1103/RevModPhys.31.920

.. _Kaltiaisenaho: https://aaltodoc.aalto.fi/bitstream/handle/123456789/21004/master_Kaltiaisenaho_Toni_2016.pdf

.. _Salvat: https://doi.org/10.1787/32da5043-en

.. _Sternheimer: https://doi.org/10.1103/PhysRevB.26.6067
