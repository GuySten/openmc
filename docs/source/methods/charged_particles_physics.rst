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

The inelastic interaction data are those of the EPICS evaluated libraries, in
particular the Evaluated Electron Data Library (EEDL), read from the eprdata
ACE format: the average energy loss to excitation, and the electroionization
cross sections and knock-on spectra for each subshell, all on a common dense
energy grid.

Elastic scattering and bremsstrahlung are taken from calculated datasets
instead, described in their sections below. The evaluated cross sections for
both are sound, but the distributions that go with them are tabulated on far
too few incident energies to interpolate -- the elastic angular distributions
on 14 to 16 energies per element spanning ten decades, the bremsstrahlung
spectra on nine, with nothing at all between 12.25 MeV and 100 GeV.

Every cross section is the integral of the distribution that is sampled from
it. Rate and shape come from one table in each channel, so an electron collides
at the rate implied by the deflections and energy losses it then gets.

Elastic Scattering
------------------

Elastic scattering changes the direction of the electron without transferring
energy to the atom. It is by a wide margin the most frequent interaction, and
the accumulation of many small deflections is what limits how deeply electrons
penetrate.

The differential cross section is a Dirac partial-wave calculation, tabulated
over the whole angular range on 375 angles at each of 96 incident energies from
50 eV to 100 MeV. ELSEPA writes 606 angles; the wide-angle end of its grid,
where the steps are a uniform half a degree, is thinned to one point in four
before the data is shipped, which changes the integrated, first and second
transport cross sections by at most 5 parts in 10\ :sup:`4`. Nothing is split out of it: the cross section that sets the
distance to the next elastic collision is the integral of the same table the
deflection is sampled from,

.. math::
    :label: elastic-integral

    \sigma_{\text{el}}(T) = 2\pi \int_0^2
    \frac{d\sigma}{d\Omega}\, d(1 - \cos\theta).

There is no forward-peak cutoff, no analytic screened-Rutherford tail beyond
it, and no transport cross section used to rescale the sampled deflection. The
distribution is tabulated as a density rather than differentiated from a
cumulative, which is where histogram binning of an evaluated table loses one to
two per cent of its first moment.

The tables are interpolated logarithmically in energy: both bracketing
distributions are sampled at the same cumulative probability and the two
deflections are combined geometrically,

.. math::
    :label: elastic-interp

    1 - \mu = (1 - \mu_i)^{1-f}\,(1 - \mu_{i+1})^{f}, \quad
    f = \frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)}.

A linear blend of two distributions a decade apart would put essentially all of
its weight on the lower, wider-angle one.

Above 100 MeV the cross section is clamped to its value there.

Partial-Wave Elastic Data
-------------------------

The differential cross sections above are computed with ELSEPA_, the Dirac
partial-wave code of Salvat, Jablonski and Powell, and are distributed with
OpenMC for :math:`Z = 1` to 99 on PENELOPE's 96-point energy grid.

The projectile moves in a central field built from four terms,

.. math::
    :label: elsepa-potential

    V(r) = V_{\text{nuc}}(r) + V_{\text{st}}(r) + V_{\text{ex}}(r) +
    V_{\text{cp}}(r),

the electrostatic potential of a Fermi nuclear charge distribution, that of the
Dirac-Fock electron density of the free neutral atom, the Furness-McCarthy
local approximation to electron exchange, and a correlation-polarization
potential in the local density approximation. There is no imaginary absorptive
term: the inelastic channels are transported explicitly rather than removed
from the elastic flux.

The radial Dirac equation is solved in that field for each partial wave,
giving the phase shifts :math:`\delta_l^{+}` and :math:`\delta_l^{-}` for the
two spin-orbit couplings :math:`j = l \pm 1/2`. These fix the direct and
spin-flip scattering amplitudes,

.. math::
    :label: elsepa-amplitudes

    \begin{aligned}
    f(\theta) &= \frac{1}{2ik} \sum_{l=0}^{\infty} \left\{
    (l + 1)\left[e^{2i\delta_l^{+}} - 1\right] +
    l\left[e^{2i\delta_l^{-}} - 1\right] \right\} P_l(\cos\theta), \\
    g(\theta) &= \frac{1}{2ik} \sum_{l=1}^{\infty} \left[
    e^{2i\delta_l^{-}} - e^{2i\delta_l^{+}} \right] P_l^1(\cos\theta),
    \end{aligned}

where :math:`k` is the relativistic wave number, and the differential cross
section follows as

.. math::
    :label: elsepa-dcs

    \frac{d\sigma}{d\Omega} = |f(\theta)|^2 + |g(\theta)|^2.

This is the table that :eq:`elastic-integral` integrates and that the transport
samples. At high energies the series converges too slowly to sum directly, and
the amplitude is factorized using its Born approximation instead.

Only the sign of the projectile's charge distinguishes a positron, but the two
sets of phase shifts are not interchangeable. Their integrated cross sections
agree to better than a per cent, which is the Born limit and is symmetric in
the charge, so a check on the collision rate alone would not notice the
difference. Their first moments do not agree at all: a positron is repelled by
the nucleus and stays out of the small-impact-parameter region that produces
the large deflections, so its transport cross section is 0.80 of the
electron's for lead at 21 MeV, 0.65 at 1 MeV, and 0.35 at 1 keV. For carbon the
same ratios are 0.98, 0.97 and 0.67. Both tables are therefore carried, and the
transport selects on the sign of the charge.

These are the settings under which PENELOPE's own database is built; for carbon
the two agree to four decimal places in the integrated and in both transport
cross sections from 100 keV to 100 MeV. The file is regenerated by the
``make_elastic_dpwa.py`` script, which records the settings it used.

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

Electroionization ejects a bound electron: Moller scattering when the
projectile is an electron, Bhabha scattering when it is a positron. The
subshell is sampled in proportion to the subshell ionization cross sections, and the kinetic energy
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

Distant and close collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both polar deflections follow from the recoil energy :math:`Q` the collision
leaves with the atom, and from nothing else. Writing :math:`p` and :math:`p'`
for the momenta of the projectile before and after, :math:`W = T_{\text{k}} + B`
for the energy it gave up and :math:`(cq)^2 = Q(Q + 2m_ec^2)` for the momentum
transfer,

.. math::
    :label: ionization-angles

    1 - \mu = \frac{(cq)^2 - (cq_-)^2}{2\,pc\,p'c},
    \qquad
    1 - \mu_{\text{k}} = \frac{(cq - cq_-)(pc + p'c - cq)}{2\,pc\,cq},

where :math:`cq_- = pc - p'c` is the smallest momentum the collision can hand
over, reached when the projectile is not deflected at all. The projectile is
deflected through the momentum transfer and the knock-on leaves along it; the
two are emitted coplanar, with azimuthal angles differing by :math:`\pi`.

Setting :math:`Q = W` recovers the free binary collision, in which these reduce
to the familiar Moller pair

.. math::
    :label: ionization-angles-free

    \mu = \left[\frac{T'(T + 2m_ec^2)}{T(T' + 2m_ec^2)}\right]^{1/2},
    \quad
    \mu_{\text{k}} = \left[\frac{W(T + 2m_ec^2)}
    {T(W + 2m_ec^2)}\right]^{1/2}.

Note both are set by the energy the projectile transferred, :math:`W`, and not
by the kinetic energy the knock-on is left with: the atom absorbs the binding
energy :math:`B` but carries away negligible momentum. Deflecting the knock-on
by :math:`T_{\text{k}}` instead would eject it too far sideways, by 22% of the
incident momentum for a tantalum K shell ionised at 100 keV.

A free electron takes up the whole transfer, but a bound one does not. While
the momentum transfer stays below the scale of the subshell the atom is excited
as a whole, through a dipole-like interaction whose recoil is far smaller than
the binary value. Following PENELOPE, that scale is the resonance energy
:math:`W_i` of the Sternheimer-Liljequist oscillator standing for the subshell,

.. math::
    :label: oscillator-resonance

    W_i^2 = \rho^2 U_i^2 + \tfrac{2}{3} f_i \Omega_{\text{p}}^2,

with :math:`U_i` the binding energy, :math:`f_i` the share of the material's
electrons the subshell holds, :math:`\Omega_{\text{p}}` the plasma energy and
:math:`\rho` the Sternheimer adjustment that makes
:math:`\sum_i f_i \ln W_i = \ln I`. These are properties of the material, not
of the atom alone, and they are the same oscillators the density-effect
correction is built from; no data beyond what the photon library already
carries is needed.

A distant interaction is split as PENELOPE splits it. The transverse part
carries no momentum at all, :math:`Q = Q_-`, and the longitudinal part a recoil
distributed as :math:`1/[Q(Q + 2m_ec^2)]` between :math:`Q_-` and :math:`W_i`.
The two are weighted by their cross sections, whose common factor
:math:`f_i/W_i` cancels, leaving

.. math::
    :label: distant-split

    \frac{P_{\text{tra}}}{P_{\text{lon}}} =
    \frac{\ln[1/(1-\beta^2)] - \beta^2 - \delta}
    {\ln\!\left[\dfrac{W_i(Q_- + 2m_ec^2)}{Q_-(W_i + 2m_ec^2)}\right]},

with :math:`\delta` the density-effect correction of the material.

Which of the two occurred is decided by how much of the evaluated cross section
at the sampled transfer the free binary collision can account for,

.. math::
    :label: close-probability

    P_{\text{close}}(W) = \min\left(1,
    \frac{d\sigma_{\text{free}}/dW}{d\sigma_{\text{eval}}/dW}\right).

PENELOPE instead cuts at :math:`W_i`, which is exact for its own delta
oscillator because that places all distant strength at exactly :math:`W_i`. The
evaluated spectra spread it over a range of :math:`W`, so the cut would hand
close kinematics to the part lying above the resonance, and there is a good deal
of it: for the carbon L\ :sub:`3` shell a quarter of the collisions land above
:math:`W_i` where the free cross section can account for a sixteenth. Deciding
pointwise leaves the close channel carrying the free cross section it should
and the rest distant, without touching the energy spectrum.

For a positron the evaluated spectrum is reweighted by the Bhabha-to-Moller
ratio below and the free cross section is Bhabha's, so the ratio cancels out of
:eq:`close-probability` and one Moller form serves both charges. This is the
same conclusion PENELOPE reaches, its distant interactions being identical for
the two.

The pair is still not exactly momentum-conserving, since the knock-on leaves
with the momentum of :math:`T_{\text{k}}` rather than of :math:`W`; no
free-electron model of a bound target can conserve both. The vacancy is passed
to the atomic relaxation model, which follows the full cascade.

Excitation is left undeflected. It is a distant interaction and so has a real
momentum transfer, but the split between the evaluated excitation and
ionization channels is not smooth in :math:`Z` -- copper's excitation cross
section is a hundredth of aluminium's, while their sums over both channels are
comparable -- so treating the two differently would import that into the
transport cross section. What it costs is bounded by the whole distant
contribution to the transport cross section, which for an atom of :math:`Z`
electrons is :math:`2\pi r_e^2 (m_ec^2)^2 Z / [\beta^2 (pc)^2]` -- independent
of every resonance energy, since :math:`\sigma_{\text{dist}}` goes as
:math:`f_i/W_i` while the deflection goes as :math:`W_i`. That is 3.6% of the
total transport cross section for hydrogen at 4.27 MeV, 0.32% for carbon and
under 0.05% for tungsten, so it matters in hydrogenous media and essentially
nowhere else.

A positron is given the same spectra, reweighted. The two processes differ:
the electrons of a Moller collision are indistinguishable, so the faster is
labelled the primary and the transfer stops at :math:`(T-B)/2`, while a
positron and the electron it ejects are distinguishable and the transfer runs
to :math:`T`. Below that limit the reweighting is by the ratio of the two free
cross sections,

.. math::
    :label: bhabha-moller-ratio

    R(\varepsilon) = \frac{d\sigma_{\text{B}}/d\varepsilon}
    {d\sigma_{\text{M}}/d\varepsilon}, \qquad \varepsilon = W/T,

which is applied by rejection during the collision, the tabulated cross section
having been raised beforehand to :math:`\max_\varepsilon R` so that it remains
a majorant of the true one. Nothing has to integrate :math:`R` over the
evaluated spectrum for this to come out right.

The ratio is better founded than either cross section alone. The final state is
the same for both projectiles -- an electron ejected with :math:`W - B`,
leaving one vacancy -- so in the Bethe decomposition the close-collision cross
section factorises into a target part, the generalised oscillator strength, and
a projectile part, and only the projectile part distinguishes Moller from
Bhabha. Whatever binding does to one it does to the other, and it cancels in
:math:`R`. This is the assumption PENELOPE makes in applying free Moller and
Bhabha cross sections oscillator by oscillator, and it is far weaker than
requiring either to be free. It also needs no threshold: both cross sections
tend to Rutherford's :math:`1/\varepsilon^2` as :math:`\varepsilon \to 0`, so
:math:`R \to 1` and the correction switches itself off exactly where the free
picture stops being trustworthy. Relativistically :math:`R = 1 - 2\varepsilon`
to good accuracy.

Above the Moller limit the evaluated spectra stop, so there is nothing to
reweight and the free Bhabha cross section is integrated and sampled directly.
That range lies far above every binding energy by construction, which is
exactly where the free description is right.

Against the ICRU-37 collision stopping power, the ratio of positron to electron
comes out at 0.981 for carbon at 1.26 MeV and 0.989 for lead at 1 MeV, against
reference values of 0.977 and 0.972. Sampling the electron spectra unchanged,
as an evaluated library on its own obliges one to do, would give 1.000.

A free Bhabha cross section integrated from each binding energy upward is
*not* a substitute for the whole channel: it comes to 0.47 to 0.53 of the
ICRU-37 collision stopping power for both carbon and lead at every energy,
having no distant collisions at all.

.. _bremsstrahlung_angle:

Bremsstrahlung Emission
-----------------------

The photon energy is sampled from the Seltzer-Berger scaled cross sections
:math:`\chi(Z, T, \kappa) = (\beta^2/Z^2)\, k\, d\sigma/dk`, tabulated in
barns against the reduced photon energy :math:`\kappa = k/T`. This is the same
table the thick-target approximation uses, carried in the photon library, and
it is given on 57 incident energies against the evaluation's nine.

Since :math:`\chi` is finite at :math:`\kappa = 0`, the whole of the
:math:`1/k` divergence of :math:`d\sigma/dk = (Z^2/\beta^2)\chi(\kappa)/k`
is explicit. The photon energy is therefore sampled by drawing :math:`k` from
:math:`1/k` over :math:`[k_{\text{cut}}, T]` and accepting it with probability
:math:`\chi(\kappa)/\chi_{\text{max}}`, with :math:`\chi` linear in
:math:`\kappa` between tabulated points and between the two bracketing
incident energies.

Bremsstrahlung has no threshold-free cross section -- the number of photons
emitted diverges logarithmically as :math:`k \to 0` -- so one has to be
chosen. The cross section stored in the electron library is the integral of the
same density above the same :math:`k_{\text{cut}}`,

.. math::
    :label: brems-integral

    \sigma_{\text{br}}(T) = \frac{Z^2}{\beta^2}
    \int_{\kappa_{\text{cut}}}^{1} \frac{\chi(\kappa)}{\kappa}\,
    d\kappa,

evaluated in closed form on each interval of the tabulated :math:`\kappa`
grid, and the threshold it used is stored beside it so that the transport
samples above the same one. The product of the two is the radiative stopping
power, which reproduces the ESTAR tabulation to better than one per cent for
carbon and for lead from 0.1 to 100 MeV.

A positron radiates less than an electron of the same energy, being repelled by
the nucleus rather than attracted to it, and its cross section is
:eq:`brems-integral` scaled by Salvat's factor :math:`F_{\text{p}}(Z,T)` of
:eq:`positron-factor` -- the same one the thick-target approximation applies.
The factor does not depend on :math:`\kappa`, so it scales the rate and leaves
the spectrum sampled above untouched. It is not a small correction at high
:math:`Z`: for lead it is 0.49 at 1 MeV and 0.84 at 21 MeV, against 0.96 and
1.00 for carbon.

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

.. _ELSEPA: https://www.sciencedirect.com/science/article/pii/S0010465504004795

.. _Koch: https://doi.org/10.1103/RevModPhys.31.920

.. _Kaltiaisenaho: https://aaltodoc.aalto.fi/bitstream/handle/123456789/21004/master_Kaltiaisenaho_Toni_2016.pdf

.. _Salvat: https://doi.org/10.1787/32da5043-en

.. _Sternheimer: https://doi.org/10.1103/PhysRevB.26.6067
