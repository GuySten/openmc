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
transported explicitly, as described in :ref:`electron_transport` below. The
interaction physics is the same either way; what differs is how much of it is
simulated one collision at a time.

In the analog single-event scheme, every elastic, excitation,
electroionization and bremsstrahlung interaction is sampled individually.
Nothing is condensed and no multiple-scattering theory is used, so no
assumption is made about the step length and the scheme remains valid where
condensed-history schemes break down, which is generally below about 1 keV.
It is also expensive: a 1 MeV electron undergoes on the order of
:math:`10^5` interactions before it ranges out.

The default is a mixed, or class II, condensed-history scheme built on top of
that same physics, described in `Condensed History`_. The interactions too
weak to be worth following individually are grouped into a step and replaced
by their first two moments; everything else is still sampled one collision at
a time, exactly as the single-event scheme would. How much is grouped is set
by the ``deflection`` and ``energy_loss`` cutoffs of the ``<cutoff>`` element,
and setting ``deflection`` to zero groups nothing and recovers single-event
transport exactly.

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

The three interaction channels below are what the transport is built from,
under either scheme. In the analog single-event scheme the distance to the
next interaction is sampled from the total electroatomic cross section, one of
the three channels is selected in proportion to its cross section, and that
interaction is sampled in full; nothing is condensed into a step. Under the
default mixed scheme the same channels are split into a grouped part and a
part still sampled this way, as `Condensed History`_ describes; the sections
below describe the physics of each channel, and the splitting is described
there.

Elastic scattering and bremsstrahlung are properties of the atom and are
looked up per element. Inelastic collisions are not: the oscillators they are
sampled from are shares of all the medium's electrons and their resonance
energies are fixed by its mean excitation energy, so that channel is the
material's and is chosen before any element is.

The electroionization cross sections of the EPICS evaluated libraries, in
particular the Evaluated Electron Data Library (EEDL) read from the eprdata
ACE format, are what the inner-shell oscillators are renormalized to. They
set the rate at which the deeply bound shells are ionized and so the
characteristic x-ray yields; the shape of the energy loss does not come from
them.

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

Inelastic Collisions
--------------------

Excitation and ionization are one channel, not two. Both are the projectile
losing energy to the atom's electrons, and both are described by the same
generalized oscillator strength (GOS); which one a collision was is a matter
of whether the transfer exceeded a binding energy. OpenMC uses the
Sternheimer-Liljequist model of that GOS, following PENELOPE_.

The Oscillator Model
~~~~~~~~~~~~~~~~~~~~

Every electron of the medium is assigned to an oscillator, one per
electroionization subshell of each element, holding :math:`f_i` electrons
bound at :math:`U_i` and resonating at

.. math::
    :label: oscillator-resonance

    W_i^2 = \rho^2 U_i^2 + \tfrac{2}{3} f_i \Omega_{\text{p}}^2,

where :math:`\Omega_{\text{p}}` is the plasma energy of the medium and
:math:`\rho` is the Sternheimer adjustment, solved so that

.. math::
    :label: sum-rules

    \sum_i f_i = Z, \qquad \sum_i f_i \ln W_i = \ln I .

The second of these is the Bethe sum rule, and it is what makes the model
quantitative: with it satisfied, the stopping power the model integrates to
*is* the Bethe stopping power, with the mean excitation energy :math:`I` of
the medium and no free parameters left. The first fixes the total number of
electrons. Both are properties of the medium rather than of any atom in it,
which is why the inelastic channel belongs to the material: the same carbon
atom has different oscillators in graphite and in methane.

The only experimental input is :math:`I`, taken from the same ESTAR/ICRU 37
table the stopping powers are compared against. The binding energies come
from the photon library. Nothing else is needed, and in particular no
tabulated inelastic differential cross section: the evaluated knock-on
spectra are not used for the shape of the loss.

Distant and Close Collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each oscillator contributes three channels, which are PENELOPE's.

A *distant longitudinal* collision excites the oscillator at exactly
:math:`W = W_i` and hands over a recoil :math:`Q` distributed as
:math:`1/[Q(Q+2m_ec^2)]` between the minimum momentum transfer :math:`Q_-`
and :math:`W_i`. A *distant transverse* collision also costs :math:`W_i` but
carries no momentum at all, :math:`Q = Q_-`; this is the channel the density
effect acts on. Their cross sections are

.. math::
    :label: distant-xs

    \sigma_{\text{lon}} = \frac{2\pi r_e^2 m_ec^2}{\beta^2}\,
      \frac{f_i}{W_i} \ln\!\left[\frac{W_i(Q_-+2m_ec^2)}
      {Q_-(W_i+2m_ec^2)}\right],
    \qquad
    \sigma_{\text{tra}} = \frac{2\pi r_e^2 m_ec^2}{\beta^2}\,
      \frac{f_i}{W_i}
      \left[\ln\frac{1}{1-\beta^2} - \beta^2 - \delta\right],

with :math:`\delta` the density-effect correction of Sternheimer_, computed
from the same oscillator table. A *close* collision is the free binary one:
Moller scattering for an electron, Bhabha for a positron, with the whole
transfer left as recoil, :math:`Q = W`, and the transfer running from
:math:`W_i` to the kinematic limit.

The three sum to the oscillator's total cross section, and their first
moments sum to its stopping power. Summed over oscillators and with the sum
rules of :eq:`sum-rules` in force, the result reproduces the Bethe stopping
power analytically -- this is the point of the construction, not a fitted
outcome, and OpenMC checks it at startup against an independent evaluation of
the ICRU 37 formula and refuses to run if the two disagree by more than one
part in a million.

Very close to threshold the distant channel would exceed what the kinematics
allow. PENELOPE's remedy is adopted unchanged: below :math:`3W_i - 2U_i` the
resonance used for the distant channels is lowered to :math:`(T+2U_i)/3` and
the recoil ceiling with it, which keeps the cross section positive and the
first moment right.

Renormalization
~~~~~~~~~~~~~~~

A delta oscillator is a poor description of an inner shell. Characteristic
x-ray yields rest directly on the rate at which those shells are ionized, and
that rate is evaluated data OpenMC already carries. So, as PENELOPE does, the
oscillators standing for shells bound above the transport cutoffs have their
cross sections scaled to the evaluated subshell ionization cross sections,

.. math::
    :label: inner-renorm

    c_i(T) = \frac{\sigma_i^{\text{eval}}(T)}
                  {f_i\,\sigma_i^{\text{model}}(T)},

every moment of the oscillator being scaled by the same factor so the channel
stays one cross section.

That scaling moves the total off the Bethe value, and the second half of
PENELOPE's scheme puts it back: the remaining oscillators are scaled by a
single factor, chosen so the total collision stopping power is the ICRU 37
value exactly. The compensation is small -- in the range 0.97 to 1.11 for the
elements tested, the inner shells carrying 8 to 20 per cent of the stopping
power -- and it is what lets the model have both the right x-ray yields and
the right stopping power. Both halves are needed; renormalizing every
subshell without the compensation shifts the stopping power by +82 per cent
in carbon and -24 per cent in lead.

Sampling a Collision
~~~~~~~~~~~~~~~~~~~~

Which oscillator a collision was with is drawn from the tabulated cumulative
of the discrete rates, then one of the three channels from their cross
sections, then :math:`W` and :math:`Q` from that channel. Both polar
deflections follow from :math:`Q` alone: writing :math:`p` and :math:`p'` for
the momenta of the projectile before and after and :math:`(cq)^2 =
Q(Q+2m_ec^2)` for the momentum transfer,

.. math::
    :label: ionization-angles

    1 - \mu = \frac{(cq)^2 - (cq_-)^2}{2\,pc\,p'c},
    \qquad
    1 - \mu_{\text{k}} = \frac{(cq - cq_-)(pc + p'c - cq)}{2\,pc\,cq},

where :math:`cq_- = pc - p'c` is the smallest momentum the collision can hand
over, reached when the projectile is not deflected at all. The projectile is
deflected through the momentum transfer and the knock-on leaves along it; the
two are emitted coplanar, with azimuthal angles differing by :math:`\pi`.
Setting :math:`Q = W` recovers the free binary collision, in which these
reduce to the familiar Moller pair

.. math::
    :label: ionization-angles-free

    \mu = \left[\frac{T'(T + 2m_ec^2)}{T(T' + 2m_ec^2)}\right]^{1/2},
    \quad
    \mu_{\text{k}} = \left[\frac{W(T + 2m_ec^2)}
    {T(W + 2m_ec^2)}\right]^{1/2}.

Both are set by the energy the projectile transferred, :math:`W`, and not by
the kinetic energy the knock-on is left with: the atom absorbs the binding
energy :math:`U_i` but carries away negligible momentum. Deflecting the
knock-on by :math:`W - U_i` instead would eject it too far sideways, by 22 per
cent of the incident momentum for a tantalum K shell ionised at 100 keV. The
pair is still not exactly momentum-conserving, since the knock-on leaves with
the momentum of :math:`W - U_i` rather than of :math:`W`; no free-electron
model of a bound target can conserve both.

A collision ionizes the atom when the transfer exceeds the binding energy of
the subshell its oscillator stands for, and the vacancy is then passed to the
atomic relaxation model, which follows the full cascade. Otherwise the atom
was excited as a whole and the energy is deposited locally. Because the
oscillator is a real subshell, which shell the vacancy is in follows from the
same draw that chose the oscillator rather than from a separate model.

The Positron
~~~~~~~~~~~~

The distant channels are identical for the two projectiles -- the atom cannot
tell what excited it -- and the close channel is Bhabha's rather than
Moller's, which differs both in shape and in running to :math:`T` instead of
:math:`(T-U_i)/2`. Everything above is therefore tabulated and sampled
separately per projectile charge, with no reweighting and no rejection.
Against the ICRU 37 collision stopping power the ratio of positron to
electron comes out at 0.981 for carbon at 1.26 MeV and 0.989 for lead at
1 MeV, against reference values of 0.977 and 0.972.

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

Condensed History
-----------------

Following every interaction of an electron is exact and slow. Most of those
interactions barely change the particle: an elastic collision that turns it
through a fraction of a degree, an ionization that costs it a few tens of eV.
The mixed, or class II, scheme of Berger separates the interactions by how
much they matter rather than by kind. Those above a cutoff -- the *hard*
interactions -- are sampled individually, exactly as the single-event scheme
samples them. Those below it -- the *soft* interactions -- are grouped into a
step and replaced by the first two moments of what they would have done.

The step runs from one hard interaction to the next. Over it the grouped
collisions contribute one deflection and one energy loss, sampled from the
artificial distributions below, and each channel's cutoff is what decides how
much of it is grouped.

Where the Cutoffs Come From
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Elastic scattering is cut by angle. The ``deflection`` cutoff, PENELOPE's
:math:`C_1`, is the largest mean deflection :math:`\langle 1 - \mu \rangle`
the grouped collisions of one step may accumulate. It is applied to the step
rather than obtained from it: in PENELOPE the step is one hard mean free path
by construction, so the soft deflection over it is
:math:`\sigma_{1,\text{soft}}/\sigma_{\text{hard}}`; here the step is a
sampled flight, which runs past a mean free path as often as not and is cut
short by the energy ceiling and by the geometry besides, so the bound is
imposed on the length directly.

The inelastic channels are cut by energy, and not by a parameter of their own.
A collision may be grouped only when nothing it would have produced would have
been transported anyway, so the soft cutoff :math:`W_{cc}` of each channel is
the transport cutoff of the secondary it makes: the electron cutoff for a
knock-on, the photon cutoff for a bremsstrahlung photon. Raising those cutoffs
therefore groups more. A second bound limits how much: no single grouped
collision may carry more than
:math:`\texttt{MAX\_SOFT\_LOSS\_SHARE} = 1/10` of the step's own energy
budget, since a step describing its loss by a mean and a variance cannot have
either resting on one event.

The step is bounded in length by both cutoffs, by the ``energy_loss``
fraction of the kinetic energy (PENELOPE's :math:`C_2`), by the distance to
the nearest boundary, and by the energy left above the particle's own
transport cutoff. Both fractions are capped at
:math:`\texttt{MAX\_STEP\_COARSENESS} = 0.2`, where PENELOPE caps them: past
that a step turning the particle through some 37 degrees at one point is no
longer describing a path, and a step taking a fifth of the kinetic energy has
moved far enough that the cross sections it began with belong to a different
particle.

A step is taken only if it would group at least
:math:`\texttt{MIN\_GROUPED\_COLLISIONS} = 30` collisions; otherwise the
transport falls back to sampling them individually for that step. Two moments
describe a sum of :math:`N` collisions to about :math:`1/\sqrt{N}` and no
better, and a step that removes ten collisions was not going to be much faster
than simulating them. This is what decides, with no input from the user, where
a run stops being condensed history: in a thin region, near an interface,
wherever the geometry cuts the step short, and at low energy, where collisions
are violent enough that few of them fit under the angular ceiling. A 10 keV
electron in carbon fits four and is transported one collision at a time; a
1 MeV one fits 150.

The Grouped Deflection
~~~~~~~~~~~~~~~~~~~~~~

Over a path :math:`s` the Legendre moments of the accumulated soft deflection
decay as

.. math::
    :label: ch-moment-decay

    \langle \mu \rangle = e^{-s/\lambda_1}, \qquad
    \langle P_2(\mu) \rangle = e^{-s/\lambda_2},

with :math:`1/\lambda_\ell = n\sigma_\ell`, where :math:`\sigma_\ell` is the
:math:`\ell`-th transport cross section of everything a step groups. Elastic
scattering supplies most of it, from the soft part of its own distribution,
but not all: the grouped inelastic collisions deflect too, through the recoil
they hand the atom, and in carbon that is a quarter of what elastic
scattering contributes -- against a per cent or two in tungsten, where
:math:`Z^2` puts nuclear elastic scattering far ahead.

What a class II scheme needs from the grouped collisions is those two moments,
so the distribution used to carry them is the simplest one that carries both
exactly: two uniform pieces meeting at :math:`\mu_0`, the lower one carrying
probability :math:`a`, with

.. math::
    :label: ch-artificial-angular

    \mu_0 = \frac{3\langle\mu^2\rangle - 1}{2\langle\mu\rangle}, \qquad
    a = \frac{1 + \mu_0 - 2\langle\mu\rangle}{2}.

Its shape means nothing; anything more elaborate would assert detail the
moments do not contain. Both moments come back exactly over the range in which
:math:`a` stays a probability, which is every pair a real angular distribution
can produce. Outside it the first moment is still exact and the second is as
close as the form allows, which is the right way round: the first is what sets
the transport mean free path. This is PENELOPE's choice.

The Grouped Energy Loss
~~~~~~~~~~~~~~~~~~~~~~~

Over the same path the grouped collisions take a mean :math:`sS` and a
variance :math:`s\Omega`, where :math:`S` is the restricted stopping power and
:math:`\Omega` the restricted straggling parameter, both integrals of the soft
part of the inelastic cross sections:

.. math::
    :label: ch-restricted-moments

    S = n \int_0^{W_{cc}} W \frac{d\sigma}{dW}\, dW, \qquad
    \Omega = n \int_0^{W_{cc}} W^2 \frac{d\sigma}{dW}\, dW.

Matching that mean and variance is the whole content of a restricted stopping
power with straggling, so again the simplest distribution carrying both is
used. Which one that is depends on how wide the loss is relative to its mean:
a uniform distribution can reach a variance of
:math:`\langle\omega\rangle^2/3` before it would have to go negative, and past
that the distribution becomes a uniform piece with an atom at zero -- the step
either loses nothing or loses a good deal, which is what a broad straggling
distribution physically is.

Because every transfer in those integrals is under the soft cutoff,
:math:`\Omega/S \le W_{cc}`, and the cutoff is in turn held under a tenth of
the step's budget, so the sampled loss cannot overshoot its mean by more than
a factor :math:`\texttt{MAX\_SOFT\_LOSS\_OVERSHOOT} = 2`. That bound is what
the hard cross section is bounded over.

The Random Hinge
~~~~~~~~~~~~~~~~

The grouped deflection and the grouped energy loss are applied at a single
point drawn uniformly along the step. Putting the whole deflection at the end
would leave the particle travelling in a straight line for the length of the
step and lose the lateral spread; putting it at the start would overstate it.
Drawing the point uniformly reproduces the correct mean lateral displacement
to first order. This is PENELOPE's random hinge.

The grouped energy loss is deposited at a point drawn uniformly inside each
leg of the step, which reproduces the profile of a constant deposition rate
along the path but not its shape within one step. A ``heating`` tally on a
mesh much finer than the step length therefore sees a deposition spread
correctly on average and not within a step.

The Hard Cross Section and the Delta Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hard cross section changes along the step as the particle loses energy, so
the distance to the next hard interaction cannot be sampled from its value at
the start. Instead a majorant :math:`\Sigma_{\max}` bounding it over the whole
energy window the step may reach is used, the distance is sampled from that,
and on arrival the interaction is accepted with probability
:math:`\Sigma_{\text{hard}}(E)/\Sigma_{\max}`. A rejected draw is a *delta
interaction*: the particle is left untouched and the next distance is sampled
from the same majorant. This is Woodcock tracking in energy rather than in
space, and it is exact for any valid majorant. The majorant is tabulated per
material over the energy grid and taken as the larger of the bounds at the two
grid points bracketing the step's window, so that it bounds the cross section
between them as well as at them. A runtime warning is issued if it is ever
found not to bound.

The Collision Stopping Power
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The soft and hard inelastic channels are two halves of one cross section, cut
at :math:`W_{cc}`: each oscillator's moments are integrated below the cutoff
for the grouped channel and above it for the discrete one, from the model of
`Inelastic Collisions`_. Nothing is pinned or rescaled at the seam, because
there is nothing to pin -- the two halves come from the same differential
cross section, so their first moments sum to the unrestricted total whatever
the cutoff is, and that total is the ICRU 37 collision stopping power by
construction.

This is checked rather than assumed. At startup the sum of the two halves is
compared, at every point of the energy grid and for both projectile charges,
against an independent evaluation of the ICRU 37 formula and against the
restricted Berger-Seltzer stopping power paired with the analytic Moller or
Bhabha tail. A disagreement beyond one part in a million aborts the run: the
same zero a missing table would give is what a wrong stopping power would
look like downstream, and nothing there could tell them apart.

The straggling is the second moment of the same restricted cross section, so
it too needs no separate model.

Known Differences From Single-Event Transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two schemes are built to agree, and ``deflection = 0`` reproduces the
single-event one exactly, but they are not identical where grouping is active.

* A grouped step deposits its loss at a point drawn uniformly along each leg,
  not continuously.

* The grouped loss is sampled from a two-moment artificial distribution
  matching the mean and variance the restricted cross section gives, not from
  the cross section itself. Only its first two moments are right.

* Tallies that count interactions rather than score energy see the grouped
  collisions as one event rather than as the many they stand for.

.. _ELSEPA: https://www.sciencedirect.com/science/article/pii/S0010465504004795

.. _Koch: https://doi.org/10.1103/RevModPhys.31.920

.. _Kaltiaisenaho: https://aaltodoc.aalto.fi/bitstream/handle/123456789/21004/master_Kaltiaisenaho_Toni_2016.pdf

.. _PENELOPE: https://doi.org/10.1787/32da5043-en

.. _Salvat: https://doi.org/10.1787/32da5043-en

.. _Sternheimer: https://doi.org/10.1103/PhysRevB.26.6067
