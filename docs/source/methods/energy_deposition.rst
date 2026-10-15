.. _methods_heating:

=============================
Heating and Energy Deposition
=============================

As particles traverse a problem, some portion of their energy is deposited at
collision sites. This energy is deposited when charged particles, including
electrons and recoil nuclei, undergo electromagnetic interactions with
surrounding electrons and ions. The information describing how much energy
is deposited for a specific reaction is referred to as
"heating numbers" and can be computed using a program like NJOY with the
``heatr`` module.

The heating rate is the product of reaction-specific coefficients and a reaction
cross section

.. math::

    H(E) = \phi(E)\sum_i\rho_i\sum_rk_{i, r}(E),

and has units energy per time, typically eV/s. Here, :math:`k_{i, r}` are the
KERMA (Kinetic Energy Release in Materials) [Mack97]_ coefficients for reaction
:math:`r` of isotope :math:`i`. The KERMA coefficients have units of energy
:math:`\times` cross-section (e.g., eV-barn) and can be used much like a reaction
cross section for the purpose of tallying energy deposition.

KERMA coefficients can be computed using the energy-balance method with a
nuclear data processing code like NJOY, which estimates the KERMA coefficients
using the following equation:

.. math::

    k_{i, r}(E) = \left(E + Q_{i, r} - \sum\limits_x \bar{E}_{i, r, x}
    \right)\sigma_{i, r}(E),

where the summation is over each secondary particle type :math:`x`. This
equation states that the energy deposited is equal to the energy of the incident
particle plus the reaction :math:`Q` value less the energy of secondary
particles that are transported away from the reaction site. For neutron
interactions, the energy-balance KERMA coefficient is

.. math::

    k_{i, r}(E) = \left(E + Q_{i, r} - \sum\limits_x \bar{E}_{i, r, n}
    - \bar{E}_{i, r, \gamma}\right)\sigma_{i, r}(E),

where :math:`\bar{E}_{i, r, n}` is the average energy of secondary neutrons and
:math:`\bar{E}_{i, r, \gamma}` is the average energy of secondary photons. For
photon and charged particle interactions the KERMA coefficient is

.. math::
    :label: energy-balance-photon

    k_{i, r}(E) = \left(E + Q_{i, r} - \sum\limits_x \bar{E}_{i, r, x}
    \right)\sigma_{i, r}(E).

where the :math:`Q` value is zero for all interactions except for pair
production and positron annihilation.

-------
Fission
-------

During a fission event, there are potentially many secondary particles, and all
must be considered. The total energy released in a fission event is typically
broken up into the following categories:

- :math:`E_{fr}` - kinetic energy of fission fragments
- :math:`E_{n,p}` - energy of prompt fission neutrons
- :math:`E_{n,d}` - energy of delayed fission neutrons
- :math:`E_{\gamma,p}` - energy of prompt fission photons
- :math:`E_{\gamma,d}` - energy of delayed fission photons
- :math:`E_{\beta}` - energy of released :math:`\beta` particles
- :math:`E_{\nu}` - energy of neutrinos

These components are defined in MF=1, MT=458 data in a standard ENDF-6 formatted
file. All these quantities may depend upon incident neutron energy, but this
dependence is not shown to make the following demonstrations cleaner. As
neutrinos scarcely interact with matter, the recoverable energy from fission is
defined as

.. math::

    E_r\equiv E_{fr} + E_{n,p} + E_{n, d} + E_{\gamma, p}
    + E_{\gamma, d} + E_{\beta}

Furthermore, the energy of the secondary neutrons and photons is given as
:math:`E_{n, p}` and :math:`E_{\gamma, p}`, respectively.

NJOY computes the fission KERMA coefficient using this energy-balance method to be

.. math::

    k_{i, f}(E) = \left[E + Q(E) - \bar{E}(E)\right]\sigma_{i, f}(E)
    = \left[E_{fr} + E_{\gamma, p}\right]\sigma_{i, j}(E)

.. note::

    The energy from delayed neutrons and photons and beta particles is intentionally
    left out from the NJOY calculations.

---------------------
OpenMC Implementation
---------------------

For fissile isotopes, OpenMC makes modifications to the heating reaction to
include all relevant components of fission energy release. These modifications
are made to the total heating reaction, MT=301. Breaking the total heating
KERMA into a fission and non-fission section, one can write

.. math::

    k_i(E) = k_{i, nf}(E) + \left[E_{fr}(E) + E_{\gamma, p}\right]\sigma_{i, f}(E)

OpenMC seeks to modify the total heating data to include energy from
:math:`\beta` particles and, conditionally, delayed photons. This conditional
inclusion depends on the simulation mode: neutron transport, or coupled
neutron-photon transport. The heating due to fission is removed using MT=318
data, and then re-built using the desired components of fission energy release
from MF=1,MT=458 data.

Neutron Transport
-----------------

For this case, OpenMC instructs ``heatr`` to produce heating coefficients
assuming that energy from photons, :math:`E_{\gamma, p}` and
:math:`E_{\gamma, d}`, is deposited at the fission site.
Let :math:`N901` represent the total heating number returned from this ``heatr``
run with :math:`N918` reflecting fission heating computed from NJOY.
:math:`M901` represent the following modification

.. math::

    M901_{i}(E)\equiv N901_{i}(E) - N918_{i}(E)
      + \left[E_{i, fr} + E_{i, \beta} + E_{i, \gamma, p}
      + E_{i, \gamma, d}\right]\sigma_{i, f}(E).

This modified heating data is stored as the MT=901 reaction and will be scored
if ``heating-local`` is included in :attr:`openmc.Tally.scores`.

Coupled Neutron-Photon Transport
--------------------------------

Here, OpenMC instructs ``heatr`` to assume that energy from photons is not
deposited locally. However, the definitions provided in the NJOY manual
indicate that, regardless of this mode, the prompt photon energy is still
included in :math:`k_{i, f}`, and therefore must be manually removed.
Let :math:`N301` represent the total heating number returned from this
``heatr`` run and :math:`M301` be

.. math::

    M301_{i}(E)\equiv N301_{i}(E) - N318_{i}(E)
      + \left[E_{i, fr}(E) + E_{i, \beta}(E)\right]\sigma_{i, f}(E).

This modified heating data is stored as the MT=301 reaction and will be scored
if ``heating`` is included in :attr:`openmc.Tally.scores`.

Photons and Charged Particles
-----------------------------

In OpenMC, energy deposition from photons or charged particles is scored using
the energy balance method based on Equation :eq:`energy-balance-photon`. Special
consideration is given to electrons and positrons as described below.

+++++++++++++++++
Charged Particles
+++++++++++++++++

OpenMC tracks photons interaction by interaction so the energy deposited in each
collision is easily attributed back to the nuclide and reaction for which the
photon interacted with. Charged particles (electrons and positrons) aren't
tracked in the same way. For charged particles, OpenMC assumes that all their
energy (less the energy of bremsstrahlung radiation) is deposited in the
material in which they were born. Charged particles created by photon
interactions are therefore accounted for in the parent photon collision for
heating tallies. In this way it is harder to trace how much energy should be
attributed in each nuclide.

According to the CSDA approximation (see :ref:`ttb`) the energy deposited by a
charged particle with kinetic energy :math:`T` in the :math:`i`-th element can
be calculated as:

.. math::

    E_{i} = \int_{0}^{R(T)} w_{i}S_{\text{col,i}} dx

where :math:`R(T)` is the CSDA range of the charged particle,
:math:`S_{\text{col},i}` is the collision stopping power of the charged particle
in the :math:`i`-th element and :math:`w_i` is the mass fraction of the
:math:`i`-th element. According to the Bethe formula the collision stopping
power of the :math:`i`-th element is proportional to :math:`Z_i/A_i`, so the
fractional collision stopping power from the :math:`i`-th element is:

.. math::

    \frac{w_{i}S_{\text{col},i}(T)}{S_{\text{col}}(T)} =
    \frac{\frac{w_{i}Z_{i}}{A_{i}}}{\sum_{i}\frac{w_{i}Z_{i}}{A_{i}}} =
    \frac{\gamma_i Z_{i}}{\sum_{i}\gamma_i Z_{i}}.

where :math:`\gamma_i` is the atomic fraction of the :math:`i`-th element.
Therefore, the energy deposited by charged particles should be attributed to
a given element according to its fractional charge density.

.. _methods_decay_sources:

------------------------------
Steady-State Decay Sources
------------------------------

Radionuclides produced by fission and by neutron activation emit photons,
electrons, and positrons when they decay. In a reactor operating at steady
state, short-lived radionuclides decay as fast as they are produced, so their
emission accompanies the reactions that produce them. When decay sources are
enabled (see :ref:`usersguide_steady_state_decay`), OpenMC models this by
emitting the decay particles at the site of each neutron collision, analogous
to the treatment of prompt secondary photons.

Inventory per Reaction
----------------------

Consider a reaction channel :math:`c`, either fission of a nuclide at one of
the incident energies tabulated in the fission product yields or an activation
reaction listed in the depletion chain, that produces nuclide :math:`i` with
yield (or branching ratio) :math:`y_{c,i}`. The reactor operates with a
piecewise-constant power history: step :math:`s` lasts from :math:`t_s` to
:math:`t_{s+1}` at a power :math:`P_s` relative to the power :math:`P_S` at the
end of the history, which is the state being simulated. Assuming that the
reaction rate is proportional to the power and that radionuclides are removed
only by decay, the number of atoms of nuclide :math:`j` present at the end of
the history per unit reaction rate is

.. math::

    a_{c,j} = \sum_i y_{c,i} \, G_{ij},

where :math:`G_{ij}` is the inventory of nuclide :math:`j` resulting from unit
production of nuclide :math:`i` at the final power. :math:`G_{ij}` is computed
with transmutation trajectory analysis (TTA): the decay network starting from
nuclide :math:`i` is unfolded into linear chains :math:`i = n_1 \to n_2 \to
\cdots \to n_m = j`, each of which has the Bateman solution. Integrated over
the power history, the contribution of one chain is

.. math::

    G_{ij} \mathrel{+}= \left( \prod_{k=1}^{m-1} b_k \lambda_k \right)
    \sum_s \frac{P_s}{P_S} \sum_{k=1}^m \frac{1}{\prod_{l \ne k}
    (\lambda_l - \lambda_k)} \int_{T - t_{s+1}}^{T - t_s} e^{-\lambda_k \tau}
    d\tau,

where :math:`\lambda_k` is the decay constant of the :math:`k`-th nuclide on the
chain, :math:`b_k` is the branching ratio from the :math:`k`-th to the
:math:`(k+1)`-th nuclide, and :math:`T = t_{S+1}` is the end of the history. A
chain is not followed beyond a nuclide through which less than a fraction
:math:`10^{-12}` of the produced atoms have passed by the end of the history.
The number of atoms that have passed through the last nuclide of a chain is
obtained from the same expression by appending a fictitious stable daughter.
Since the decay network contains no cycles, every nuclide appears at most once
on a chain. Different nuclides with nearly equal decay constants, for which the
Bateman solution is singular, are handled by shifting one of the decay
constants by a relative amount of :math:`10^{-8}`. The inventories
:math:`G_{ij}` are computed once for each radionuclide that is produced and are
shared among all reaction channels.

Emission at Collisions
----------------------

The decay source of nuclide :math:`j` for particle type :math:`p` is
:math:`S_j^p(E)`, normalized so that its integral is the emission rate per atom
in the depletion chain. Each reaction event of channel :math:`c` is therefore
accompanied by

.. math::

    Y_c^p = \sum_j a_{c,j} \int S_j^p(E) \, dE

particles of type :math:`p`. At a neutron collision with nuclide :math:`i`, the
expected number of particles of type :math:`p` emitted is

.. math::

    \frac{1}{\sigma_{t}} \left[ \sum_{c \in \text{fission}} f_c \sigma_f Y_c^p
    + \sum_{c \in \text{activation}} k_{\text{eff}} \sigma_c Y_c^p \right],

where the factors :math:`f_c` linearly interpolate the fission channels in the
incident energy (outside of the range of tabulated yields, the closest channel
is used). As for prompt photons from non-fission reactions, the weight of
particles from activation is multiplied by :math:`k_{\text{eff}}` in
eigenvalue calculations to properly balance energy release and deposition.
For each emitted particle, a channel is sampled in proportion to its
contribution, then an emitting nuclide :math:`j` with probability proportional
to :math:`a_{c,j} \int S_j^p(E) dE`, and finally an energy from
:math:`S_j^p(E)`. The direction is sampled isotropically and the emitting
nuclide is recorded so that a :class:`~openmc.ParentNuclideFilter` can be used
to tally the contribution of each radionuclide. Electrons and positrons are
processed at their birth site like those from photon interactions.

Consistency with Fission Energy Release
---------------------------------------

For fissionable nuclides with fission product yields in the depletion chain,
delayed fission photons are produced explicitly from the decay of fission
products, so the scaling of the prompt fission photon yield by
:math:`(E_{\gamma,p} + E_{\gamma,d})/E_{\gamma,p}` is not applied to them. Other
fissionable nuclides keep the scaling if it is enabled. When fission product
electrons are emitted, the delayed beta energy :math:`E_{\beta}` is
deposited by those electrons, and it is removed from the fission heating
:math:`k_{i,f}` described above.

----------
References
----------

.. [Mack97] Abdou, M.A., Maynard, C.W., and Wright, R.Q. MACK: computer
   program to calculate neutron energy release parameters (fluence-to-kerma
   factors) and multigroup neutron reaction cross sections from nuclear data
   in ENDF Format. Oak Ridge National Laboratory report ORNL-TM-3994.
