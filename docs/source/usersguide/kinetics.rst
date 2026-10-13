.. _kinetics:

===================
Kinetics parameters
===================

OpenMC has the capability to estimate the following adjoint-weighted effective
generation time :math:`\Lambda_{\text{eff}}` and the effective delayed neutron
fraction :math:`\beta_{\text{eff}}`. These parameters are calculated using the
iterated fission probability (IFP) method [Hurwitz_1964]_ based on a similar
approach as in `Serpent 2 <https://doi.org/10.1016/j.anucene.2013.10.032>`_. The
implementation in OpenMC is limited to eigenvalue calculations and is described
in more details in [Dorville_2025]_.

----------------------------------
Iterated Fission Probability (IFP)
----------------------------------

With IFP, additional information needs to be recorded during the simulation
compared to a typical eigenvalue calculation. OpenMC stores an additional
set of values for every fission neutron simulated: the lifetimes of its
:math:`N_{\text{gen}}` direct ancestors for :math:`\Lambda_{\text{eff}}`, and
for :math:`\beta_{\text{eff}}` the fission events that emitted its ancestors
and itself (the fissioning nuclide and reaction, and the incident and outgoing
energies).

:math:`N_{\text{gen}}` is referred to as the number of generations in the
IFP method and corresponds to the number of generations between the birth of
a fission neutron and the time its score is added to the IFP tally. By default,
OpenMC considers 10 generations but this value can be modified by the user via
the ``ifp_n_generation`` settings in the Python API::

    settings.ifp_n_generation = 5

``ifp_n_generation`` should be greater than 0, but should also be lower than
or equal to the number of inactive batches declared for the calculation.
The respect of these constraints is verified by OpenMC before any calculation.

OpenMC will automatically detect the type of data that needs to be stored based
on the tally scores selected by the user. This guarantees that only information
of interest are stored during a simulation and avoids using extra memory when
only one parameter is needed. The following table shows the tally scores that
are needed to compute kinetics parameters in OpenMC:

.. table:: **OpenMC tally scores needed to calculate adjoint-weighted kinetics parameters**
    :align: center

    =============================== ============================ ========================== ========
    OpenMC tally score \\ Parameter :math:`\Lambda_{\text{eff}}` :math:`\beta_{\text{eff}}` Both
    =============================== ============================ ========================== ========
    ``ifp-time-numerator``          X                                                       X
    ``ifp-beta-numerator``                                       X                          X
    ``ifp-denominator``             X                            X                          X
    =============================== ============================ ========================== ========

|

.. note:: Because the memory footprint of additional data is generally non-negligible
    with IFP, it is recommended to choose the value for ``ifp_n_generation`` carefully.
    For every fission neutron simulated, :math:`\Lambda_{\text{eff}}` stores
    :math:`N_{\text{gen}}` floating point values (8 bytes each) and
    :math:`\beta_{\text{eff}}` stores :math:`N_{\text{gen}} + 1` fission records
    (12 bytes each).

The ``ifp-beta-numerator`` score does not score whether an ancestor was emitted
as a delayed neutron, which happens for a fraction :math:`\beta` of fission
neutrons, but the probability that it was emitted in each delayed group given
the fission event that emitted it and its outgoing energy :math:`E'`:

.. math::
    :label: ifp_delayed_probability

    w_g(E') = \frac{\beta_g \chi_{d,g}(E')}{(1 - \beta) \chi_p(E') +
    \sum_{g'} \beta_{g'} \chi_{d,g'}(E')},

where :math:`\chi_p` and :math:`\chi_{d,g}` are the prompt and delayed group
emission spectra of the fissioning nuclide at the incident energy, and
:math:`\beta_g` its delayed neutron fraction of group :math:`g`. The progeny
of a neutron depend on how it was emitted only through its outgoing energy, so
this has the same expectation as scoring the delayed group that was sampled,
with a much lower variance. It requires outgoing energy distributions whose
density can be evaluated (tabular, Maxwell, evaporation, and Watt spectra).

IFP scores can be tallied by nuclide, in which case they are binned by the
nuclide whose fission emitted the ancestor rather than by the nuclide in which
the scoring fission occurs. The contribution of nuclide :math:`i` to the
effective delayed neutron fraction is then the ``ifp-beta-numerator`` score
of nuclide :math:`i` divided by the total ``ifp-denominator`` score. When
``ifp_n_generation`` equals the number of inactive batches, ancestors in the
first active batch can be source neutrons rather than fission neutrons, which
only contribute to the total.

-----------------------------
Obtaining kinetics parameters
-----------------------------

The ``Model`` class can be used to automatically generate all IFP tallies using
the Python API with :attr:`openmc.Settings.ifp_n_generation` greater than 0 and
the :meth:`openmc.Model.add_ifp_kinetics_tallies` method::

    model = openmc.Model(geometry, settings=settings)
    model.add_kinetics_parameters_tallies(num_groups=6)  # Add 6 precursor groups

Alternatively, each of the tallies can be manually defined using group-wise or
total :math:`\beta_{\text{eff}}` specified by providing a 6-group
:class:`openmc.DelayedGroupFilter`::

    beta_tally = openmc.Tally(name="group-beta-score")
    beta_tally.scores = ["ifp-beta-numerator"]

    # Add DelayedGroupFilter to enable group-wise tallies
    beta_tally.filters = [openmc.DelayedGroupFilter(list(range(1, 7)))]

Here is an example showing how to declare the three available IFP scores in a
single tally::

    tally = openmc.Tally(name="ifp-scores")
    tally.scores = [
        "ifp-time-numerator",
        "ifp-beta-numerator",
        "ifp-denominator"
    ]

The effective generation time :math:`\Lambda_{\text{eff}}` is calculated
by dividing the result of the ``ifp-time-numerator`` score by the one obtained
for ``ifp-denominator`` and by the :math:`k_{\text{eff}}` of the simulation:

.. math::
    :label: lambda_eff

    \Lambda_{\text{eff}} = \frac{S_{\text{ifp-time-numerator}}}{S_{\text{ifp-denominator}} \times k_{\text{eff}}}

The effective delayed neutron fraction :math:`\beta_{\text{eff}}` is calculated
by dividing the result of the ``ifp-beta-numerator`` score by the one obtained
for ``ifp-denominator``:

.. math::
    :label: beta_eff

    \beta_{\text{eff}} = \frac{S_{\text{ifp-beta-numerator}}}{S_{\text{ifp-denominator}}}

The kinetics parameters can be retrieved directly from a statepoint file using
the :meth:`openmc.StatePoint.ifp_results` method::

    with openmc.StatePoint(output_path) as sp:
        generation_time, beta_eff = sp.get_kinetics_parameters()

.. only:: html

   .. rubric:: References

.. [Hurwitz_1964] H. Hurwitz Jr., "Naval Reactors Physics Handbook", volume 1, p. 864.
    Radkowsky, A. (Ed.), Naval Reactors, Division of Reactor Development, U.S.
    Atomic Energy Commission (1964).

.. [Dorville_2025] J. Dorville, L. Labrie-Cleary, and P. K. Romano, "Implementation
    of the Iterated Fission Probability Method in OpenMC to Compute Adjoint-Weighted
    Kinetics Parameters", International Conference on Mathematics and Computational
    Methods Applied to Nuclear Science and Engineering (M&C 2025), Denver, April 27-30,
    2025.
