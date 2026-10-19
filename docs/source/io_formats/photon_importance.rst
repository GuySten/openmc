.. _io_photon_importance:

=============================
Photon Importance File Format
=============================

Written by :meth:`openmc.AdjointPopulations.export_photon_importance` from a
run with probe photons, for folding delayed-photon sources outside OpenMC. A
delayed-photon source :math:`S_j(E)` per fission of the nuclides of bin
:math:`j` gives the effective delayed photoneutron fraction

.. math::

    \beta_\mathrm{ph} = \sum_j \int S_j(E)\, I_j(E)\, dE ,

with :math:`I_j` linearly interpolated on its grid. Over any set of batches,
:math:`I_j(E)` is the mean of **batch_numerator** divided by the mean of
**fission_importance**, so a fold carries the run's batch statistics.

A probed reaction :math:`r` (an activation, say Al27(n,gamma)) has a table of
its own, the importance per reaction of a photon born at its sites. The
photons :math:`S_r(E)` of what it makes, per atom made, add
:math:`\int S_r(E)\, I_r(E)\, dE` to :math:`\beta_\mathrm{ph}`.

**/**

:Attributes: - **filetype** (*char[]*) -- 'photon_importance'.
             - **version** (*int[2]*) -- Major and minor version of the
               format, currently 1.1 (1.0 had no probed reactions).
             - **openmc_version** (*char[]*) -- Version of the OpenMC Python
               API that wrote the file.
             - **nuclides** (*char[][]*) -- Names of the fissioning-nuclide
               bins; 'other' holds every fissioning nuclide not named, and
               'all' every nuclide when the run had a single bin.
             - **depth** (*int*) -- Depth of the side populations at which
               the importance was read.
             - **n_batches** (*int*) -- Number of active batches.
             - **keff** (*double*) -- Eigenvalue of the run.
             - **threshold** (*double*) -- Photoneutron threshold of the
               target medium [eV].
             - **rtol** (*double*) -- Tolerance of linear interpolation of the
               tables.
             - **floor** (*double*) -- Fraction of the largest importance
               below which **rtol** is absolute.

:Datasets: - **fission_importance** (*double[]*) -- Fission-root importance
             of each batch at **depth**.
           - **delayed_importance** (*double[][]*) -- Delayed-root importance
             of each batch and delayed group, [batch, group]; with
             **fission_importance** it gives :math:`\beta_\mathrm{eff}` per
             group, paired with the photon importance batch by batch.
           - **fission_lifetime** (*double[]*) -- Lifetime-weighted
             fission-root sum of each batch; the generation time is its mean
             over :math:`k` times the mean of **fission_importance**.

**/nuclides/<name>/**

:Attributes: - **fission_fraction** (*double*) -- The bin's share of all
               fissions, from **fission_weight**.

:Datasets: - **energy** (*double[]*) -- Photon energies [eV] of the table, on
             which linear interpolation is within **rtol** of the rebuilt
             importance.
           - **importance** (*double[]*) -- :math:`I_j(E)`: the
             photoneutron reactivity, per unit fission-neutron importance, that
             a photon line of unit intensity per fission of the bin's nuclides
             adds.
           - **std_dev** (*double[]*) -- Batch standard deviation of
             **importance**.
           - **batch_numerator** (*double[][]*) -- Per-batch numerators of
             **importance**, [batch, energy].
           - **fission_weight** (*double[]*) -- Collision estimate of the bin's
             fissions in each batch.

**/reactions/<label>/**, one per probed reaction (e.g. ``Al27(n,gamma)``),
listed in the root attribute **reactions** (*char[][]*):

:Attributes: - **nuclide** (*char[]*), **mt** (*int*) -- The reaction.
             - **reactions_per_fission** (*double*) -- Its reactions per
               fission, from **reaction_weight** over the bins'
               **fission_weight**.
             - **branching_source** (*char[]*) -- 'chain' if **products**
               and **branching_ratio** were written from a depletion chain,
               else 'none' (no chain was given, or it lacks the reaction).

:Datasets: - **energy**, **importance**, **std_dev**, **batch_numerator** --
             As for a nuclide bin, per reaction at the reaction's sites.
           - **reaction_weight** (*double[]*) -- Collision estimate of the
             reaction's reactions in each batch.
           - **products** (*char[][]*), **branching_ratio** (*double[]*) --
             What one reaction makes, ground and isomeric states apart
             (e.g. ``Ag110`` and ``Ag110_m1``), with the chain's fixed
             branching ratios. Only when **branching_source** is 'chain';
             never for fission, whose products are the fission yields'.
