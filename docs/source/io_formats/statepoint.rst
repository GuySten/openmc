.. _io_statepoint:

=======================
State Point File Format
=======================

The current version of the statepoint file format is 18.2.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the
               statepoint file format.
             - **openmc_version** (*int[3]*) -- Major, minor, and release
               version number for OpenMC.
             - **git_sha1** (*char[40]*) -- Git commit SHA-1 hash.
             - **date_and_time** (*char[]*) -- Date and time the summary was
               written.
             - **path** (*char[]*) -- Path to directory containing input files.
             - **tallies_present** (*int*) -- Flag indicating whether tallies
               are present (1) or not (0).
             - **source_present** (*int*) -- Flag indicating whether the source
               bank is present (1) or not (0).

:Datasets: - **seed** (*int8_t*) -- Pseudo-random number generator seed.
           - **stride** (*uint64_t*) -- Pseudo-random number generator stride.
           - **energy_mode** (*char[]*) -- Energy mode of the run, either
             'continuous-energy' or 'multi-group'.
           - **run_mode** (*char[]*) -- Run mode used, either 'eigenvalue' or
             'fixed source'.
           - **n_particles** (*int8_t*) -- Number of particles used per generation.
           - **n_batches** (*int*) -- Number of batches to simulate.
           - **current_batch** (*int*) -- The number of batches already simulated.
           - **n_inactive** (*int*) -- Number of inactive batches. Only present
             when `run_mode` is 'eigenvalue'.
           - **generations_per_batch** (*int*) -- Number of generations per
             batch. Only present when `run_mode` is 'eigenvalue'.
           - **k_generation** (*double[]*) -- k-effective for each generation
             simulated.
           - **entropy** (*double[]*) -- Shannon entropy for each generation
             simulated.
           - **k_col_abs** (*double*) -- Sum of product of collision/absorption
             estimates of k-effective.
           - **k_col_tra** (*double*) -- Sum of product of
             collision/track-length estimates of k-effective.
           - **k_abs_tra** (*double*) -- Sum of product of
             absorption/track-length estimates of k-effective.
           - **k_combined** (*double[2]*) -- Mean and standard deviation of a
             combined estimate of k-effective.
           - **n_realizations** (*int*) -- Number of realizations for global
             tallies.
           - **global_tallies** (*double[][2]*) -- Accumulated sum and
             sum-of-squares for each global tally.
           - **source_bank** (Compound type) -- Source bank information for each
             particle. The compound type has fields ``r``, ``u``, ``E``,
             ``time``, ``wgt``, ``delayed_group``, ``surf_id``, and
             ``particle``, which represent the position, direction, energy,
             time, weight, delayed group, surface ID, and particle type
             (PDG number), respectively. Only present when `run_mode` is
             'eigenvalue'.

**/tallies/**

:Attributes: - **n_tallies** (*int*) -- Number of user-defined tallies.
             - **ids** (*int[]*) -- User-defined unique ID of each tally.

**/tallies/meshes/**

:Attributes: - **n_meshes** (*int*) -- Number of meshes in the problem.
             - **ids** (*int[]*) -- User-defined unique ID of each mesh.

.. _mesh-spec-hdf5:

**/tallies/meshes/mesh <uid>/**

:Attributes: - **id** (*int*) -- ID of the mesh

:Datasets: - **name** (*char[]*) -- Name of the mesh.
           - **type** (*char[]*) -- Type of mesh.
           - **dimension** (*int*) -- Number of mesh cells in each dimension.
           - **Regular Mesh Only:**
              - **lower_left** (*double[]*) -- Coordinates of lower-left corner of
                mesh.
              - **upper_right** (*double[]*) -- Coordinates of upper-right corner
                of mesh.
              - **width** (*double[]*) -- Width of each mesh cell in each
                dimension.
           - **Rectilinear Mesh Only:**
              - **x_grid** (*double[]*) -- Mesh divisions along the x-axis.
              - **y_grid** (*double[]*) -- Mesh divisions along the y-axis.
              - **z_grid** (*double[]*) -- Mesh divisions along the z-axis.
           - **Cylindrical & Spherical Mesh Only:**
              - **r_grid** (*double[]*) -- The mesh divisions along the r-axis.
              - **phi_grid** (*double[]*) -- The mesh divisions along the phi-axis.
              - **origin** (*double[]*) -- The origin in cartesian coordinates.
           - **Spherical Mesh Only:**
              - **theta_grid** (*double[]*) -- The mesh divisions along the theta-axis.
           - **Unstructured Mesh Only:**
              - **filename** (*char[]*) -- Name of the mesh file.
              - **library** (*char[]*) -- Mesh library used to represent the
                                          mesh ("moab" or "libmesh").
              - **length_multiplier** (*double*) Scaling factor applied to the mesh.
              - **options** (*char[]*) -- Special options that control spatial
                                          search data structures used.
              - **volumes** (*double[]*) -- Volume of each mesh cell.
              - **vertices** (*double[]*) -- x, y, z values of the mesh vertices.
              - **connectivity** (*int[]*) -- Connectivity array for the mesh
                cells.
              - **element_types** (*int[]*) -- Mesh element types.

**/tallies/filters/**

:Attributes: - **n_filters** (*int*) -- Number of filters in the problem.
             - **ids** (*int[]*) -- User-defined unique ID of each filter.

**/tallies/filters/filter <uid>/**

:Datasets: - **type** (*char[]*) -- Type of the j-th filter. Can be 'universe',
             'material', 'cell', 'cellborn', 'surface', 'mesh', 'energy',
             'energyout', 'distribcell', 'mu', 'polar', 'azimuthal',
             'delayedgroup', or 'energyfunction'.
           - **n_bins** (*int*) -- Number of bins for the j-th filter. Not
             present for 'energyfunction' filters.
           - **bins** (*int[]* or *double[]*) -- Value for each filter bin of
             this type. Not present for 'energyfunction' filters.
           - **energy** (*double[]*) -- Energy grid points for energyfunction
             interpolation. Only used for 'energyfunction' filters.
           - **y** (*double[]*) -- Interpolant values for energyfunction
             interpolation. Only used for 'energyfunction' filters.

             :Attributes:
                          - **interpolation** (*int*) -- Interpolation type. Only used for
                            'energyfunction' filters.

**/tallies/derivatives/derivative <id>/**

:Datasets: - **independent variable** (*char[]*) -- Independent variable of
             tally derivative.
           - **material** (*int*) -- ID of the perturbed material.
           - **nuclide** (*char[]*) -- Alias of the perturbed nuclide.
           - **estimator** (*char[]*) -- Type of tally estimator, either
             'analog', 'tracklength', or 'collision'.

**/tallies/tally <uid>/**

:Attributes:
             - **internal** (*int*) -- Flag indicating the presence of tally
               data (0) or absence of tally data (1). All user defined
               tallies will have a value of 0 unless otherwise instructed.
             - **multiply_density** (*int*) -- Flag indicating whether reaction
               rates should be multiplied by atom density (1) or not (0).
             - **higher_moments** (*int*) -- Flag indicating whether
               higher-order tally moments are enabled (1) or not (0).

:Datasets: - **n_realizations** (*int*) -- Number of realizations.
           - **n_filters** (*int*) -- Number of filters used.
           - **filters** (*int[]*) -- User-defined unique IDs of the filters on
             the tally
           - **nuclides** (*char[][]*) -- Array of nuclides to tally. Note that
             if no nuclide is specified in the user input, a single 'total'
             nuclide appears here.
           - **derivative** (*int*) -- ID of the derivative applied to the
             tally.
           - **n_score_bins** (*int*) -- Number of scoring bins for a single
             nuclide.
           - **score_bins** (*char[][]*) -- Values of specified scores.
           - **results** (*double[][][2]*) -- Accumulated sum and sum-of-squares
             for each bin of the i-th tally. The first dimension represents
             combinations of filter bins, the second dimensions represents
             scoring bins, and the third dimension has two entries for the sum
             and the sum-of-squares.

**/adjoint_populations/**

Present only if adjoint side populations were requested.

:Datasets: - **n_generation** (*int*) -- Depth to which each root was grown.
           - **n_batches** (*int*) -- Number of active batches recorded.
           - **n_class** (*int*) -- Number of scored classes: fission (0),
             delayed (1) and photoneutron (2) roots, and the photoneutron
             branches of fission-root (3) and delayed-root (4) trees.
           - **n_tag** (*int*) -- Number of tags per population: 0 for prompt,
             otherwise the delayed group.
           - **photoneutrons** (*int*) -- Whether photoneutrons were taken as a
             population.
           - **perturbed_importance** (*int*) -- Whether photoneutron branches
             were grown.
           - **weight** (*double[]*) -- Summed weight per batch, population,
             tag and depth (0 to n_generation), in that order.
           - **weight_t0** (*double[]*) -- As **weight**, times the lifetime
             the root had reached at the fission that started each line.
           - **site_weight** (*double[]*) -- Target weight of each class in the
             last generation.
           - **n_roots_last_generation** (*int8_t[]*) -- Roots (branches for
             classes 3 and 4) grown per class in the last generation.
           - **raw_weight_last_generation** (*double[]*) -- Weight of each
             class in the last generation, before roulette.
           - **n_histories** (*int8_t*) -- Number of side-population histories
             tracked.
           - **root_fraction** (*double*) -- Roots per generation of each
             population as a fraction of the particles.
           - **photoneutron_energy_bins** (*double[]*) -- Energy-group edges
             [eV] of the grouped photoneutron record. Present only if
             requested.
           - **photoneutron_energy_variable** (*char[]*) -- ``photon_birth``
             or ``photoneutron``. Present only with the bins.
           - **photoneutron_energy_weight** (*double[]*) -- Summed
             photoneutron-root weight per batch, energy group and depth, in
             that order, summed over fissioning nuclides. Present only with
             the bins.
           - **photoneutron_fission_nuclides** (*char[]*) -- Space-separated
             names of the fissioning-nuclide bins, the last being ``other``.
             Present only if requested.
           - **photoneutron_nuclide_weight** (*double[]*) -- Summed
             photoneutron-root weight per batch, fissioning-nuclide bin,
             energy group (one if no edges were given) and depth, in that
             order. Present only with the nuclides.
           - **probe_energies** (*double[]*) -- Probe photon line energies
             [eV]. Present only with probes, as are the datasets below.
           - **probe_fission_nuclides** (*char[]*) -- Space-separated names
             of the probes' fissioning-nuclide bins: the listed nuclides and
             ``other``, or ``all``.
           - **probe_n_labels** (*int*) -- Number of probe labels: 0
             uncollided (empty with rays on), 1 after coherent scattering
             only, 2 after an energy-changing collision.
           - **probe_weight** (*double[]*) -- Summed probe-root weight per
             batch, fissioning-nuclide bin, line, label and depth, in that
             order.
           - **probe_fission_weight** (*double[]*) -- Probed fission weight,
             the sum of w/k sigma_f/sigma_t over the recorded fission events,
             per batch and fissioning-nuclide bin: the probe photon weight
             emitted per line.
           - **probe_site_weight** (*double[2]*) -- Target weights of the
             probe photons and of the probe roots in the last generation.
           - **n_probe_photons_last_generation**,
             **n_probe_roots_last_generation** (*int8_t*) -- Probe photons
             and probe roots in the last generation, after roulette.
           - **n_probe_histories** (*int8_t*) -- Number of probe-photon
             histories tracked.
           - **probe_root_fraction** (*double*) -- Probe roots per generation
             as a fraction of the particles.
           - **probe_trigger** (*double[2]*) -- Threshold and floor of the
             probe-importance trigger. Present only if one was set.
           - **ray_neutron_energies** (*double[]*) -- The ray probes'
             photoneutron comb [eV]. Present only with rays, as are the
             datasets below.
           - **ray_weight** (*double[]*) -- Summed uncollided photoneutron
             importance per batch, fissioning-nuclide bin, line and depth.
           - **ray_comb_weight** (*double[]*) -- The ray trees' summed
             weight per batch, fissioning-nuclide bin, comb energy and depth.
           - **ray_fission_weight** (*double[]*) -- Fission weight of the
             events the rays were cast from, per batch and nuclide bin.
           - **ray_site_weight** (*double[2]*),
             **n_rays_last_generation**, **n_ray_roots_last_generation**,
             **n_ray_segments** -- Target weights, counts and the number of
             ray segments walked.
           - **ray_root_fraction** (*double*) -- Ray roots per generation as
             a fraction of the particles.

**/runtime/**

All values are given in seconds and are measured on the master process.

:Datasets: - **total initialization** (*double*) -- Time spent reading inputs,
             allocating arrays, etc.
           - **reading cross sections** (*double*) -- Time spent loading cross
             section libraries (this is a subset of initialization).
           - **simulation** (*double*) -- Time spent between initialization and
             finalization.
           - **transport** (*double*) -- Time spent transporting particles.
           - **inactive batches** (*double*) -- Time spent in the inactive
             batches (including non-transport activities like communicating
             sites).
           - **active batches** (*double*) -- Time spent in the active batches
             (including non-transport activities like communicating sites).
           - **synchronizing fission bank** (*double*) -- Time spent sampling
             source particles from fission sites and communicating them to other
             processes for load balancing.
           - **sampling source sites** (*double*) -- Time spent sampling source
             particles from fission sites.
           - **SEND-RECV source sites** (*double*) -- Time spent communicating
             source sites between processes for load balancing.
           - **accumulating tallies** (*double*) -- Time spent communicating
             tally results and evaluating their statistics.
           - **writing statepoints** (*double*) -- Time spent writing statepoint
             files
