.. _io_source:

==================
Source File Format
==================

Normally, source data is stored in a state point file. However, it is possible
to request that the source be written separately, in which case the format used
is that documented here.

When surface source writing is triggered, a source file named
``surface_source.h5`` is written with only the sources on specified surfaces,
following the same format, with the addition of the group and batch datasets
described below.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the source
               file format.

:Datasets:

           - **source_bank** (Compound type) -- Source bank information for each
             particle. The compound type has fields ``r``, ``u``, ``E``,
             ``time``, ``wgt``, ``delayed_group``, ``surf_id`` and ``particle``,
             which represent the position, direction, energy, time, weight,
             delayed group, surface ID, and particle type (PDG number),
             respectively.

           The following datasets and attribute are present only in surface
           source files. They impose two levels of structure on the source
           bank; see :ref:`surface_source_structure`.

           A *group* is the set of sites banked by one source history. Its
           sites are contiguous and are correlated with one another. A
           calculation reading the file must emit a group as a single history,
           not as one history per site, or scores defined per history such as
           pulse-height tallies are split into several smaller scores.

           A *batch* is the set of groups banked during one batch. No source
           particle contributes to more than one batch, so batches are
           statistically independent of one another. A batch also records how
           many source particles produced it, including those that put nothing
           across the recording surface.

           Sites are stored rank-major, so the groups of one batch are
           contiguous within a rank but not across ranks. The batch datasets
           therefore hold one entry per rank and batch -- a *segment* --
           ordered by rank and then by batch, alongside an ``n_ranks``
           attribute. A reader merges same-numbered segments across ranks, so
           that the batch count, the source particle count of each batch, and
           the unit of statistical independence do not depend on how many ranks
           wrote the file. This is what :attr:`openmc.ParticleList.batches`
           returns.

           - **group_offsets** (*int64[]*) -- Index into ``source_bank`` at
             which each group begins, with a trailing entry equal to the total
             number of sites, so that group ``g`` spans
             ``[group_offsets[g], group_offsets[g + 1])``.
           - **batch_offsets** (*int64[]*) -- Index into ``group_offsets`` at
             which each segment begins, with a trailing entry equal to the
             total number of groups, so that segment ``s`` owns groups
             ``[batch_offsets[s], batch_offsets[s + 1])``. Segment
             ``s = rank * n_batches + batch``, where ``n_batches`` is
             ``(len(batch_offsets) - 1) // n_ranks``.
           - **batch_n_particles** (*int64[]*) -- Number of source particles
             simulated for each segment. Summing over the segments of one batch
             gives the source particles simulated for that batch.
           - **batch_complete** (*int[]*) -- Whether each segment is a complete
             sample of itself. Zero indicates that sites were discarded because
             the surface source bank filled up partway through, in which case
             the batch that segment belongs to is biased and should be
             discarded when the file is used as a source.
           - **n_ranks** (*int*) -- Attribute giving the number of MPI ranks
             the segments are spread over, needed to merge them into batches.
             One for a file written by a serial calculation.
           - **n_source_particles** (*int64*) -- Attribute giving the total
             number of source particles represented by the file, equal to the
             sum of ``batch_n_particles``.

MCPL surface source files
-------------------------

When a surface source is written in MCPL format the same structure is stored
using facilities the MCPL format already provides, following the convention
established by MCNP's surface source files, where each track records the
history that produced it and the header records the number of histories run
separately from the number of tracks stored.

- The **user-flags** field of each particle holds its group index. All sites
  sharing a value were produced by one source history. Header comments document
  this, as the MCPL format requires for any use of the user-flags field.
- The header blob **openmc_batch_structure** holds the batch boundaries in
  units of groups, the source particle count of each batch, and the completeness
  flags, as text. Its first line is ``openmc_batch_structure v1``, followed by
  ``n_ranks`` and ``n_batches`` and then the three arrays. As in the HDF5
  format, the arrays hold one entry per rank and batch, ordered by rank and
  then by batch.
- The header **stat:sum** entry ``openmc_np1`` holds the number of source
  particles represented by the file. Note that for a surface source this is the
  count for the batches actually written, which differs from the whole-run
  total when the bank fills early or ``max_source_files`` splits the run.

.. warning::
    The user-flags field is also used by the MCNP converters distributed with
    MCPL to carry SSW surface IDs: ``ssw2mcpl --surf`` writes them there, and
    ``mcpl2ssw`` assumes the field holds a surface ID unless it is given an
    explicit ``-s<ID>``. OpenMC writes group indices instead, and these are not
    surface IDs -- they start at zero, whereas valid SSW surface IDs are
    1--999999. Always pass ``-s<ID>`` to ``mcpl2ssw`` when converting an OpenMC
    surface source file, or the group indices will be silently taken as surface
    IDs.
