.. _io_track:

=================
Track File Format
=================

The current revision of the particle track file format is 3.2.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the track
               file format.

:Datasets:
           - **track_<b>_<g>_<p>** (Compound type) -- Particle track information
             for source particle in batch *b*, generation *g*, and particle
             number *p*. particle. The compound type has fields ``r``, ``u``,
             ``E``, ``time``, ``wgt``, ``cell_id``, ``cell_instance``, and
             ``material_id``, which represent the position (each coordinate in
             [cm]), direction, energy in [eV], time in [s], weight, cell ID,
             cell instance, and material ID, respectively. When the particle is
             present in a cell with no material assigned, the material ID is
             given as -1. Note that this array contains information for one or
             more primary/secondary particles originating. The starting index
             for each primary/secondary particle is given by the ``offsets``
             attribute.

             :Attributes: - **n_particles** (*int*) -- Number of
                            primary/secondary particles for the source history.
                          - **offsets** (*int[]*) Offset (starting index) into
                            the array for each primary/secondary particle. The
                            last offset should match the total size of the
                            array.
                          - **particles** (*int32_t[]*) -- Particle type for
                            each primary/secondary particle (PDG number).
                          - **root_id** (*int64_t*) -- Particle number of the
                            source particle at the root of the history. Only
                            present when the dataset holds a secondary particle
                            that is stored separately from that source
                            particle, which happens when transporting with the
                            shared secondary bank.
