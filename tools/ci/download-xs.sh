#!/bin/bash
# Touching this script invalidates the CI test-data cache.
set -ex

# Download HDF5 data
if [[ ! -e $HOME/nndc_hdf5/cross_sections.xml ]]; then
    wget -q -O - https://anl.box.com/shared/static/teaup95cqv8s9nn56hfn7ku8mmelr95p.xz | tar -C $HOME -xJ
fi

# Download ENDF/B-VII.1 distribution
ENDF=$HOME/endf-b-vii.1
if [[ ! -d $ENDF/neutrons || ! -d $ENDF/photoat || ! -d $ENDF/atomic_relax ]]; then
    wget -q -O - https://anl.box.com/shared/static/4kd2gxnf4gtk4w1c8eua5fsua22kvgjb.xz | tar -C $HOME -xJ
fi

# Optional photonuclear test data.
#
# The tarballs above are the ones the project controls, and neither carries
# photonuclear cross sections or the ENDF gamma sublibrary. Until that data is
# part of the official test library, the photonuclear tests skip themselves
# when it is absent, and this overlay is opt-in: set OPENMC_PHOTONUCLEAR_URL to
# a tarball that unpacks a `photonuclear/` directory into $HOME/nndc_hdf5 and,
# optionally, a `gammas/` directory into $ENDF.
if [[ -n "$OPENMC_PHOTONUCLEAR_URL" ]]; then
    if [[ ! -d $HOME/nndc_hdf5/photonuclear ]]; then
        wget -q -O - "$OPENMC_PHOTONUCLEAR_URL" | tar -C $HOME -xJ
    fi
fi
