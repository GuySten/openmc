#!/usr/bin/env bash
# Build OpenMC variants with different sizeof(SourceSite).
# Usage: profiling/build_variants.sh [BUILD_ROOT] [variant ...]
# Variants: base min pad64 pad128 (serial) and base-mpi min-mpi pad64-mpi pad128-mpi
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${1:-$HOME/builds}; shift || true
VARIANTS=${*:-"base min pad64 pad128 base-mpi min-mpi pad64-mpi pad128-mpi"}
mkdir -p "$ROOT"
for v in $VARIANTS; do
  flags="-fno-omit-frame-pointer"
  mpi=OFF
  case $v in *-mpi) mpi=ON ;; esac
  kind=${v%-mpi}
  case $kind in
    base) ;;
    min) flags+=" -DOPENMC_PROFILE_SOURCESITE_MINIMAL" ;;
    pad*) flags+=" -DOPENMC_PROFILE_SOURCESITE_PAD=${kind#pad}" ;;
  esac
  echo "== $v: MPI=$mpi CXXFLAGS='$flags'"
  cmake -S "$SRC" -B "$ROOT/$v" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="$flags" -DOPENMC_USE_MPI=$mpi \
    -DOPENMC_BUILD_TESTS=OFF > "$ROOT/$v.cmake.log" 2>&1
  make -C "$ROOT/$v" -j"$(nproc)" > "$ROOT/$v.build.log" 2>&1
done
