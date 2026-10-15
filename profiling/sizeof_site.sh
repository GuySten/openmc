#!/usr/bin/env bash
# Print struct sizes for each variant's flags.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
INC="-I$SRC/include -I/usr/include/hdf5/serial -I${BUILD_INC:-$HOME/builds/base/include}"
for v in "base:" "min:-DOPENMC_PROFILE_SOURCESITE_MINIMAL" "pad64:-DOPENMC_PROFILE_SOURCESITE_PAD=64" "pad128:-DOPENMC_PROFILE_SOURCESITE_PAD=128"; do
  name=${v%%:*}; flags=${v#*:}
  g++ -std=c++17 -O0 $INC $flags "$SRC/profiling/sizeof_site.cpp" -o /tmp/sizeof_$name 2>&1 | head -5
  echo "$name: $(/tmp/sizeof_$name)"
done
