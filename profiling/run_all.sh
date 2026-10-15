#!/usr/bin/env bash
# Full benchmark matrix used for profiling/REPORT.md.
# Usage: profiling/run_all.sh BUILD_ROOT WORKDIR RESULTS_DIR [first_step]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${1:-$HOME/builds}; W=${2:-$HOME/runs}; R=${3:-$HERE/results}; FIRST=${4:-1}
mkdir -p "$R"
run() { python3 "$HERE/run_bench.py" --builds "$B" --workdir "$W" "$@"; }
V="base min pad64 pad128"
step() { [ "$1" -ge "$FIRST" ]; }
# serial (1 rank x all threads)
step 1 && run --variants $V --cases eigen-1e5 eigen-1e6 fixed-1e5 fixed-1e6 --repeat 3 --results "$R/serial.jsonl"
step 2 && run --variants $V --cases fixed-1e5-local fixed-1e5-surf --repeat 2 --results "$R/serial.jsonl"
# eigen-1e7 needs ~7*1e7*sizeof bytes: pad64/pad128 do not fit in 15 GB
step 3 && run --variants $V --cases eigen-5e6 --repeat 1 --results "$R/serial.jsonl"
step 4 && run --variants min base --cases eigen-1e7 --repeat 1 --results "$R/serial.jsonl"
# MPI (4 ranks x 1 thread, 2 ranks x 2 threads)
step 5 && run --variants $V --cases mpi4-eigen-4e5 mpi4-eigen-4e6 mpi4-fixed-4e5 mpi2-eigen-2e6 --repeat 2 --results "$R/mpi.jsonl"
# perf profiles of base and pad128
step 6 && run --variants base pad128 --cases eigen-1e6 fixed-1e6 eigen-5e6 --perf --results "$R/perf.jsonl"
step 7 && run --variants base pad128 --cases mpi4-eigen-4e6 mpi4-fixed-4e5 --perf --results "$R/perf.jsonl"
true
