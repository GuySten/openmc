# How much do bank memory and time depend on `sizeof(SourceSite)`?

Branch `sourcesite-profiling`, based on develop `46c4c60` (equal to upstream
develop at the time). All numbers below were measured on one 4-core machine
with multigroup models; see [Limitations](#limitations).

## Summary

* **Memory is the real cost; run time barely moves.** In eigenvalue mode the
  resident bank memory is exactly **7 · N · sizeof(SourceSite)** per rank
  (fission bank 3N + source bank N + a 3N scratch array allocated every
  generation in `synchronize_bank`). Each byte of `SourceSite` costs
  **6.7 MB per 10⁶ particles per rank**. Fixed source with the shared
  secondary bank costs 1.9 MB per byte per 10⁶ source particles.
* **Run time.** Bank operations are 3–4 % of wall time at 10⁵–10⁶ particles
  per rank and grow ~linearly with site size (≈2–3 ns per byte per particle per
  generation, serial). In total wall time the size effect is smaller than the
  run-to-run noise of transport (±1–10 %). Shrinking 136 → 104 B buys at most
  ≈1–2 % at 10⁶ particles.
* **Exception: large banks.** At 5·10⁶–10⁷ particles per rank, where the banks
  approach node memory, `synchronize_bank` became super-linear: 12–15 % of the
  run at 136 B, and the 136 → 104 B shrink saved 15 % of wall time at 10⁷
  (single runs).
* **Much of the size dependence is waste, not necessary traffic.**
  `synchronize_bank` value-initializes a fresh `vector<SourceSite>(3N)` every
  generation: 3N·sizeof bytes written and, above glibc's 32 MB mmap threshold,
  page-faulted from the OS each generation (measured: 214 k vs 27 k page
  faults at 10⁵ particles, 136 vs 104 B). `SharedArray::reserve` likewise
  writes the whole 3N fission-bank capacity. Fixing these two (no change to
  `SourceSite`) would cut eigenvalue bank memory from 7N to about (k+2)N sites
  and remove most of the measured per-byte time.
* **But `SourceSite` will grow.** Open work on the GuySten branches plus open
  upstream PRs would take it from 136 B to ~240 B if merged as they are
  (+70 % bank memory). Most of the added fields fall into a few generic
  categories (lineage, clock, origin) and overlap each other.
* **Recommendation:** (1) fix the bank allocation waste first; (2) stop the
  growth with generic fields rather than per-feature fields (options 1/3);
  (3) side arrays only for large, rarely used data, as IFP already does;
  (4) a full structure-of-arrays bank (option 4) and per-bank site types
  (option 2) are not justified by these measurements.

## Setup

| item | value |
|---|---|
| source | develop `46c4c607518c` + the profiling-only build parameter below |
| compiler | GCC 13.3.0 (Ubuntu 24.04), `-O2 -g` (RelWithDebInfo) `-fno-omit-frame-pointer` |
| libraries | HDF5 1.10.10, OpenMPI 4.1.6, pugixml 1.14, fmt 9.1.0 (from apt: the pinned FetchContent tarballs could not be downloaded from github.com in this container) |
| machine | 4 vCPU Intel Xeon @ 2.80 GHz, 33 MB L3, 16 GB RAM, no swap |
| profiler | `perf` 6.8, `-e cpu-clock -F 999 --call-graph fp` (hardware counters not available) |
| threads | serial builds: 1 rank × 4 OpenMP threads; MPI: 4 ranks × 1 thread or 2 × 2, all on one node (shared-memory transport) |

### Build parameter (the only `src/` change)

`include/openmc/particle_data.h` and `initialize_mpi()` in `src/initialize.cpp`,
marked *PROFILING ONLY*, set through `CMAKE_CXX_FLAGS`:

* `-DOPENMC_PROFILE_SOURCESITE_PAD=N` appends `char profile_pad[N]`.
* `-DOPENMC_PROFILE_SOURCESITE_MINIMAL` removes `parent_nuclide`, `wgt_born`,
  `wgt_ww_born`, `n_split`, `n_collision` (replaced by stateless static members
  that discard writes and read back the default; valid only without weight
  windows, D1S and collision filters, which these benchmarks do not use).
  `parent_id`/`progeny_id` stay because the bank sort needs them, so the
  minimum reachable here is 104 B, not the 84 B written to source files.
* The MPI datatype is now built from the fields present (plus the padding as
  `MPI_BYTE`) and resized to `sizeof(SourceSite)`.

| variant | sizeof(SourceSite) |
|---|---|
| `min` | 104 B |
| `base` (develop) | 136 B |
| `pad64` | 200 B |
| `pad128` | 264 B |

`sizeof(ParticleData)` is 904 B on develop and does not change.

### Benchmarks

Multigroup, C5G7 7-group UO2/water data from `examples/pincell_multigroup`
(no cross-section download). Continuous-energy runs were not attempted.

| case | model | stresses |
|---|---|---|
| `eigen-N` | infinite (reflective) UO2 pin cell, k ≈ 1.31 | fission bank, `sort_bank`, `synchronize_bank` |
| `fixed-N` | 30×30 lattice of the same pins, vacuum radially, fixed source; fission neutrons become secondaries; shared secondary bank on; ≈3–6 secondaries per source, up to 0.7 M sites in one generation, 180–370 generations | shared secondary bank collection, sort, MPI redistribution |
| `fixed-1e5-local` | same with `shared_secondary_bank = false` | particle-local secondary banks |
| `fixed-1e5-surf` | same with surface source writing on the outer boundary (`max_particles = 2·10⁶`, 1.26 M sites written) | surface source bank |
| `mpi4-*`, `mpi2-*` | the same models on 4 or 2 ranks | `synchronize_bank` send/recv, `synchronize_global_secondary_bank` |

### Reproduce

```bash
profiling/build_variants.sh ~/builds                 # 8 builds: {base,min,pad64,pad128}{,-mpi}
BUILD_INC=~/builds/base/include profiling/sizeof_site.sh
profiling/run_all.sh ~/builds ~/runs profiling/results   # full matrix (~4 h here)
python3 profiling/analyze.py profiling/results/serial.jsonl profiling/results/mpi.jsonl
python3 profiling/perf_summary.py ~/runs/eigen-1e6/perf.base.r0.data
```

`run_bench.py` runs a single case/variant and parses OpenMC's timers,
`/usr/bin/time` (wall, peak RSS) and the secondary-generation log.
Raw results: `results/serial.jsonl`, `results/mpi.jsonl`, `results/perf.jsonl`,
`results/perf_summary.jsonl`; tables: `results/analysis.md`.

## Results

Mean ± half-range over repeats (n = 3 for the serial 10⁵/10⁶ cases, 2 for MPI
and the local/surf cases, 1 for 5·10⁶ and 10⁷). *Non-transport* = total −
"time in transport only"; in fixed-source mode all bank work happens inside
the transport timer, so there it is ≈0 and perf is the only breakdown.

### Eigenvalue, serial (1 rank × 4 threads)

| case | sizeof | elapsed [s] | non-transport [s] | sync bank [s] | peak RSS [MB] |
|---|---|---|---|---|---|
| eigen-1e5 (20 batches) | 104 | 16.63 ± 1.37 | 0.34 | 0.207 | 89 |
| | 136 | 15.88 ± 0.96 | 0.75 | 0.532 | 110 |
| | 200 | 17.31 ± 1.71 | 1.06 | 0.757 | 153 |
| | 264 | 17.49 ± 1.61 | 1.39 | 0.966 | 196 |
| eigen-1e6 (6 batches) | 104 | 48.92 ± 4.68 | 2.00 | 1.266 | 721 |
| | 136 | 53.04 ± 1.82 | 2.55 | 1.569 | 934 |
| | 200 | 49.19 ± 1.73 | 3.30 | 2.056 | 1362 |
| | 264 | 54.75 ± 4.91 | 4.21 | 2.680 | 1789 |
| eigen-5e6 (3 batches) | 104 | 131.06 | 11.78 | 8.85 | 3528 |
| | 136 | 127.66 | 20.07 | 15.65 | 4596 |
| | 200 | 143.34 | 27.23 | 20.80 | 6732 |
| | 264 | 151.45 | 31.51 | 25.15 | 8869 |
| eigen-1e7 (3 batches) | 104 | 258.07 | 24.91 | 18.85 | 7038 |
| | 136 | 305.17 | 52.86 | 45.20 | 9174 |

eigen-1e7 at 200/264 B needs 14–18 GB and was not run.

### Fixed source, serial

| case | sizeof | elapsed [s] | peak RSS [MB] |
|---|---|---|---|
| fixed-1e5 (shared bank) | 104 / 136 / 200 / 264 | 23.7 / 23.9 / 23.6 / 25.0 (±1.1–1.8) | 52 / 62 / 82 / 103 |
| fixed-1e6 (shared bank) | 104 / 136 / 200 / 264 | 86.7 / 88.7 / 91.5 / 92.8 (±0.6–11.6) | 303 / 389 / 531 / 603 |
| fixed-1e5-local (no shared bank) | 104 / 136 / 200 / 264 | 21.5 / 19.9 / 19.9 / 21.4 (±0.1–3.1) | 19 / 19 / 19 / 19 |
| fixed-1e5-surf | 104 / 136 / 200 / 264 | 21.0 / 22.3 / 24.3 / 22.9 (±0.5–2.8) | 250 / 321 / 464 / 606 |

### MPI (one node)

| case | sizeof | elapsed [s] | sync bank [s] | send/recv [s] | peak RSS per rank [MB] |
|---|---|---|---|---|---|
| mpi4-eigen-4e5 (10⁵/rank) | 104 / 136 / 200 / 264 | 43.7 / 42.7 / 44.2 / 44.5 | 1.91 / 2.29 / 3.26 / 3.28 | 0.035 / 0.050 / 0.073 / 0.095 | 101 / 122 / 165 / 208 |
| mpi4-eigen-4e6 (10⁶/rank) | 104 / 136 / 200 / 264 | 128.1 / 126.7 / 129.5 / 136.5 | 2.29 / 4.48 / 3.94 / 10.11 | 0.128 / 0.167 / 0.165 / 0.224 | 733 / 946 / 1374 / 1801 |
| mpi2-eigen-2e6 (10⁶/rank) | 104 / 136 / 200 / 264 | 75.0 / 78.0 / 82.3 / 84.3 | 1.55 / 2.33 / 2.38 / 2.88 | 0.115 / 0.148 / 0.157 / 0.208 | 732 / 946 / 1374 / 1801 |
| mpi4-fixed-4e5 (10⁵/rank) | 104 / 136 / 200 / 264 | 78.1 / 78.8 / 78.3 / 79.2 (±0.2–0.7) | – | – | 64 / 74 / 94 / 115 |

The MPI "sync bank" time includes waiting for the slowest rank, which is why it
is noisier than the serial numbers. Send/recv of sites between ranks (the part
that scales with message size) is small on one node: 0.1–0.2 s per run.

### Where the time goes (perf, share of all sampled CPU time)

Inclusive ("children") share; serial functions cost the same seconds of wall
time, so multiply by the thread count (4) for their share of wall time.

| profile | sort_bank | synchronize_bank | sync_global_secondary_bank | collect + thread-bank push_back | create_fission_sites | from_source + revive | add_surf_source_to_bank |
|---|---|---|---|---|---|---|---|
| eigen-1e6 base | 0.12 % | 0.97 % | – | – | 0.87 % | 0.06 % | 1.06 % |
| eigen-1e6 pad128 | 0.21 % | 1.52 % | – | – | 0.88 % | 0.09 % | 1.34 % |
| eigen-5e6 base | 0.11 % | 0.76 % | – | – | 0.88 % | 0.07 % | 1.58 % |
| eigen-5e6 pad128 | 0.25 % | 9.21 % | – | – | 0.89 % | 0.09 % | 1.09 % |
| fixed-1e6 base | – | – | – | 0.37 % | 0.69 % | 0.27 % | 0.18 % |
| fixed-1e6 pad128 | – | – | – | 0.87 % | 0.75 % | 0.35 % | 0.17 % |
| mpi4-eigen-4e6 base | 0.15 % | 2.75 % | – | – | 1.03 % | 0.24 % | 0.42 % |
| mpi4-eigen-4e6 pad128 | 0.29 % | 6.46 % | – | – | 1.05 % | 0.31 % | 0.42 % |
| mpi4-fixed-4e5 base | – | – | 6.20 % | 0.36 % | 0.80 % | 0.50 % | 0.19 % |
| mpi4-fixed-4e5 pad128 | – | – | 4.90 % | 0.89 % | 0.83 % | 0.48 % | 0.18 % |

`create_fission_sites` is dominated by sampling the fission physics, not by the
site copy, and is the same at 136 and 264 B. `from_source` /
`event_revive_from_secondary` (reading one site per particle) are below 0.5 %
and only weakly size dependent. `sync_global_secondary_bank` in the MPI
fixed-source case does not grow with size (6.2 % → 4.9 %): it is the
`MPI_Allgather` / `MPI_Alltoallv` called in each of ~470 secondary
generations, dominated by waiting for the slowest rank (OpenMPI busy-polls).

### Fraction of run time in bank operations (base, 136 B)

| case | bank operations, wall | made up of |
|---|---|---|
| eigen-1e5 | 3.4 % | `synchronize_bank` 0.53 s of 15.9 s, `sort_bank` < 0.1 s |
| eigen-1e6 | 3.4 % | `synchronize_bank` 1.57 s + `sort_bank` ≈ 0.2 s of 53 s |
| eigen-5e6 | 12.3 % | `synchronize_bank` 15.7 s of 128 s |
| eigen-1e7 | 15 % | `synchronize_bank` 45.2 s of 305 s |
| fixed-1e6 | ≈1 % | collection/sort ≈ 0.6 s serial + parallel push_back/revive |
| mpi4-eigen-4e6 | 3.5 % | `synchronize_bank` 4.5 s of 127 s (incl. waiting) |
| mpi4-fixed-4e5 | ≈6 % | mostly MPI waiting in the per-generation redistribution, not size dependent |

### Slope per byte of `SourceSite`

Least-squares slope over all runs of the four variants:

| case | wall time [ms/B] | non-transport [ms/B] | peak RSS [MB/B] | 136 → 104 B, wall |
|---|---|---|---|---|
| eigen-1e5 | 8 (noise) | 6.1 | 0.67 | −1.6 % (noise) |
| eigen-1e6 | 23 (noise) | 13.4 | 6.68 | −1.4 % (noise) |
| eigen-5e6 | 147 | 118 | 33.4 | −3.7 % |
| eigen-1e7 (2 points) | 1472 | 873 | 66.8 | −15 % |
| fixed-1e5 | 7 (noise) | – | 0.32 | −0.9 % (noise) |
| fixed-1e6 | 38 (noise) | – | 1.88 | −1.4 % (noise) |
| fixed-1e5-local | 1 (noise) | – | 0.00 | – |
| fixed-1e5-surf | 13 (noise) | 1.4 | 2.23 | −1.9 % (noise) |
| mpi4-eigen-4e6 | 54 | 47 | 6.68 per rank | −1.4 % |
| mpi4-fixed-4e5 | 5 (noise) | – | 0.32 per rank | −0.2 % |

"(noise)" marks slopes smaller than the run-to-run spread of the total wall
time; the non-transport and RSS slopes are clean. Normalised:

* **Memory:** eigenvalue 6.7 B of RSS per byte of site per particle (= 7N);
  shared secondary bank ≈2.7 B per byte per site of the largest generation;
  surface-source writing 1 B per byte per `max_particles` (the bank is
  allocated, and value-initialised, at full capacity).
* **Time, eigenvalue:** 2–3 ns per byte of site per particle per generation at
  10⁵–10⁶ particles per rank, serial; 8 ns at 5·10⁶ (memory pressure).

## Why: the cost is mostly avoidable allocation, not bank traffic

1. **`synchronize_bank` scratch array.** `vector<SourceSite> temp_sites(3 *
   work_per_rank)` is allocated, value-initialised (all default member
   initialisers run) and freed every generation, although at most
   `tooth_end - tooth_start` (= `n_particles` without MPI) sites are used — a
   number known before the allocation. Its "Sampling source sites" timer is
   90 % of the sync time.
2. **Page faults dominate.** glibc serves allocations above its dynamic mmap
   threshold (max 32 MB) with fresh pages each time. At 10⁵ particles the
   scratch array is 31 MB at 104 B (heap, reused) but 41 MB at 136 B (fresh
   mmap every generation): measured 27 k vs 214 k minor page faults per run
   (`/usr/bin/time -f %R`), and with the adaptive threshold disabled
   (`MALLOC_MMAP_THRESHOLD_` set) the 104 B build rose to 164 k faults and
   twice the sampling time. This, not the bytes copied, explains the large
   104 → 136 B step at 10⁵ particles. At 10⁷ particles the 4 GB per-generation
   allocation also competes for memory (RSS 9.2 of 16 GB), which explains the
   super-linear 45 s.
3. **Reserved capacity is resident.** `SharedArray::reserve` and the sized
   constructor use `make_unique<T[]>(n)`, which value-initialises every
   element, so the 3N fission-bank capacity is resident memory although only
   ≈kN is used. The same holds for the surface-source bank (allocated at
   `max_particles`) and the MPI receive buffer in
   `synchronize_global_secondary_bank`, and `extend_uninitialized` initialises
   its new block when it grows.

**Fix (not implemented here):** size the scratch exactly (or comb straight
into `source_bank` without MPI) and reuse it; allocate bank capacity without
initialising it (`SourceSite` is trivially copyable). Expected: eigenvalue bank
memory from 7N to ≈(k+2)N sites (−50 % here), most of the per-byte sync time
gone, no change to `SourceSite`. The fission bank keeps its 3N reservation, so
behaviour for large k is unchanged. A further step — fusing sort, comb and copy
into one parallel pass using `offset[parent_id] + progeny_id` — is possible but
`sort_bank` is only ≈0.1–0.3 % of CPU time.

**Side finding (unrelated to size):** `add_surf_source_to_bank` costs 1–1.6 %
of CPU in eigenvalue runs without any surface source: every surface has
`surf_source_` set when no surface list is given, so the function is called on
every crossing and returns early.

## How `SourceSite` is growing

### GuySten branches (all 120 scanned)

| branch | added fields | sizeof |
|---|---|---|
| `detector-response` | `shadow_depth`, `shadow_tag`, `shadow_t0`, `fission_nuclide`, `lifetime`, `r_born`, `cell_born` | 200 B |
| `claude/bold-heisenberg-m0fy16` | `lifetime`, `r_born`, `cell_born` | 176 B |
| `photonuclear-kinetics`, `probe-rays` | `shadow_depth`, `shadow_tag`, `shadow_t0`, `fission_nuclide` | 160 B |
| `super-history` | `super_gen`, `wgt_super` | 152 B |
| `bep-toy`, `level-bep`, `photonuclear-perturbation`, `superhistory-perts` | `super_gen`, `bep_tree`, `adjoint_id` | 144 B |
| `superhistoy-2` | `super_gen`, `adjoint_id` | 144 B |
| `claude/elegant-brahmagupta-fdujqt` | `lifetime` | 144 B |
| `shared-pht`, `claude/openmc-issue-4165-track-shared-secondary` | `parent_id` → `ancestor_index` (rename) | 136 B |

### Open upstream PRs (read-only, via git)

Open PRs were approximated by those with a `refs/pull/N/merge` ref (231),
plus recent PRs (#3700+) without one; GitHub API access to upstream was not
used.

| PR | adds | purpose |
|---|---|---|
| #4178 | `double lifetime` | IFP lifetime clock (same as the branches) |
| #4199 | `int birth_mesh_bin` | mesh bin of the fission-site parent's birth |
| #3479 | `double E_parent`, `int fission_nuclide` | sensitivity analysis |
| #3656 | `int n_repl` | split counter (duplicates `n_split`) |
| #3109 | `int source_index`, `bool ext` | which source a site came from |
| #3114, #2520, #4035 | `int source_id` | which source a site came from |
| #2196 (2022, stale) | separate `SecondarySourceSite` with `vector<LocalCoord>` | the only attempt at per-bank site types |

Merged as they are: 132 B of fields today + ~76 B (branches) + ~25 B
(upstream-only) ≈ 233 B → **≈240 B**, between the `pad64` and `pad128`
measurements: +70 % bank memory, +50 % bank time, a few % of run time at
10⁶ particles per rank, more at large N.

### The added data falls into a few concepts

| concept | fields (develop, branches, upstream) |
|---|---|
| lineage — which family, who is its root, how many generations since | `parent_id`, `progeny_id`, `ancestor_index`, `super_gen`, `bep_tree`, `adjoint_id`, `shadow_tag`, `shadow_depth` |
| clock | `time`, `lifetime`, `shadow_t0` |
| origin | `delayed_group`, `parent_nuclide`, `fission_nuclide`, `E_parent`, `birth_mesh_bin`, `source_id`/`source_index`/`ext` |
| weight control | `wgt_born`, `wgt_ww_born`, `n_split`, `n_repl`, `wgt_super` |
| split birth state | `r_born`, `cell_born` |

IFP, superhistory, BEP and the adjoint populations are conceptually the same
mechanism (estimate importance by following a tagged ancestor's descendants),
so a single generic lineage record (`family`, `root`, `generation`) used by all
of them would replace ~8 method-specific fields and would let the methods run
together, which they can conceptually (the real conflicts are two weight
controllers on one population, and per-history aggregation with the shared
bank, which needs a root id). Technically, today's code forbids BEP + IFP,
BEP + superhistory tallies and adjoint populations + shared bank / weight
windows / survival biasing / UFS; adjoint populations vs BEP/superhistory are
unchecked.

## Conclusion on the design options

| option | what the numbers say |
|---|---|
| fix the bank allocation waste (no `SourceSite` change) | **Do first.** Halves eigenvalue bank memory and removes most of the per-byte time; also lowers the cost of every future byte. |
| (1) unify feature fields into generic ones | **Yes.** Cheapest way to stop growth: the open fields overlap heavily (nuclide, clock, lineage, source id, split counters). Keeps one type, one MPI/HDF5 layout. |
| (3) fixed set of K tag slots | **Yes, as the generic lineage/origin record** rather than per-method reinterpreted slots, since the methods are conceptually combinable. |
| (4-hybrid) AoS core + optional side arrays on the large global banks | **Only for large, rarely used data** (e.g. `r_born`/`cell_born`, fission-site-only tags such as `birth_mesh_bin`/`E_parent`), following the existing IFP side arrays. |
| (4) full SoA banks | **Not justified.** The size-dependent run time is a few % and mostly removable by the allocation fix; SoA would add cache misses to record-at-a-time access (`create_fission_sites`, `from_source`), passes to the sort and messages to MPI exchanges. |
| (2) different site types per bank | **Not justified.** Saves ≈20–28 B on fission sites today, but the new fields are needed on both fission and secondary sites, and it doubles the MPI/HDF5/ctypes/MCPL plumbing. |

With the allocation fix and generic fields, `SourceSite` would stay around
150–160 B and cost ≈(k+2)·N·sizeof of resident memory instead of today's
7·N·136 B, i.e. less than develop uses now despite the added features.

## Limitations

* One 4-core node; MPI over shared memory, so inter-node message costs are not
  measured (send/recv grows with size but is 0.1–0.2 s here).
* Multigroup only. Continuous-energy transport is several times slower per
  history, so bank shares of run time would be smaller there; memory
  numbers are unchanged.
* Total-time differences are within the transport noise of this machine
  (shared cloud VM, ±1–10 %); the bank timers, perf shares and RSS are
  reliable. 5·10⁶ and 10⁷ cases are single runs.
* After a container restart late in the session the host's absolute timings
  changed (≈3× slower `synchronize_bank`); only the page-fault counts from
  that period are used.
* The `min` variant removes fields and their stores as well as bytes, and it
  crosses glibc's 32 MB mmap threshold at 10⁵ particles, so 104 B points are
  not pure size effects; the `pad64`/`pad128` points are.
* The branch and PR survey reads struct definitions and compatibility checks
  only; the methods were not run.
