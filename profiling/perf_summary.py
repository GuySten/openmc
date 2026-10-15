#!/usr/bin/env python3
"""Summarize bank-related functions from a 'perf record' profile.

Prints, for each function of interest, the inclusive ("children") share of
all samples and the corresponding CPU seconds (samples / sampling frequency).
For serial code (sort_bank, synchronize_bank, the secondary-bank collection
and MPI redistribution) CPU seconds equal wall seconds; for code that runs
inside OpenMP regions (create_fission_sites, from_source, ...) divide by the
number of threads to get the wall-time share.

Usage: perf_summary.py PERF_DATA [--freq 999] [--json]
"""
import argparse
import json
import os
import re
import subprocess

PERF = os.environ.get('PERF', '/usr/lib/linux-tools-6.8.0-147/perf')

# label: regex on the demangled symbol (matched against perf's symbol column)
FUNCS = {
    'sort_bank': r'openmc::sort_bank\(',
    'synchronize_bank': r'openmc::synchronize_bank\(\)',
    'sync_global_secondary_bank':
        r'openmc::synchronize_global_secondary_bank\(',
    'collect_sorted_secondary_banks':
        r'collect_sorted_history_secondary_banks\(',
    'thread_bank_push_back': r'vector<openmc::SourceSite.*_M_realloc_insert',
    'create_fission_sites': r'openmc::create_fission_sites\(',
    'create_secondary': r'openmc::Particle::create_secondary\(',
    'split': r'openmc::Particle::split\(',
    'from_source': r'openmc::Particle::from_source\(',
    'revive_from_secondary': r'event_revive_from_secondary\(',
    'add_surf_source_to_bank': r'openmc::add_surf_source_to_bank\(',
    'write_source_bank': r'openmc::write_source_bank\(',
    'write_h5_source_point': r'openmc::write_h5_source_point\(',
    'initialize_source': r'^initialize_source$',
    'init_fission_bank': r'openmc::init_fission_bank\(',
    'allocate_banks': r'openmc::allocate_banks\(',
    'page_fault (kernel)': r'^asm_exc_page_fault$',
    'transport (all)': r'openmc::transport_history_based_single_particle\(',
}


def summarize(path, freq):
    out = subprocess.run(
        [PERF, 'report', '-i', path, '--children', '--sort', 'symbol', '-g',
         'none', '--stdio'], capture_output=True, text=True).stdout
    m = re.search(r'# Samples: (\d+)([KM]?)', out)
    mult = {'': 1, 'K': 1e3, 'M': 1e6}[m.group(2)]
    approx_samples = int(m.group(1)) * mult
    m = re.search(r'# Event count \(approx.\): (\d+)', out)
    # cpu-clock event count is in ns of CPU time
    cpu_seconds = int(m.group(1)) * 1e-9 if m else approx_samples / freq
    rows = []
    for line in out.splitlines():
        mm = re.match(r'\s+([\d.]+)%\s+([\d.]+)%\s+\[(.)\]\s+(.*?)\s*$',
                      line)
        if mm:
            rows.append((float(mm.group(1)), float(mm.group(2)),
                         mm.group(3), mm.group(4)))
    res = {'cpu_seconds_total': cpu_seconds}
    for label, pat in FUNCS.items():
        children = self_ = 0.0
        for ch, se, _, sym in rows:
            if '@plt' in sym:
                continue
            if re.search(pat, sym):
                # Several clones (e.g. OpenMP outlined fns) may match; the
                # inclusive share of the outermost one is the max.
                children = max(children, ch)
                self_ += se
        res[label] = {'children_pct': children, 'self_pct': self_,
                      'cpu_s': children / 100 * cpu_seconds}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('perf_data')
    ap.add_argument('--freq', type=float, default=999)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    res = summarize(a.perf_data, a.freq)
    if a.json:
        print(json.dumps(res))
        return
    print(f"total CPU seconds sampled: {res['cpu_seconds_total']:.2f}")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:32s} {v['children_pct']:7.3f}% incl {v['self_pct']:7.3f}% "
                  f"self {v['cpu_s']:9.3f} cpu-s")


if __name__ == '__main__':
    main()
