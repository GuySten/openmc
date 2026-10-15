#!/usr/bin/env python3
"""Run the SourceSite benchmarks for several builds and collect timings.

Each run records OpenMC's own timers (parsed from stdout), wall time and peak
RSS (from /usr/bin/time), and appends one JSON line per run to --results.

Example:
  python run_bench.py --builds ~/builds --variants base min pad64 pad128 \\
      --cases eigen-1e6 fixed-1e5 --repeat 3 --results results/serial.jsonl

Cases are defined in CASES below; each is (model.py arguments, MPI ranks,
OpenMP threads per rank). An MPI case is run with the '<variant>-mpi' build.
With --perf, each run is wrapped in 'perf record' and a per-function profile
summary is stored next to the run directory (see perf_summary.py).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = os.environ.get('PERF', '/usr/lib/linux-tools-6.8.0-147/perf')
NPROC = os.cpu_count()

# name: (model.py args, mpi ranks, threads per rank)
CASES = {
    'eigen-1e5': ('eigen --particles 100000 --batches 20 --inactive 10', 1, NPROC),
    'eigen-1e6': ('eigen --particles 1000000 --batches 6 --inactive 3', 1, NPROC),
    'eigen-5e6': ('eigen --particles 5000000 --batches 3 --inactive 1', 1, NPROC),
    'eigen-1e7': ('eigen --particles 10000000 --batches 3 --inactive 1', 1, NPROC),
    'fixed-1e5': ('fixed --lattice 30 --particles 100000 --batches 5', 1, NPROC),
    'fixed-1e6': ('fixed --lattice 30 --particles 1000000 --batches 2', 1, NPROC),
    'fixed-1e5-local': ('fixed --lattice 30 --particles 100000 --batches 5 '
                        '--shared-bank false', 1, NPROC),
    'fixed-1e5-surf': ('fixed --lattice 30 --particles 100000 --batches 5 '
                       '--surf-write 2000000', 1, NPROC),
    # MPI: particles are per run, split across ranks
    'mpi4-eigen-4e6': ('eigen --particles 4000000 --batches 6 --inactive 3', 4, 1),
    'mpi4-eigen-4e5': ('eigen --particles 400000 --batches 20 --inactive 10', 4, 1),
    'mpi4-fixed-4e5': ('fixed --lattice 30 --particles 400000 --batches 5', 4, 1),
    'mpi2-eigen-2e6': ('eigen --particles 2000000 --batches 6 --inactive 3', 2, 2),
}

TIMERS = {
    'init': r'Total time for initialization\s+=\s+(\S+)',
    'simulation': r'Total time in simulation\s+=\s+(\S+)',
    'transport': r'Time in transport only\s+=\s+(\S+)',
    'inactive': r'Time in inactive batches\s+=\s+(\S+)',
    'active': r'Time in active batches\s+=\s+(\S+)',
    'bank_sync': r'Time synchronizing fission bank\s+=\s+(\S+)',
    'bank_sample': r'Sampling source sites\s+=\s+(\S+)',
    'bank_sendrecv': r'SEND/RECV source sites\s+=\s+(\S+)',
    'statepoint': r'Time writing statepoints\s+=\s+(\S+)',
    'elapsed': r'Total time elapsed\s+=\s+(\S+)',
    'rate_active': r'Calculation Rate \(active\)\s+=\s+(\S+)',
}


def parse(stdout):
    out = {}
    for key, pat in TIMERS.items():
        m = re.search(pat, stdout)
        if m:
            out[key] = float(m.group(1))
    gens = re.findall(r'Secondary generation\s+\d+\s+tracks:\s+(\d+)', stdout)
    if gens:
        tracks = [int(g) for g in gens]
        out['secondary_tracks_total'] = sum(tracks)
        out['secondary_tracks_max_generation'] = max(tracks)
        out['secondary_generations'] = len(tracks)
    m = re.search(r'Creating source file .* with (\d+) particles', stdout)
    if m:
        out['surface_sites'] = int(m.group(1))
    return out


def prepare(case, workdir):
    args, _, _ = CASES[case]
    d = workdir / case
    if not (d / 'model.xml').exists():
        subprocess.run([sys.executable, str(HERE / 'model.py'), *args.split(),
                        '-o', str(d)], check=True, stdout=subprocess.DEVNULL)
    return d


def run(build_root, variant, case, workdir, perf=False, tag=''):
    _, ranks, threads = CASES[case]
    d = prepare(case, workdir)
    bdir = Path(build_root) / (variant + ('-mpi' if ranks > 1 else ''))
    exe = bdir / 'bin' / 'openmc'
    cmd = [str(exe)]
    if ranks > 1:
        cmd = ['mpiexec', '--allow-run-as-root', '--oversubscribe',
               '--bind-to', 'none', '-n', str(ranks)] + cmd
    timefile = d / 'time.txt'
    cmd = ['/usr/bin/time', '-f', '%e %M', '-o', str(timefile)] + cmd
    if perf:
        pdata = d / f'perf.{variant}{tag}.data'
        cmd = [PERF, 'record', '-e', 'cpu-clock', '-F', '999', '--call-graph',
               'fp', '-o', str(pdata)] + cmd
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               OMP_PROC_BIND='false')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    (d / f'stdout.{variant}{tag}.txt').write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f'{variant} {case} failed:\n{p.stdout[-3000:]}'
                           f'\n{p.stderr[-3000:]}')
    rec = {'variant': variant, 'case': case, 'ranks': ranks,
           'threads': threads, 'perf': perf, 'wall': wall}
    rec.update(parse(p.stdout))
    # /usr/bin/time wraps mpiexec, so with MPI %M is the largest single
    # process (one rank), not the sum.
    elapsed, maxrss = timefile.read_text().split()[-2:]
    rec['time_elapsed'] = float(elapsed)
    rec['maxrss_kb'] = int(maxrss)
    for f in d.glob('statepoint.*.h5'):
        f.unlink()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--builds', default=os.path.expanduser('~/builds'))
    ap.add_argument('--workdir', default=os.path.expanduser('~/runs'))
    ap.add_argument('--variants', nargs='+',
                    default=['base', 'min', 'pad64', 'pad128'])
    ap.add_argument('--cases', nargs='+', required=True)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--perf', action='store_true')
    ap.add_argument('--results', required=True)
    a = ap.parse_args()
    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    # Interleave variants within each repeat so that drifts in machine load
    # affect all variants equally.
    for case in a.cases:
        for rep in range(a.repeat):
            for v in a.variants:
                try:
                    rec = run(a.builds, v, case, workdir, perf=a.perf,
                              tag=f'.r{rep}')
                except RuntimeError as e:
                    print(e, file=sys.stderr)
                    rec = {'variant': v, 'case': case, 'failed': True}
                rec['repeat'] = rep
                print(json.dumps(rec), flush=True)
                with open(a.results, 'a') as f:
                    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
