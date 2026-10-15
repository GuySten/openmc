#!/usr/bin/env python3
"""Turn results/*.jsonl into the markdown tables used in REPORT.md.

For each case: mean (and spread) of wall time, OpenMC bank timers and peak
RSS per variant, plus a least-squares slope of each quantity versus
sizeof(SourceSite).

Usage: analyze.py results/serial.jsonl [results/mpi.jsonl ...]
"""
import json
import statistics as st
import sys
from collections import defaultdict

SIZE = {'min': 104, 'base': 136, 'pad64': 200, 'pad128': 264}
ORDER = ['min', 'base', 'pad64', 'pad128']


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx)**2 for x in xs)
    return sum((x - mx)*(y - my) for x, y in zip(xs, ys)) / sxx


def fmt(vals, digits=2):
    if not vals:
        return '-'
    m = st.mean(vals)
    if len(vals) > 1:
        return f'{m:.{digits}f} ± {(max(vals) - min(vals)) / 2:.{digits}f}'
    return f'{m:.{digits}f}'


def main(files):
    recs = []
    for f in files:
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    recs = [r for r in recs if not r.get('perf') and not r.get('failed')]
    by = defaultdict(list)
    for r in recs:
        by[(r['case'], r['variant'])].append(r)
    cases = sorted({r['case'] for r in recs},
                   key=lambda c: (c.startswith('mpi'), c))

    print('| case | sizeof | n | elapsed [s] | transport [s] | '
          'non-transport [s] | sync bank [s] | sample [s] | send/recv [s] | '
          'peak RSS [MB] |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    slopes = []
    for c in cases:
        xs = defaultdict(list)
        for v in ORDER:
            rs = by.get((c, v))
            if not rs:
                continue
            g = lambda k: [r[k] for r in rs if k in r]
            nont = [r['elapsed'] - r['transport'] for r in rs]
            rss = [r['maxrss_kb'] / 1024 for r in rs]
            print(f"| {c} | {SIZE[v]} | {len(rs)} | {fmt(g('elapsed'))} | "
                  f"{fmt(g('transport'))} | {fmt(nont)} | "
                  f"{fmt(g('bank_sync'), 3)} | {fmt(g('bank_sample'), 3)} | "
                  f"{fmt(g('bank_sendrecv'), 3)} | {fmt(rss, 0)} |")
            for r in rs:
                for k in ('elapsed', 'transport', 'bank_sync', 'bank_sample',
                          'bank_sendrecv'):
                    if k in r:
                        xs[k].append((SIZE[v], r[k]))
                xs['non_transport'].append((SIZE[v],
                                            r['elapsed'] - r['transport']))
                xs['rss_mb'].append((SIZE[v], r['maxrss_kb'] / 1024))
        row = {'case': c}
        for k, pts in xs.items():
            if len({p[0] for p in pts}) > 1:
                row[k] = slope([p[0] for p in pts], [p[1] for p in pts])
        base = by.get((c, 'base'))
        if base:
            row['elapsed_base'] = st.mean(r['elapsed'] for r in base)
        slopes.append(row)

    print()
    print('Slopes per byte of sizeof(SourceSite) (least squares over all '
          'runs of the four variants):')
    print()
    print('| case | elapsed [ms/B] | elapsed [%/B of base] | '
          'non-transport [ms/B] | sync bank [ms/B] | peak RSS [MB/B] | '
          'elapsed change 136→104 B [%] |')
    print('|---|---|---|---|---|---|---|')
    for r in slopes:
        e = r.get('elapsed')
        eb = r.get('elapsed_base')
        pct = 100 * e / eb if e is not None and eb else None
        g = lambda k, s=1e3, d=2: (f'{r[k]*s:.{d}f}' if k in r else '-')
        print(f"| {r['case']} | {g('elapsed')} | "
              f"{'-' if pct is None else f'{pct:.4f}'} | "
              f"{g('non_transport')} | {g('bank_sync')} | "
              f"{g('rss_mb', 1, 3)} | "
              f"{'-' if pct is None else f'{-32*pct:.2f}'} |")


if __name__ == '__main__':
    main(sys.argv[1:])
