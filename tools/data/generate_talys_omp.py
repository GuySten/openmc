#!/usr/bin/env python3
"""Generate openmc/data/talys_neutron_omp.json from a TALYS source tree.

The file holds the nucleus-specific Koning-Delaroche neutron parameters of
structure/optical/neutron/n-*.omp and the neutron optical-model adjustments
of structure/best/*/n-*.talys that TALYS uses to produce TENDL.

Usage: generate_talys_omp.py /path/to/talys [output.json]

TALYS: https://github.com/arjankoning1/talys (MIT licence)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from openmc.data import ATOMIC_NUMBER
from openmc.data.talys import read_best_adjustments, read_omp_file


def main(talys, output):
    talys = Path(talys)
    commit = subprocess.run(['git', '-C', str(talys), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    version = re.search(r'TALYS-([0-9.]+)',
                        (talys / 'source' / 'talys.f90').read_text()).group(1)

    local = {}
    for path in sorted((talys / 'structure/optical/neutron').glob('n-*.omp')):
        symbol = path.stem[2:]
        for A, params in read_omp_file(path).items():
            local[f'{symbol}{A}'] = params

    adjust = {}
    for path in sorted((talys / 'structure/best').glob('*/n-*.talys')):
        m = re.fullmatch(r'n-([A-Z][a-z]?)0*(\d+)(m?)\.talys', path.name)
        if m is None or m.group(3):
            continue
        name = f'{m.group(1)}{m.group(2)}'
        assert m.group(1) in ATOMIC_NUMBER
        adj = read_best_adjustments(path)
        if adj:
            adjust[name] = adj

    data = {
        'source': f'TALYS-{version}, https://github.com/arjankoning1/talys '
                  f'commit {commit}',
        'license': 'MIT License, Copyright (c) A.J. Koning',
        'local': local,
        'adjust': adjust,
    }
    Path(output).write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    print(f'{len(local)} local parameter sets, '
          f'{len(adjust)} nuclides with adjustments')


if __name__ == '__main__':
    default = Path(__file__).parents[2] / 'openmc/data/talys_neutron_omp.json'
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else default)
