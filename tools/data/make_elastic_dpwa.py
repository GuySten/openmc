#!/usr/bin/env python
"""Generate DPWA elastic-scattering cross sections for electron transport.

EPICS/EEDL tabulates elastic cross sections densely but the angular
distributions on only 16 energies per element, with nothing between 0.256 and
10 MeV.  A Dirac partial-wave calculation has no such gap.  This script runs
ELSEPA once per element and projectile over PENELOPE's 96-point energy grid and
writes the differential cross sections to a single HDF5 file, with one
zero-padded atomic-number group per element holding an 'electron' and a
'positron' subgroup.  Nothing else is written.

Both projectiles are needed.  The integrated cross sections differ by under a
per cent, which is the Born limit and is symmetric in the charge, but the first
transport cross section is not: a positron is repelled by the nucleus and stays
out of the small-impact-parameter region that produces the large deflections.
For lead its sigma_tr1 is 0.80 of the electron's at 21 MeV and 0.65 at 1 MeV.  ELSEPA also reports
the integrated and the first- and second-transport cross sections, computed
from the phase shifts rather than from the tabulated distribution, and they
disagree with integrals of that distribution by up to about a percent.  The
consumer integrates the table it samples, so shipping those numbers beside it
would only invite something to use the wrong one.

ELSEPA is run with a Fermi nuclear charge distribution, Dirac-Fock electron
density, Furness-McCarthy exchange, LDA correlation-polarization and no
absorption (an inelastic channel that is modelled separately), which are the
settings under which the published DPWA databases are built.  Run for carbon,
the output reproduces PENELOPE's own ELSEPA-derived database to four decimal
places in all three cross sections from 100 keV to 100 MeV.

ELSEPA is the Dirac partial-wave code of Salvat, Jablonski and Powell,
Computer Physics Communications 165 (2005) 157-190,
https://www.sciencedirect.com/science/article/pii/S0010465504004795.  It is
Apache-2.0, repackaged by J. Hidding of the Netherlands eScience Center at
https://github.com/eScienceCenter/elsepa.  Point --elscata and --elsepa-data
at a built copy.
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from openmc.data import ATOMIC_SYMBOL

# Angular grid thinning, applied when the file is written. ELSEPA's grid is
# geometric below this deflection and uniform above it.
WIDE_ANGLE = 0.1
WIDE_ANGLE_STRIDE = 4



# PENELOPE's electron energy grid in eV, kept identical so the two databases
# can be compared point by point
energies = np.array([
    50, 60, 70, 80, 90, 100, 125, 150,
    175, 200, 250, 300, 350, 400, 450, 500,
    600, 700, 800, 900, 1000, 1250, 1500, 1750,
    2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000,
    7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000,
    25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000,
    80000, 90000, 100000, 125000, 150000, 175000, 200000, 250000,
    300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000,
    900000, 1e+06, 1.25e+06, 1.5e+06, 1.75e+06, 2e+06, 2.5e+06, 3e+06,
    3.5e+06, 4e+06, 4.5e+06, 5e+06, 6e+06, 7e+06, 8e+06, 9e+06,
    1e+07, 1.25e+07, 1.5e+07, 1.75e+07, 2e+07, 2.5e+07, 3e+07, 3.5e+07,
    4e+07, 4.5e+07, 5e+07, 6e+07, 7e+07, 8e+07, 9e+07, 1e+08,])

# ELSEPA writes its differential cross sections on a fixed angular grid
n_angles = 606

# Kinetic energy appears in the header of each dcs_*.dat file
energy_re = re.compile(r'Kinetic energy\s*=\s*([0-9.E+-]+)\s*eV')


def write_input(Z, path, ielec):
    """Write an ELSEPA input deck covering the whole energy grid.

    ``ielec`` is -1 for an electron projectile and +1 for a positron.
    """
    with open(path, 'w') as f:
        f.write(f'IZ    {Z:4}          atomic number\n')
        f.write('MNUCL   3          Fermi nuclear charge distribution\n')
        f.write(f'NELEC {Z:4}          neutral atom\n')
        f.write('MELEC   4          Dirac-Fock electron density\n')
        f.write('MUFFIN  0          free atom\n')
        f.write(f'IELEC {ielec:4}          {"electron" if ielec < 0 else "positron"}\n')
        f.write('MEXCH   1          Furness-McCarthy exchange\n')
        f.write('MCPOL   2          LDA correlation-polarization\n')
        f.write('VPOLA  -1          measured atomic polarizability\n')
        f.write('VPOLB  -1          default b_pol\n')
        f.write('MABS    0          no absorption; an inelastic channel\n')
        f.write('IHEF    1          Born factorization at high energy\n')
        for energy in energies:
            f.write(f'EV      {energy:.4E}\n')


def read_dcs(path):
    """Read one ELSEPA differential cross section file.

    Returns the kinetic energy in eV, 1 - cos(theta) on the angular grid, and
    the cross section in cm^2/sr.  ELSEPA tabulates (1 - cos(theta))/2, which is
    doubled here to match the deflection variable the transport samples.
    """
    energy = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith('#'):
                if energy is None:
                    match = energy_re.search(line)
                    if match:
                        energy = float(match.group(1))
            else:
                words = line.split()
                if len(words) >= 4:
                    rows.append((float(words[1]), float(words[2])))

    if energy is None or len(rows) != n_angles:
        raise ValueError(f'{path}: {len(rows)} angles, energy {energy}')
    values = np.array(rows)
    return energy, 2.0*values[:, 0], values[:, 1]


def run_element(Z, elscata, elsepa_data, ielec):
    """Run ELSEPA for one element and projectile, and return its DCS."""
    workdir = tempfile.mkdtemp(prefix=f'elsepa_z{Z:03}_')
    try:
        write_input(Z, os.path.join(workdir, 'in.txt'), ielec)
        env = dict(os.environ, ELSEPA_DATA=str(elsepa_data))
        with open(os.path.join(workdir, 'in.txt')) as stdin, \
             open(os.path.join(workdir, 'run.log'), 'w') as stdout:
            subprocess.run([str(elscata)], stdin=stdin, stdout=stdout,
                           stderr=subprocess.STDOUT, cwd=workdir, env=env,
                           check=True)

        paths = sorted(Path(workdir).glob('dcs_*.dat'))
        if len(paths) != energies.size:
            raise RuntimeError(f'Z={Z}: {len(paths)} of {energies.size} '
                               'energies produced a cross section')

        mu = None
        dcs = np.empty((energies.size, n_angles))
        found = np.empty(energies.size)
        for i, path in enumerate(paths):
            found[i], mu_i, dcs[i] = read_dcs(path)
            if mu is None:
                mu = mu_i

        order = np.argsort(found)
        found, dcs = found[order], dcs[order]
        if not np.allclose(found, energies, rtol=1e-6):
            raise RuntimeError(f'Z={Z}: energies do not match the grid')

        return Z, ielec, mu, dcs
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate an HDF5 library of DPWA elastic-scattering '
        'cross sections by running ELSEPA.'
    )
    parser.add_argument(
        '-o', '--output', default='elastic_dpwa.h5',
        help='output HDF5 filename (default: %(default)s)'
    )
    parser.add_argument(
        '--zmin', type=int, default=1,
        help='lowest atomic number (default: %(default)s)'
    )
    parser.add_argument(
        '--zmax', type=int, default=100,
        help='highest atomic number (default: %(default)s)'
    )
    parser.add_argument(
        '--jobs', type=int, default=os.cpu_count() or 1,
        help='elements to run concurrently (default: %(default)s)'
    )
    parser.add_argument(
        '--elscata', default=os.environ.get('ELSEPA', 'elscata'),
        help="path to the ELSEPA 'elscata' executable"
    )
    parser.add_argument(
        '--elsepa-data', default=os.environ.get('ELSEPA_DATA'),
        help="path to the ELSEPA data directory holding the z_NNN.den files"
    )
    args = parser.parse_args()

    if args.elsepa_data is None:
        parser.error('pass --elsepa-data or set ELSEPA_DATA')
    elscata = shutil.which(args.elscata) or args.elscata
    if not os.path.exists(elscata):
        parser.error(f'ELSEPA executable not found: {elscata}')

    # ==========================================================================
    # RUN ELSEPA FOR EACH ELEMENT AND GENERATE ELASTIC DPWA HDF5 FILE

    print(f'Generating {args.output}...')

    atomic_numbers = range(args.zmin, args.zmax + 1)
    projectiles = {-1: 'electron', 1: 'positron'}
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_element, Z, elscata, args.elsepa_data, ielec)
                   for Z in atomic_numbers for ielec in projectiles]
        for future in futures:
            Z, ielec, mu, dcs = future.result()
            print('Processing {} {} data...'.format(
                ATOMIC_SYMBOL[Z], projectiles[ielec]))
            results[Z, ielec] = (mu, dcs)

    reference_mu = results[args.zmin, -1][0]

    # Thin the wide-angle end of ELSEPA's grid. Below 1-cos(theta) = 0.1 the
    # grid is geometric and carries the forward peak, so every point is kept.
    # Above it the grid is uniform half-degree steps, and taking one in four
    # changes the integrated, first and second transport cross sections by at
    # most 5e-4 -- against the 0.2-1.4% by which the tabulated integral already
    # differs from ELSEPA's own phase-shift totals. It drops the file from
    # 24.4 MB to 14.9 MB, which matters because this ships in every wheel.
    keep = reference_mu < WIDE_ANGLE
    wide = np.where(~keep)[0]
    keep[wide[::WIDE_ANGLE_STRIDE]] = True
    keep[0] = True
    keep[-1] = True
    reference_mu = reference_mu[keep]

    with h5py.File(args.output, 'w') as f:
        f.attrs['filetype'] = np.bytes_('elastic_dpwa')
        f.attrs['source'] = np.bytes_(
            'Dirac partial-wave elastic cross sections computed with ELSEPA: '
            'Salvat, Jablonski and Powell, Comput. Phys. Commun. 165 (2005) '
            '157-190. Generated by make_elastic_dpwa.py.'
        )

        # Write energies and the shared angular grid
        f.create_dataset('energy', data=energies, compression='gzip',
                         compression_opts=9, shuffle=True)
        f.create_dataset('mu', data=reference_mu, compression='gzip',
                         compression_opts=9, shuffle=True)

        for Z in atomic_numbers:
            # Create group for this element
            group = f.create_group(f'{Z:03}')
            for ielec, name in projectiles.items():
                mu, dcs = results[Z, ielec]
                if not np.allclose(mu[keep], reference_mu, rtol=1e-12,
                                   atol=0.0):
                    raise RuntimeError(f'Z={Z} {name}: angular grid differs')
                dcs = dcs[:, keep]

                # Stored as log(dcs): the cross section spans many decades and
                # its logarithm is smooth, which compresses to 9.9 MB against
                # 16.9 MB for the cross section itself. Round trip is accurate
                # to 4e-6, well inside the precision of the calculation.
                log_dcs = np.log(dcs).astype(np.float32)
                group.create_group(name).create_dataset(
                    'log_dcs', data=log_dcs, chunks=log_dcs.shape,
                    compression='gzip', compression_opts=9, shuffle=True)

    size = os.path.getsize(args.output) / 1e6
    print(f'Wrote {args.output} ({size:.1f} MB)')


if __name__ == '__main__':
    main()
