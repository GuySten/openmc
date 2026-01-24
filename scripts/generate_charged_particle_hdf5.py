#!/usr/bin/env python3
"""
Generate HDF5 files from ENDF charged particle data.

This script processes all incident charged particle ENDF files and converts
them to OpenMC's HDF5 format for use in charged particle transport.

Usage:
    python generate_charged_particle_hdf5.py /path/to/ENDF-B-VIII.1 /path/to/output
"""

import argparse
import os
from pathlib import Path

import openmc.data


# Mapping of particle types to their ENDF directory names
PARTICLE_DIRS = {
    'proton': ['protons', 'protons-version.VIII.1'],
    'deuteron': ['deuterons', 'deuterons-version.VIII.1'],
    'triton': ['tritons', 'tritons-version.VIII.1'],
    'helion': ['helium3s', 'helium3s-version.VIII.1'],  # He-3
    'alpha': ['alphas', 'alphas-version.VIII.1'],
}


def find_endf_files(endf_root: Path) -> dict:
    """Find all ENDF charged particle files in the given directory."""
    files_by_particle = {}

    for particle, dir_names in PARTICLE_DIRS.items():
        files_by_particle[particle] = []

        for dir_name in dir_names:
            particle_dir = endf_root / dir_name
            if particle_dir.exists():
                endf_files = list(particle_dir.glob('*.endf'))
                files_by_particle[particle].extend(endf_files)
                break  # Use first matching directory

    return files_by_particle


def process_particle_files(endf_files: list, output_dir: Path, particle_type: str):
    """Process ENDF files for a single particle type."""

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for endf_path in sorted(endf_files):
        try:
            # Load from ENDF
            data = openmc.data.IncidentChargedParticle.from_endf(endf_path)

            # Generate output filename: {particle}_{target}.h5
            # e.g., deuteron_H2.h5, proton_Li7.h5
            output_name = f"{data.projectile}_{data.target_name}.h5"
            output_path = output_dir / output_name

            # Export to HDF5
            data.export_to_hdf5(output_path, mode='w')

            # Print summary
            n_reactions = len(data.reactions)
            mts = list(data.reactions.keys())
            print(f"  ✓ {output_name}: {n_reactions} reactions {mts}")

            successful += 1

        except Exception as e:
            print(f"  ✗ {endf_path.name}: {e}")
            failed += 1

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Generate HDF5 files from ENDF charged particle data'
    )
    parser.add_argument(
        'endf_root',
        type=Path,
        help='Root directory of ENDF library (e.g., /path/to/ENDF-B-VIII.1)'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for HDF5 files'
    )
    parser.add_argument(
        '--particle',
        choices=['proton', 'deuteron', 'triton', 'helion', 'alpha', 'all'],
        default='all',
        help='Process only specified particle type (default: all)'
    )

    args = parser.parse_args()

    if not args.endf_root.exists():
        print(f"Error: ENDF root directory not found: {args.endf_root}")
        return 1

    # Find all ENDF files
    files_by_particle = find_endf_files(args.endf_root)

    total_successful = 0
    total_failed = 0

    print(f"\nGenerating charged particle HDF5 from: {args.endf_root}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    for particle, endf_files in files_by_particle.items():
        if args.particle != 'all' and args.particle != particle:
            continue

        if not endf_files:
            print(f"\n{particle.upper()}: No ENDF files found")
            continue

        print(f"\n{particle.upper()}: {len(endf_files)} files")
        print("-" * 40)

        # Create particle-specific output directory
        particle_output = args.output_dir / particle

        successful, failed = process_particle_files(
            endf_files, particle_output, particle
        )

        total_successful += successful
        total_failed += failed

    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_successful} successful, {total_failed} failed")
    print(f"Output: {args.output_dir}")

    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    exit(main())
