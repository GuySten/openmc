"""Multigroup benchmark models that stress OpenMC's SourceSite banks.

Uses the 7-group C5G7 UO2/water data from examples/pincell_multigroup, so no
cross-section download is needed.

  eigen  : infinite (reflective) UO2 pin cell, eigenvalue mode. Stresses the
           fission bank, sort_bank() and synchronize_bank().
  fixed  : finite NxN lattice of the same pins, vacuum radially, reflective
           axially, with a fixed source. Fission neutrons become secondaries,
           and the shared secondary bank is forced on, so the global
           secondary bank, its sort and its MPI redistribution are exercised.
           With --surf-write, sites crossing the outer boundary are written
           to a surface source file.

Example:
  python model.py eigen  -o run_dir --particles 1000000 --batches 6 --inactive 3
  python model.py fixed  -o run_dir --particles 100000 --batches 3 --lattice 17
"""
import argparse
from pathlib import Path

import numpy as np
import openmc

GROUP_EDGES = [1e-5, 0.0635, 10.0, 1.0e2, 1.0e3, 0.5e6, 1.0e6, 20.0e6]


def make_library(path):
    groups = openmc.mgxs.EnergyGroups(group_edges=GROUP_EDGES)
    uo2 = openmc.XSdata('UO2', groups)
    uo2.order = 0
    uo2.set_total([0.1779492, 0.3298048, 0.4803882, 0.5543674, 0.3118013,
                   0.3951678, 0.5644058])
    uo2.set_absorption([8.0248E-03, 3.7174E-03, 2.6769E-02, 9.6236E-02,
                        3.0020E-02, 1.1126E-01, 2.8278E-01])
    s = np.array(
        [[[0.1275370, 0.0423780, 0.0000094, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.3244560, 0.0016314, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.4509400, 0.0026792, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.4525650, 0.0055664, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0001253, 0.2714010, 0.0102550, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0012968, 0.2658020, 0.0168090],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0085458, 0.2730800]]])
    uo2.set_scatter_matrix(np.rollaxis(s, 0, 3))
    uo2.set_fission([7.21206E-03, 8.19301E-04, 6.45320E-03, 1.85648E-02,
                     1.78084E-02, 8.30348E-02, 2.16004E-01])
    uo2.set_nu_fission([2.005998E-02, 2.027303E-03, 1.570599E-02,
                        4.518301E-02, 4.334208E-02, 2.020901E-01,
                        5.257105E-01])
    uo2.set_chi([5.8791E-01, 4.1176E-01, 3.3906E-04, 1.1761E-07, 0.0, 0.0,
                 0.0])

    h2o = openmc.XSdata('LWTR', groups)
    h2o.order = 0
    h2o.set_total([0.15920605, 0.412969593, 0.59030986, 0.58435, 0.718,
                   1.2544497, 2.650379])
    h2o.set_absorption([6.0105E-04, 1.5793E-05, 3.3716E-04, 1.9406E-03,
                        5.7416E-03, 1.5001E-02, 3.7239E-02])
    s = np.array(
        [[[0.0444777, 0.1134000, 0.0007235, 0.0000037, 0.0000001, 0.0, 0.0],
          [0.0, 0.2823340, 0.1299400, 0.0006234, 0.0000480, 0.0000074,
           0.0000010],
          [0.0, 0.0, 0.3452560, 0.2245700, 0.0169990, 0.0026443, 0.0005034],
          [0.0, 0.0, 0.0, 0.0910284, 0.4155100, 0.0637320, 0.0121390],
          [0.0, 0.0, 0.0, 0.0000714, 0.1391380, 0.5118200, 0.0612290],
          [0.0, 0.0, 0.0, 0.0, 0.0022157, 0.6999130, 0.5373200],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.1324400, 2.4807000]]])
    h2o.set_scatter_matrix(np.rollaxis(s, 0, 3))

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([uo2, h2o])
    lib.export_to_hdf5(str(path))


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    make_library(out / 'mgxs.h5')

    uo2 = openmc.Material(name='UO2 fuel')
    uo2.set_density('macro', 1.0)
    uo2.add_macroscopic('UO2')
    water = openmc.Material(name='Water')
    water.set_density('macro', 1.0)
    water.add_macroscopic('LWTR')
    materials = openmc.Materials([uo2, water])
    materials.cross_sections = (out / 'mgxs.h5').resolve()

    pitch = 1.26
    fuel_or = openmc.ZCylinder(r=0.54)
    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.particles = args.particles
    settings.batches = args.batches
    settings.output = {'tallies': False}
    settings.sourcepoint = {'write': False}
    settings.statepoint = {'batches': [args.batches]}

    if args.mode == 'eigen':
        box = openmc.model.RectangularPrism(pitch, pitch,
                                            boundary_type='reflective')
        cells = [openmc.Cell(fill=uo2, region=-fuel_or),
                 openmc.Cell(fill=water, region=+fuel_or & -box)]
        geometry = openmc.Geometry(cells)
        settings.run_mode = 'eigenvalue'
        settings.inactive = args.inactive
        ll, ur = (-pitch/2, -pitch/2, -1), (pitch/2, pitch/2, 1)
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(ll, ur),
            constraints={'fissionable': True})
    else:
        n = args.lattice
        pin = openmc.Universe(cells=[
            openmc.Cell(fill=uo2, region=-fuel_or),
            openmc.Cell(fill=water, region=+fuel_or)])
        lat = openmc.RectLattice()
        lat.pitch = (pitch, pitch)
        lat.lower_left = (-n*pitch/2, -n*pitch/2)
        lat.universes = [[pin]*n for _ in range(n)]
        half = n*pitch/2
        xy = openmc.model.RectangularPrism(2*half, 2*half,
                                           boundary_type='vacuum')
        zlo = openmc.ZPlane(-10.0, boundary_type='reflective')
        zhi = openmc.ZPlane(10.0, boundary_type='reflective')
        root = openmc.Cell(fill=lat, region=-xy & +zlo & -zhi)
        geometry = openmc.Geometry([root])
        settings.run_mode = 'fixed source'
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box((-half, -half, -10), (half, half, 10)),
            energy=openmc.stats.Discrete([2.0e6], [1.0]),
            constraints={'fissionable': True})
        settings.shared_secondary_bank = args.shared_bank
        if args.surf_write:
            settings.surf_source_write = {
                'surface_ids': [s.id for s in xy.component_surfaces],
                'max_particles': args.surf_write}

    model = openmc.Model(geometry, materials, settings)
    model.export_to_model_xml(out / 'model.xml')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['eigen', 'fixed'])
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--particles', type=int, default=100000)
    p.add_argument('--batches', type=int, default=5)
    p.add_argument('--inactive', type=int, default=2)
    p.add_argument('--lattice', type=int, default=17)
    p.add_argument('--shared-bank', type=lambda s: s.lower() in ('1', 'true'),
                   default=True)
    p.add_argument('--surf-write', type=int, default=0,
                   help='max surface-source sites to write (0 = off)')
    build(p.parse_args())
