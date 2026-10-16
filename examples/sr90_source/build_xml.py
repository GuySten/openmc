"""Sr90/Y90 beta source in a steel capsule surrounded by water

Sr90 (28.8 y) decays to Y90 (64 h), and both undergo unique first-forbidden
beta decays. In secular equilibrium, each Sr90 decay is followed by a Y90
decay. The beta spectra are described with openmc.stats.BetaSpectrum using the
shape factor of unique first-forbidden transitions, which gives the measured
average energies (196 keV for Sr90 and 933 keV for Y90); the allowed shape
would underestimate the average energy of Sr90 beta particles by 11%.

Electrons are not transported by OpenMC: their energy is deposited where they
are born, and with photon transport, the bremsstrahlung photons they emit are
transported (thick-target bremsstrahlung approximation).

"""

import openmc

###############################################################################
# Materials

source_material = openmc.Material(name='SrO')
source_material.add_element('Sr', 1.0)
source_material.add_element('O', 1.0)
source_material.set_density('g/cm3', 4.7)

steel = openmc.Material(name='Stainless steel')
steel.add_element('Fe', 0.70, 'wo')
steel.add_element('Cr', 0.19, 'wo')
steel.add_element('Ni', 0.11, 'wo')
steel.set_density('g/cm3', 8.0)

water = openmc.Material(name='Water')
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 1.0)

###############################################################################
# Geometry: source pellet, steel capsule, and a water sphere

pellet_sphere = openmc.Sphere(r=0.1)
capsule_sphere = openmc.Sphere(r=0.15)
water_sphere = openmc.Sphere(r=20.0, boundary_type='vacuum')

pellet = openmc.Cell(fill=source_material, region=-pellet_sphere)
capsule = openmc.Cell(fill=steel, region=+pellet_sphere & -capsule_sphere)
phantom = openmc.Cell(fill=water, region=+capsule_sphere & -water_sphere)
geometry = openmc.Geometry([pellet, capsule, phantom])

###############################################################################
# Source: electrons from Sr90 and Y90 with equal activities

sr90 = openmc.stats.BetaSpectrum(
    endpoint=546.0e3, atomic_number=39, mass_number=90,
    forbiddenness=1, unique=True)
y90 = openmc.stats.BetaSpectrum(
    endpoint=2279.8e3, atomic_number=40, mass_number=90,
    forbiddenness=1, unique=True)
energy = openmc.stats.Mixture([1.0, 1.0], [sr90, y90])

source = openmc.IndependentSource(
    space=openmc.stats.spherical_uniform(r_outer=0.1),
    angle=openmc.stats.Isotropic(),
    energy=energy,
    particle='electron'
)

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.source = source
settings.particles = 100000
settings.batches = 10
settings.photon_transport = True

###############################################################################
# Tallies: energy deposition in each cell and the bremsstrahlung spectrum

cell_filter = openmc.CellFilter([pellet, capsule, phantom])
heating = openmc.Tally(name='heating')
heating.filters = [cell_filter]
heating.scores = ['heating']

photon_filter = openmc.ParticleFilter(['photon'])
energy_filter = openmc.EnergyFilter.from_group_structure('VITAMIN-J-42')
spectrum = openmc.Tally(name='bremsstrahlung spectrum')
spectrum.filters = [photon_filter, openmc.CellFilter([phantom]), energy_filter]
spectrum.scores = ['flux']

model = openmc.Model(geometry=geometry, settings=settings,
                     tallies=[heating, spectrum])
model.export_to_model_xml()
