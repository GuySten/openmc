# Sr90/Y90 Beta Source

This example models a Sr90/Y90 beta source in secular equilibrium, sealed in a
steel capsule and surrounded by water. The beta spectra of both nuclides are
described with `openmc.stats.BetaSpectrum` using the shape factor of unique
first-forbidden transitions.

To build the model, run `python build_xml.py`, and then run `openmc`. The
statepoint file contains the energy deposited in each cell and the spectrum of
bremsstrahlung photons in the water.
